// Bipartite matching between decoder queries and ground-truth boxes.
//
// PORT FROM: RT-DETR `HungarianMatcher` (use_focal_loss=True). The matching
// cost combines a focal classification term, an L1 box term, and a GIoU term;
// the optimal assignment is solved on CPU with the O(n³) Hungarian algorithm
// (no MLX/linear_sum_assignment equivalent exists).

import Foundation
import MLX

/// Ground truth for one image. `labels` and `boxes` are always required (used by
/// the matcher and the classification/box losses). `masks` and `orderRanks` are
/// optional supervision for the mask and reading-order heads — when absent, those
/// loss terms are skipped.
public struct Target {
    public var labels: [Int]      // (T,)
    public var boxes: MLXArray    // (T, 4) cxcywh in [0,1]
    /// Per-instance GT masks `(T, mH, mW)` at mask-feature resolution, values in {0,1}.
    public var masks: MLXArray?
    /// Per-instance reading-order rank `(T,)`, 0 = read first. Used to build the
    /// pairwise-precedence target for the global-pointer head.
    public var orderRanks: [Int]?
    public init(labels: [Int], boxes: MLXArray, masks: MLXArray? = nil, orderRanks: [Int]? = nil) {
        self.labels = labels
        self.boxes = boxes
        self.masks = masks
        self.orderRanks = orderRanks
    }
}

/// Cost weights for the matcher and the loss (RT-DETR defaults). These are not
/// present in ``PPDocLayoutV3Config``, so they are carried here explicitly.
public struct MatchCost: Sendable {
    public var classWeight: Float = 2.0
    public var bboxWeight: Float = 5.0
    public var giouWeight: Float = 2.0
    public var focalAlpha: Float = 0.25
    public var focalGamma: Float = 2.0
    public init() {}
}

/// Hungarian (Kuhn–Munkres) matcher producing `(queryIndex, targetIndex)` pairs.
public enum HungarianMatcher {
    /// Match one image's queries to its targets.
    /// - Parameters:
    ///   - logits: `(Q, numClasses)` raw class logits.
    ///   - boxes: `(Q, 4)` predicted cxcywh boxes in `[0,1]`.
    ///   - target: ground-truth labels + cxcywh boxes for this image.
    ///   - cost: cost weights.
    /// - Returns: `T` pairs, one per target, each `(queryIndex, targetIndex)`.
    public static func match(logits: MLXArray, boxes: MLXArray, target: Target, cost: MatchCost)
        -> [(Int, Int)]
    {
        let T = target.labels.count
        let Q = logits.dim(0)
        if T == 0 { return [] }

        let prob = sigmoid(logits) // (Q, C)
        // Focal classification cost gathered at the target class columns.
        let posAll = cost.focalAlpha * pow(1 - prob, cost.focalGamma) * (-log(clip(prob, min: 1e-8)))
        let negAll = (1 - cost.focalAlpha) * pow(prob, cost.focalGamma) * (-log(clip(1 - prob, min: 1e-8)))
        let tgtIds = MLXArray(target.labels.map { Int32($0) })
        let posCost = takeAlong(posAll, broadcast(tgtIds.reshaped([1, T]), to: [Q, T]), axis: 1)
        let negCost = takeAlong(negAll, broadcast(tgtIds.reshaped([1, T]), to: [Q, T]), axis: 1)
        let costClass = posCost - negCost // (Q, T)

        // L1 box cost in cxcywh.
        let predB = boxes.expandedDimensions(axis: 1)        // (Q,1,4)
        let tgtB = target.boxes.expandedDimensions(axis: 0)  // (1,T,4)
        let costBbox = abs(predB - tgtB).sum(axis: -1)       // (Q,T)

        // GIoU cost in xyxy.
        let costGiou = -BoxOps.generalizedBoxIoU(
            BoxOps.cxcywhToXyxy(boxes), BoxOps.cxcywhToXyxy(target.boxes)) // (Q,T)

        let C = cost.bboxWeight * costBbox + cost.classWeight * costClass + cost.giouWeight * costGiou
        // Solve with targets as rows, queries as columns (T <= Q satisfies rows <= cols).
        let costTQ = C.transposed(1, 0) // (T, Q)
        eval(costTQ)
        let flat = costTQ.asArray(Float.self) // row-major (T, Q)

        let rowToCol = hungarian(cost: flat, rows: T, cols: Q) // rowToCol[t] = query index
        return (0..<T).map { (rowToCol[$0], $0) }
    }

    /// Solve the rectangular assignment problem minimizing total cost. Assigns
    /// each of the `rows` rows to a distinct column; requires `rows <= cols`.
    /// `cost` is row-major `rows × cols`.
    /// - Returns: `rows`-length array where element `i` is the column matched to row `i`.
    static func hungarian(cost: [Float], rows n: Int, cols m: Int) -> [Int] {
        if n == 0 { return [] }
        precondition(n <= m, "Hungarian requires rows (\(n)) <= cols (\(m))")
        let inf = Float.greatestFiniteMagnitude

        // 1-indexed potential-method Hungarian (cp-algorithms). u over rows, v over cols.
        var u = [Float](repeating: 0, count: n + 1)
        var v = [Float](repeating: 0, count: m + 1)
        var p = [Int](repeating: 0, count: m + 1) // p[j] = row matched to column j (0 = none)
        var way = [Int](repeating: 0, count: m + 1)

        func a(_ i: Int, _ j: Int) -> Float { cost[(i - 1) * m + (j - 1)] }

        for i in 1...n {
            p[0] = i
            var j0 = 0
            var minv = [Float](repeating: inf, count: m + 1)
            var used = [Bool](repeating: false, count: m + 1)
            repeat {
                used[j0] = true
                let i0 = p[j0]
                var delta = inf
                var j1 = 0
                for j in 1...m where !used[j] {
                    let cur = a(i0, j) - u[i0] - v[j]
                    if cur < minv[j] { minv[j] = cur; way[j] = j0 }
                    if minv[j] < delta { delta = minv[j]; j1 = j }
                }
                for j in 0...m {
                    if used[j] { u[p[j]] += delta; v[j] -= delta }
                    else { minv[j] -= delta }
                }
                j0 = j1
            } while p[j0] != 0
            repeat {
                let j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
            } while j0 != 0
        }

        // Invert p (column→row) to row→column, 0-indexed.
        var rowToCol = [Int](repeating: -1, count: n)
        for j in 1...m where p[j] != 0 { rowToCol[p[j] - 1] = j - 1 }
        return rowToCol
    }
}
