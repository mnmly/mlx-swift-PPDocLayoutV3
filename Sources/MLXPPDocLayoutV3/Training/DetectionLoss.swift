// Set criterion for head-only fine-tuning.
//
// PORT FROM: RT-DETR `RTDetrLoss` (VarifocalLoss classification + L1 box + GIoU),
// with per-decoder-layer auxiliary (deep-supervision) losses, plus PP-DocLayoutV3
// mask and reading-order heads:
//   - classification: VarifocalLoss (IoU-quality targets), per layer.
//   - boxes: L1 + GIoU on matched queries, per layer.
//   - masks: BCE + Dice on matched queries (last layer), when GT masks present.
//   - order: global-pointer multilabel categorical cross-entropy over the
//     pairwise-precedence target (last layer), when GT reading-order present.
//     The precedence target is the inverse of inference `_get_order_seqs`:
//     cell (i,j), i<j, is positive iff region-at-query-i precedes region-at-query-j.
//
// Matching (Hungarian, focal cost) is unchanged by the VFL swap.

import Foundation
import MLX

/// Weights and class count for ``DetectionLoss``. Box weights live in
/// ``MatchCost`` (shared with the matcher); the head loss weights are here.
public struct DetectionLossConfig {
    public var cost: MatchCost
    public var numClasses: Int
    /// VarifocalLoss weight (RT-DETR uses 1.0).
    public var vflWeight: Float
    /// Mask loss weight (applied to BCE + Dice). Skipped when no GT masks.
    public var maskWeight: Float
    /// Reading-order loss weight. Skipped when no GT reading-order.
    public var orderWeight: Float
    public init(
        numClasses: Int, cost: MatchCost = MatchCost(),
        vflWeight: Float = 1, maskWeight: Float = 1, orderWeight: Float = 1
    ) {
        self.numClasses = numClasses
        self.cost = cost
        self.vflWeight = vflWeight
        self.maskWeight = maskWeight
        self.orderWeight = orderWeight
    }
}

/// Breakdown of the (scalar) loss into its components.
public struct LossBreakdown {
    public var classLoss: MLXArray
    public var bboxLoss: MLXArray
    public var giouLoss: MLXArray
    public var maskLoss: MLXArray
    public var orderLoss: MLXArray
    public var total: MLXArray
}

/// Hungarian-matched detection loss with per-layer auxiliary supervision.
public enum DetectionLoss {
    /// Numerically stable element-wise binary cross-entropy with logits.
    /// `ce = relu(x) - x*y + log(1 + exp(-|x|))`.
    static func bceWithLogits(_ logits: MLXArray, _ targets: MLXArray) -> MLXArray {
        maximum(logits, 0) - logits * targets + log(1 + exp(-abs(logits)))
    }

    /// Element-wise sigmoid focal loss (before reduction). Kept as a documented
    /// building block / for tests; the classification loss itself is VFL below.
    static func focalPerElement(_ logits: MLXArray, _ targets: MLXArray, alpha: Float, gamma: Float)
        -> MLXArray
    {
        let prob = sigmoid(logits)
        let ce = bceWithLogits(logits, targets)
        let pT = prob * targets + (1 - prob) * (1 - targets)
        var loss = ce * pow(1 - pT, gamma)
        let alphaT = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alphaT * loss
        return loss
    }

    /// Per-layer, per-image bipartite matches `[layer][image] -> [(query, target)]`.
    /// Matching is non-differentiable (it evaluates a cost matrix and runs the
    /// Hungarian solver on CPU), so it must be computed *outside* any
    /// `valueAndGrad` trace and fed to ``loss(output:targets:matches:config:)``.
    public static func matches(output: DecoderOutput, targets: [Target], cost: MatchCost)
        -> [[[(Int, Int)]]]
    {
        output.interLogits.indices.map { l in
            let logits = output.interLogits[l], boxes = output.interBoxes[l]
            return (0..<logits.dim(0)).map { b in
                HungarianMatcher.match(
                    logits: logits[b], boxes: boxes[b], target: targets[b], cost: cost)
            }
        }
    }

    /// VarifocalLoss for one image at one decoder layer. Returns the unnormalized
    /// sum over (query, class). Targets and weights are detached (constants); only
    /// `logits` carries gradient.
    static func vfl(
        logits: MLXArray, boxes: MLXArray, target: Target, pairs: [(Int, Int)],
        numClasses C: Int, alpha: Float, gamma: Float
    ) -> MLXArray {
        let Q = logits.dim(0)
        var oneHot = [Float](repeating: 0, count: Q * C)
        for (q, t) in pairs { oneHot[q * C + target.labels[t]] = 1 }
        let oneHotT = MLXArray(oneHot, [Q, C]) // (Q,C) constant

        var targetScore = MLXArray.zeros([Q, C])
        if !pairs.isEmpty {
            let Tc = pairs.count
            let qIdx = MLXArray(pairs.map { Int32($0.0) })
            let tIdx = MLXArray(pairs.map { Int32($0.1) })
            let predM = boxes[qIdx]            // (Tc,4)
            let tgtM = target.boxes[tIdx]      // (Tc,4)
            let (iou, _) = BoxOps.boxIoU(BoxOps.cxcywhToXyxy(predM), BoxOps.cxcywhToXyxy(tgtM))
            let r = MLXArray(0..<Tc)
            let diag = stopGradient(iou[r, r]) // (Tc,) IoU quality, detached
            // Scatter matched IoUs to their query rows via a selection matrix.
            var sel = [Float](repeating: 0, count: Tc * Q)
            for (k, p) in pairs.enumerated() { sel[k * Q + p.0] = 1 }
            let S = MLXArray(sel, [Tc, Q])
            let iouQ = matmul(diag.reshaped([1, Tc]), S).reshaped([Q, 1]) // (Q,1)
            targetScore = iouQ * oneHotT
        }

        let predScore = stopGradient(sigmoid(logits)) // detached
        let weight = alpha * pow(predScore, gamma) * (1 - oneHotT) + targetScore
        return (weight * bceWithLogits(logits, targetScore)).sum()
    }

    /// BCE + Dice mask loss for one image (last layer). Returns the unnormalized
    /// sum over matched instances, or `nil` if no GT masks / no matches.
    static func maskLoss(predMasks: MLXArray, target: Target, pairs: [(Int, Int)]) -> MLXArray? {
        guard let gtMasks = target.masks, !pairs.isEmpty else { return nil }
        let qIdx = MLXArray(pairs.map { Int32($0.0) })
        let tIdx = MLXArray(pairs.map { Int32($0.1) })
        let pred = predMasks[qIdx]   // (Tc,mH,mW)
        let gt = gtMasks[tIdx]       // (Tc,mH,mW)

        let bce = bceWithLogits(pred, gt).mean(axes: [1, 2]) // (Tc,)
        let p = sigmoid(pred)
        let num = 2 * (p * gt).sum(axes: [1, 2]) + 1
        let den = p.sum(axes: [1, 2]) + gt.sum(axes: [1, 2]) + 1
        let dice = 1 - num / den                              // (Tc,)
        return (bce + dice).sum()
    }

    /// Global-pointer reading-order loss for one image (last layer). Multilabel
    /// categorical cross-entropy over the pairwise-precedence target for matched
    /// queries. Returns `nil` if no GT order / fewer than two matches.
    static func orderLoss(orderLogits: MLXArray, target: Target, pairs: [(Int, Int)]) -> MLXArray? {
        guard let ranks = target.orderRanks, pairs.count >= 2 else { return nil }
        let Q = orderLogits.dim(0)

        var flatIdx: [Int32] = []
        var tHost: [Float] = []
        for a in 0..<pairs.count {
            for b in (a + 1)..<pairs.count {
                let (qa, ta) = pairs[a], (qb, tb) = pairs[b]
                let i = min(qa, qb), j = max(qa, qb)        // valid cell is j > i
                let regionI = qa < qb ? ta : tb
                let regionJ = qa < qb ? tb : ta
                flatIdx.append(Int32(i * Q + j))
                tHost.append(ranks[regionI] < ranks[regionJ] ? 1 : 0)
            }
        }
        if flatIdx.isEmpty { return nil }

        let y = orderLogits.reshaped([Q * Q])[MLXArray(flatIdx)] // (n,) logits, differentiable
        let t = MLXArray(tHost)                                  // (n,) constant
        let yy = (1 - 2 * t) * y
        let neg = yy - t * 1e12
        let pos = yy - (1 - t) * 1e12
        let zero = MLXArray([Float(0)])
        return logSumExp(concatenated([neg, zero])) + logSumExp(concatenated([pos, zero]))
    }

    /// Differentiable detection loss given precomputed `matches`. Classification
    /// and box terms are summed across all decoder layers (last + aux); mask and
    /// order terms use the last layer only. Contains no `eval`/host transfer of
    /// traced values, so it is safe inside `valueAndGrad`.
    public static func loss(
        output: DecoderOutput, targets: [Target], matches: [[[(Int, Int)]]],
        config: DetectionLossConfig
    ) -> LossBreakdown {
        let numLayers = output.interLogits.count
        let numBoxes = Float(max(1, targets.reduce(0) { $0 + $1.labels.count }))
        let B = output.interLogits[0].dim(0)

        var classTotal = MLXArray(Float(0))
        var bboxTotal = MLXArray(Float(0))
        var giouTotal = MLXArray(Float(0))

        for l in 0..<numLayers {
            let logits = output.interLogits[l] // (B,Q,C)
            let boxes = output.interBoxes[l]   // (B,Q,4) cxcywh
            for b in 0..<B {
                let imgBoxes = boxes[b]
                let target = targets[b]
                let pairs = matches[l][b]

                classTotal = classTotal + vfl(
                    logits: logits[b], boxes: imgBoxes, target: target, pairs: pairs,
                    numClasses: config.numClasses, alpha: config.cost.focalAlpha,
                    gamma: config.cost.focalGamma)

                if !pairs.isEmpty {
                    let qIdx = MLXArray(pairs.map { Int32($0.0) })
                    let tIdx = MLXArray(pairs.map { Int32($0.1) })
                    let predMatched = imgBoxes[qIdx]
                    let tgtMatched = target.boxes[tIdx]
                    bboxTotal = bboxTotal + abs(predMatched - tgtMatched).sum()

                    let giou = BoxOps.generalizedBoxIoU(
                        BoxOps.cxcywhToXyxy(predMatched), BoxOps.cxcywhToXyxy(tgtMatched))
                    let r = MLXArray(0..<pairs.count)
                    giouTotal = giouTotal + (1 - giou[r, r]).sum()
                }
            }
        }

        // Mask and order: last decoder layer only.
        let last = numLayers - 1
        var maskTotal = MLXArray(Float(0))
        var orderTotal = MLXArray(Float(0))
        for b in 0..<B {
            let pairs = matches[last][b]
            if let m = maskLoss(predMasks: output.outMasks[b], target: targets[b], pairs: pairs) {
                maskTotal = maskTotal + m
            }
            if let o = orderLoss(orderLogits: output.orderLogits[b], target: targets[b], pairs: pairs) {
                orderTotal = orderTotal + o
            }
        }

        let classLoss = config.vflWeight * classTotal / numBoxes
        let bboxLoss = config.cost.bboxWeight * bboxTotal / numBoxes
        let giouLoss = config.cost.giouWeight * giouTotal / numBoxes
        let maskLossTerm = config.maskWeight * maskTotal / numBoxes
        let orderLossTerm = config.orderWeight * orderTotal / Float(B)
        return LossBreakdown(
            classLoss: classLoss, bboxLoss: bboxLoss, giouLoss: giouLoss,
            maskLoss: maskLossTerm, orderLoss: orderLossTerm,
            total: classLoss + bboxLoss + giouLoss + maskLossTerm + orderLossTerm)
    }

    /// Convenience: match then compute the loss in one call. Not for use inside
    /// `valueAndGrad` (it matches internally, which evaluates tensors).
    public static func loss(output: DecoderOutput, targets: [Target], config: DetectionLossConfig)
        -> LossBreakdown
    {
        let m = matches(output: output, targets: targets, cost: config.cost)
        return loss(output: output, targets: targets, matches: m, config: config)
    }
}
