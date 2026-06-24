// Post-processing: decode raw model outputs into detections + reading order.
//
// PORT FROM: reference/python/modular_pp_doclayout_v3.py
//   (post_process_object_detection, _get_order_seqs)
// Polygon extraction (cv2 findContours/approxPolyDP) lives in Polygon.swift.

import Foundation
import MLX

/// A single decoded detection.
public struct Detection: Sendable {
    public var score: Float
    public var label: Int
    /// xyxy box in target-image pixel coordinates.
    public var box: (Float, Float, Float, Float)
    public var orderRank: Int
    public var polygon: [(Float, Float)]
    var queryIndex: Int
}

public enum PostProcess {
    /// Reading-order sequence per query: rank of each query in reading order.
    /// `orderLogits`: (B, Q, Q). Returns (B, Q) Int ranks.
    static func orderSeqs(_ orderLogits: MLXArray) -> MLXArray {
        let s = sigmoid(orderLogits)             // (B,Q,Q)
        let B = s.dim(0), Q = s.dim(1)
        // votes = sum_i triu(s, diag=1)[i, :] + sum_i (1 - s^T)[i,:] tril(diag=-1)
        let rows = MLXArray(0..<Q).reshaped([Q, 1])
        let cols = MLXArray(0..<Q).reshaped([1, Q])
        let triuMask = (cols .> rows).reshaped([1, Q, Q])   // strictly upper
        let trilMask = (cols .< rows).reshaped([1, Q, Q])   // strictly lower
        let upper = MLX.where(triuMask, s, MLXArray(Float(0)))
        let lowerTerm = MLX.where(trilMask, 1 - s.transposed(0, 2, 1), MLXArray(Float(0)))
        let votes = upper.sum(axis: 1) + lowerTerm.sum(axis: 1) // (B,Q)
        let pointers = argSort(votes, axis: 1)                  // ascending
        // order_seq[pointers[i]] = i  → inverse permutation
        var seq = MLXArray.zeros([B, Q], dtype: .int32)
        let ranks = broadcast(MLXArray(0..<Q).asType(.int32).reshaped([1, Q]), to: [B, Q])
        seq = scatterInverse(pointers: pointers, ranks: ranks, B: B, Q: Q)
        return seq
    }

    /// Build inverse permutation: out[b, pointers[b,i]] = ranks[b,i].
    private static func scatterInverse(pointers: MLXArray, ranks: MLXArray, B: Int, Q: Int) -> MLXArray {
        // CPU scatter (Q is small, 300).
        eval(pointers, ranks)
        let p = pointers.asArray(Int32.self)
        let r = ranks.asArray(Int32.self)
        var out = [Int32](repeating: 0, count: B * Q)
        for b in 0..<B {
            for i in 0..<Q {
                out[b * Q + Int(p[b * Q + i])] = r[b * Q + i]
            }
        }
        return MLXArray(out, [B, Q])
    }

    /// Decode detections for a single image (B=1).
    /// - targetSize: (height, width) of the original image in pixels.
    /// - inputSize: model input side (800); used for the mask→box scale ratio.
    /// - polygons: when true, extract mask polygons (slower CPU work).
    public static func decode(
        output: PPDocLayoutV3Output, targetSize: (Int, Int), threshold: Float = 0.5,
        numClasses: Int, inputSize: Int = 800, polygons: Bool = true
    ) -> [Detection] {
        let logits = output.logits            // (1,Q,C)
        let boxes = output.predBoxes          // (1,Q,4) cxcywh
        let Q = logits.dim(1)
        let (imgH, imgW) = targetSize

        // cxcywh → xyxy, scaled to target pixels.
        let centers = boxes[0..., 0..., 0..<2]
        let dims = boxes[0..., 0..., 2..<4]
        let tl = centers - 0.5 * dims
        let br = centers + 0.5 * dims
        var xyxy = concatenated([tl, br], axis: -1) // (1,Q,4) normalized
        let scale = MLXArray([Float(imgW), Float(imgH), Float(imgW), Float(imgH)]).reshaped([1, 1, 4])
        xyxy = xyxy * scale

        let orderSeq = orderSeqs(output.orderLogits) // (1,Q)

        // topk over flattened (Q*C) scores.
        let scores = sigmoid(logits).reshaped([Q * numClasses])
        let order = argSort(-scores)                 // descending
        let topIdx = order[..<Q]                      // (Q,)
        eval(scores, topIdx, xyxy, orderSeq)

        let scoreArr = scores.asArray(Float.self)
        let idxArr = topIdx.asArray(Int32.self)
        let boxArr = xyxy.reshaped([Q, 4]).asArray(Float.self)
        let orderArr = orderSeq.reshaped([Q]).asArray(Int32.self)

        // collect over-threshold detections (query index = idx / C)
        var dets: [Detection] = []
        for k in 0..<Q {
            let flat = Int(idxArr[k])
            let score = scoreArr[flat]
            if score < threshold { continue }
            let label = flat % numClasses
            let qi = flat / numClasses
            let b = (boxArr[qi * 4], boxArr[qi * 4 + 1], boxArr[qi * 4 + 2], boxArr[qi * 4 + 3])
            dets.append(Detection(score: score, label: label, box: b,
                                  orderRank: Int(orderArr[qi]), polygon: [], queryIndex: qi))
        }
        // sort by reading-order rank.
        dets.sort { $0.orderRank < $1.orderRank }

        guard polygons, !dets.isEmpty else { return dets }

        // Gather masks for detected queries, binarize, extract polygons.
        let masks = output.outMasks                 // (1, Q, mH, mW)
        let mH = masks.dim(2), mW = masks.dim(3)
        let qis = dets.map { $0.queryIndex }
        let idxArrQ = MLXArray(qis.map { Int32($0) })
        let gathered = take(masks.reshaped([Q, mH, mW]), idxArrQ, axis: 0) // (N, mH, mW)
        let binar = (sigmoid(gathered) .> threshold)
        eval(binar)
        let flatBool = binar.asType(.int32).asArray(Int32.self)
        var maskList: [[Bool]] = []
        let plane = mH * mW
        for n in 0..<dets.count {
            var m = [Bool](repeating: false, count: plane)
            for j in 0..<plane { m[j] = flatBool[n * plane + j] != 0 }
            maskList.append(m)
        }
        let scaleRatio = (Float(inputSize) / Float(imgW), Float(inputSize) / Float(imgH))
        let boxesForPoly = dets.map { $0.box }
        let polys = Polygon.extractPolygons(boxes: boxesForPoly, masks: maskList,
                                            maskW: mW, maskH: mH, scaleRatio: scaleRatio)
        for i in 0..<dets.count { dets[i].polygon = polys[i] }
        return dets
    }
}
