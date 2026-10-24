// Bounding-box utilities for the training set criterion.
//
// Boxes are normalized to [0, 1]. The decoder emits `cxcywh`; IoU/GIoU are
// computed in `xyxy`. Mirrors the box ops used by the original RT-DETR
// `HungarianMatcher` / `SetCriterion`.

import Foundation
import MLX

/// Box conversions and pairwise (generalized) IoU, in normalized [0, 1] coords.
public enum BoxOps {
    /// Convert `(cx, cy, w, h)` boxes to `(x1, y1, x2, y2)`. Last axis is size 4.
    public static func cxcywhToXyxy(_ boxes: MLXArray) -> MLXArray {
        let cx = boxes[.ellipsis, 0], cy = boxes[.ellipsis, 1]
        let w = boxes[.ellipsis, 2], h = boxes[.ellipsis, 3]
        let x1 = cx - w * 0.5, y1 = cy - h * 0.5
        let x2 = cx + w * 0.5, y2 = cy + h * 0.5
        return stacked([x1, y1, x2, y2], axis: -1)
    }

    /// Convert `(x1, y1, x2, y2)` boxes to `(cx, cy, w, h)`. Last axis is size 4.
    public static func xyxyToCxcywh(_ boxes: MLXArray) -> MLXArray {
        let x1 = boxes[.ellipsis, 0], y1 = boxes[.ellipsis, 1]
        let x2 = boxes[.ellipsis, 2], y2 = boxes[.ellipsis, 3]
        let cx = (x1 + x2) * 0.5, cy = (y1 + y2) * 0.5
        let w = x2 - x1, h = y2 - y1
        return stacked([cx, cy, w, h], axis: -1)
    }

    /// Pairwise IoU and union between two sets of `xyxy` boxes.
    /// - Parameters:
    ///   - boxes1: `(N, 4)`.
    ///   - boxes2: `(M, 4)`.
    /// - Returns: `(iou, union)` each `(N, M)`.
    static func boxIoU(_ boxes1: MLXArray, _ boxes2: MLXArray) -> (MLXArray, MLXArray) {
        let area1 = boxArea(boxes1).expandedDimensions(axis: 1) // (N,1)
        let area2 = boxArea(boxes2).expandedDimensions(axis: 0) // (1,M)

        let b1 = boxes1.expandedDimensions(axis: 1) // (N,1,4)
        let b2 = boxes2.expandedDimensions(axis: 0) // (1,M,4)

        let lt = maximum(b1[.ellipsis, ..<2], b2[.ellipsis, ..<2]) // (N,M,2)
        let rb = minimum(b1[.ellipsis, 2...], b2[.ellipsis, 2...]) // (N,M,2)
        let wh = clip(rb - lt, min: 0)                            // (N,M,2)
        let inter = wh[.ellipsis, 0] * wh[.ellipsis, 1]           // (N,M)

        let union = area1 + area2 - inter
        let iou = inter / clip(union, min: 1e-7)
        return (iou, union)
    }

    /// Pairwise generalized IoU between two sets of `xyxy` boxes.
    /// - Parameters:
    ///   - boxes1: `(N, 4)`.
    ///   - boxes2: `(M, 4)`.
    /// - Returns: `(N, M)` GIoU in `[-1, 1]`.
    public static func generalizedBoxIoU(_ boxes1: MLXArray, _ boxes2: MLXArray) -> MLXArray {
        let (iou, union) = boxIoU(boxes1, boxes2)

        let b1 = boxes1.expandedDimensions(axis: 1) // (N,1,4)
        let b2 = boxes2.expandedDimensions(axis: 0) // (1,M,4)
        let lt = minimum(b1[.ellipsis, ..<2], b2[.ellipsis, ..<2]) // (N,M,2)
        let rb = maximum(b1[.ellipsis, 2...], b2[.ellipsis, 2...]) // (N,M,2)
        let wh = clip(rb - lt, min: 0)                            // (N,M,2)
        let areaEnclosing = wh[.ellipsis, 0] * wh[.ellipsis, 1]   // (N,M)

        return iou - (areaEnclosing - union) / clip(areaEnclosing, min: 1e-7)
    }

    /// Area of each `xyxy` box. Input `(..., 4)`, output `(...)`.
    static func boxArea(_ boxes: MLXArray) -> MLXArray {
        let w = boxes[.ellipsis, 2] - boxes[.ellipsis, 0]
        let h = boxes[.ellipsis, 3] - boxes[.ellipsis, 1]
        return clip(w, min: 0) * clip(h, min: 0)
    }
}
