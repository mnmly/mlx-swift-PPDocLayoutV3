// Multi-scale deformable attention (grid_sample path, disable_custom_kernels).
//
// PORT FROM: reference/python/modeling_pp_doclayout_v3.py
//   (PPDocLayoutV3MultiscaleDeformableAttention + multi_scale_deformable_attention)
// Reuses GridSampleKernel.swift (bilinear, align_corners=false).

import Foundation
import MLX
import MLXNN

public final class MSDeformableAttention: Module {
    @ModuleInfo(key: "sampling_offsets") var samplingOffsets: Linear
    @ModuleInfo(key: "attention_weights") var attentionWeights: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "output_proj") var outputProj: Linear

    let dModel: Int, nHeads: Int, nLevels: Int, nPoints: Int, headDim: Int

    public init(dModel: Int, nHeads: Int, nLevels: Int, nPoints: Int) {
        self.dModel = dModel; self.nHeads = nHeads
        self.nLevels = nLevels; self.nPoints = nPoints
        self.headDim = dModel / nHeads
        self._samplingOffsets = ModuleInfo(wrappedValue: Linear(dModel, nHeads * nLevels * nPoints * 2), key: "sampling_offsets")
        self._attentionWeights = ModuleInfo(wrappedValue: Linear(dModel, nHeads * nLevels * nPoints), key: "attention_weights")
        self._valueProj = ModuleInfo(wrappedValue: Linear(dModel, dModel), key: "value_proj")
        self._outputProj = ModuleInfo(wrappedValue: Linear(dModel, dModel), key: "output_proj")
        super.init()
    }

    /// - query: (B, Q, d_model) decoder hidden states (position embeddings added here).
    /// - referencePoints: (B, Q, 4) center/wh, normalized [0,1].
    /// - value: (B, S, d_model) encoder hidden states (S = sum Hi*Wi).
    /// - spatialShapes: [(Hi, Wi)] per level.
    public func callAsFunction(
        _ query: MLXArray, positionEmbeddings: MLXArray?, referencePoints: MLXArray,
        value: MLXArray, spatialShapes: [(Int, Int)]
    ) -> MLXArray {
        let B = query.dim(0), Q = query.dim(1)
        let q = positionEmbeddings != nil ? query + positionEmbeddings! : query

        let vProj = valueProj(value).reshaped([B, -1, nHeads, headDim])

        let offsets = samplingOffsets(q).reshaped([B, Q, nHeads, nLevels, nPoints, 2])
        var w = attentionWeights(q).reshaped([B, Q, nHeads, nLevels * nPoints])
        w = softmax(w, axis: -1).reshaped([B, Q, nHeads, nLevels, nPoints])

        // 4-coord reference points: center + offset/nPoints * wh * 0.5  (broadcast levels)
        let refCenter = referencePoints[0..., 0..., 0..<2].reshaped([B, Q, 1, 1, 1, 2])
        let refWH = referencePoints[0..., 0..., 2...].reshaped([B, Q, 1, 1, 1, 2])
        let samplingLocations = refCenter + offsets / Float(nPoints) * refWH * 0.5
        let samplingGrids = 2 * samplingLocations - 1 // [-1, 1]

        var output: MLXArray? = nil
        var offset = 0
        for (lvl, (H, W)) in spatialShapes.enumerated() {
            let sz = H * W
            let vLvl = vProj[0..., offset..<(offset + sz), 0..., 0...]
            let vSpatial = vLvl.reshaped([B, H, W, nHeads, headDim])
                .transposed(0, 3, 1, 2, 4)
                .reshaped([B * nHeads, H, W, headDim])

            let sampLoc = samplingGrids[0..., 0..., 0..., lvl, 0..., 0...] // (B,Q,nHeads,nPoints,2)
            let grid = sampLoc.transposed(0, 2, 1, 3, 4).reshaped([B * nHeads, Q, nPoints, 2])

            let sampled = gridSample([vSpatial, grid])[0] // (B*nHeads, Q, nPoints, headDim)
            let sampledR = sampled.reshaped([B, nHeads, Q, nPoints, headDim])

            let wl = w[0..., 0..., 0..., lvl, 0...]
                .transposed(0, 2, 1, 3) // (B,nHeads,Q,nPoints)
                .expandedDimensions(axis: -1)
            let contrib = (sampledR * wl).sum(axis: 3) // (B,nHeads,Q,headDim)
            output = output.map { $0 + contrib } ?? contrib
            offset += sz
        }

        let out = output!.transposed(0, 2, 1, 3).reshaped([B, Q, dModel])
        return outputProj(out)
    }
}
