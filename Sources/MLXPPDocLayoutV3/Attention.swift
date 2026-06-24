// Shared multi-head self-attention + 2D sinusoidal position embedding.
//
// PORT FROM: reference/python/modeling_pp_doclayout_v3.py
//   (PPDocLayoutV3SelfAttention, build_2d_sinusoidal_position_embedding)

import Foundation
import MLX
import MLXNN
import MLXFast

/// Multi-headed self-attention. Position embeddings (when provided) are added to
/// queries and keys but NOT values. Separate q/k/v/o projections (no fused QKV).
public final class SelfAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var oProj: Linear
    let numHeads: Int
    let headDim: Int
    let scale: Float

    public init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = 1.0 / Float(headDim).squareRoot()
        self._qProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "q_proj")
        self._kProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "k_proj")
        self._vProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "v_proj")
        self._oProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "out_proj")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, positionEmbeddings: MLXArray? = nil) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        let qkIn = positionEmbeddings != nil ? x + positionEmbeddings! : x
        func heads(_ t: MLXArray) -> MLXArray {
            t.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        }
        let q = heads(qProj(qkIn))
        let k = heads(kProj(qkIn))
        let v = heads(vProj(x))
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        let merged = out.transposed(0, 2, 1, 3).reshaped(B, L, numHeads * headDim)
        return oProj(merged)
    }
}

/// 2D sinusoidal position embedding `[sin_h | cos_h | sin_w | cos_w]`,
/// row-major (H-outer). Returns `(1, H*W, embedDim)`.
func sine2DPositionEmbedding(height: Int, width: Int, embedDim: Int, temperature: Float) -> MLXArray {
    let posDim = embedDim / 4
    let omegaIdx = MLXArray(0..<posDim).asType(.float32) / Float(posDim)
    let omega = 1.0 / pow(MLXArray(temperature), omegaIdx) // (posDim,)

    let gridH = MLXArray(0..<height).asType(.float32) // (H,)
    let gridW = MLXArray(0..<width).asType(.float32)   // (W,)
    // meshgrid(ij): gh[h,w]=h, gw[h,w]=w, flattened row-major.
    let ghFlat = broadcast(gridH.reshaped([height, 1]), to: [height, width]).reshaped([height * width])
    let gwFlat = broadcast(gridW.reshaped([1, width]), to: [height, width]).reshaped([height * width])

    let embH = ghFlat.reshaped([height * width, 1]) * omega.reshaped([1, posDim]) // (HW, posDim)
    let embW = gwFlat.reshaped([height * width, 1]) * omega.reshaped([1, posDim])

    let pos = concatenated([sin(embH), cos(embH), sin(embW), cos(embW)], axis: 1) // (HW, embedDim)
    return pos.reshaped([1, height * width, embedDim])
}
