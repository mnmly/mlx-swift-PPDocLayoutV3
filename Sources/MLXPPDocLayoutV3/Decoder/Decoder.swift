// Deformable decoder with iterative bbox refinement + per-layer mask/order/class.
//
// PORT FROM: reference/python/modeling_pp_doclayout_v3.py
//   (PPDocLayoutV3DecoderLayer, PPDocLayoutV3Decoder, PPDocLayoutV3GlobalPointer,
//    inverse_sigmoid)

import Foundation
import MLX
import MLXNN

func inverseSigmoid(_ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let c = clip(x, min: 0, max: 1)
    let x1 = clip(c, min: eps)
    let x2 = clip(1 - c, min: eps)
    return log(x1 / x2)
}

/// Reading-order global pointer head (causal-masked bilinear scores).
public final class GlobalPointer: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    let headSize: Int

    public init(dModel: Int, headSize: Int) {
        self.headSize = headSize
        self._dense = ModuleInfo(wrappedValue: Linear(dModel, headSize * 2), key: "dense")
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        let B = inputs.dim(0), L = inputs.dim(1)
        let qk = dense(inputs).reshaped([B, L, 2, headSize])
        let q = qk[0..., 0..., 0, 0...] // (B,L,headSize)
        let k = qk[0..., 0..., 1, 0...]
        var logits = matmul(q, k.transposed(0, 2, 1)) / Float(headSize).squareRoot()
        // mask = tril(ones(L,L)) (incl. diagonal) → those positions set to -1e4
        let rows = MLXArray(0..<L).reshaped([L, 1])
        let cols = MLXArray(0..<L).reshaped([1, L])
        let trilMask = (cols .<= rows).reshaped([1, L, L]) // lower triangle incl diagonal
        logits = MLX.where(trilMask, MLXArray(Float(-1e4)), logits)
        return logits
    }
}

/// One decoder layer: self-attn → deformable cross-attn → FFN (all post-norm).
final class DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SelfAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "encoder_attn") var encoderAttn: MSDeformableAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm
    let act: (MLXArray) -> MLXArray

    init(_ cfg: PPDocLayoutV3Config) {
        let dim = cfg.dModel
        self._selfAttn = ModuleInfo(wrappedValue: SelfAttention(hiddenSize: dim, numHeads: cfg.decoderAttentionHeads), key: "self_attn")
        self._selfAttnLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim, eps: cfg.layerNormEps), key: "self_attn_layer_norm")
        self._encoderAttn = ModuleInfo(
            wrappedValue: MSDeformableAttention(dModel: dim, nHeads: cfg.decoderAttentionHeads, nLevels: cfg.numFeatureLevels, nPoints: cfg.decoderNPoints),
            key: "encoder_attn")
        self._encoderAttnLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim, eps: cfg.layerNormEps), key: "encoder_attn_layer_norm")
        self._fc1 = ModuleInfo(wrappedValue: Linear(dim, cfg.decoderFFNDim), key: "fc1")
        self._fc2 = ModuleInfo(wrappedValue: Linear(cfg.decoderFFNDim, dim), key: "fc2")
        self._finalLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim, eps: cfg.layerNormEps), key: "final_layer_norm")
        self.act = activation(cfg.decoderActivationFunction)
        super.init()
    }

    func callAsFunction(
        _ hidden: MLXArray, posEmb: MLXArray, referencePoints: MLXArray,
        value: MLXArray, spatialShapes: [(Int, Int)]
    ) -> MLXArray {
        var h = hidden + selfAttn(hidden, positionEmbeddings: posEmb)
        h = selfAttnLayerNorm(h)
        h = h + encoderAttn(h, positionEmbeddings: posEmb, referencePoints: referencePoints, value: value, spatialShapes: spatialShapes)
        h = encoderAttnLayerNorm(h)
        h = h + fc2(act(fc1(h)))
        return finalLayerNorm(h)
    }
}

/// Output of the decoder. `logits`/`predBoxes`/`orderLogits`/`outMasks` hold the
/// last layer (sufficient for `ForObjectDetection` inference). `interLogits` and
/// `interBoxes` hold the per-layer class logits / cxcywh boxes, used only by the
/// training set criterion for auxiliary (deep-supervision) losses; they are lazy
/// and impose no cost on the inference path that ignores them.
public struct DecoderOutput {
    public var logits: MLXArray       // (B, Q, numLabels)
    public var predBoxes: MLXArray    // (B, Q, 4) cxcywh
    public var orderLogits: MLXArray  // (B, Q, Q)
    public var outMasks: MLXArray     // (B, Q, mH, mW)
    public var interLogits: [MLXArray] // per layer: (B, Q, numLabels)
    public var interBoxes: [MLXArray]  // per layer: (B, Q, 4) cxcywh
}

public final class Decoder: Module {
    @ModuleInfo(key: "layers") var layers: [DecoderLayer]
    @ModuleInfo(key: "query_pos_head") var queryPosHead: MLPPredictionHead
    @ModuleInfo(key: "class_embed") var classEmbed: Linear
    @ModuleInfo(key: "bbox_embed") var bboxEmbed: MLPPredictionHead

    public init(_ cfg: PPDocLayoutV3Config) {
        var built: [DecoderLayer] = []
        for _ in 0..<cfg.decoderLayers { built.append(DecoderLayer(cfg)) }
        self._layers = ModuleInfo(wrappedValue: built, key: "layers")
        self._queryPosHead = ModuleInfo(wrappedValue: MLPPredictionHead(4, 2 * cfg.dModel, cfg.dModel, numLayers: 2), key: "query_pos_head")
        self._classEmbed = ModuleInfo(wrappedValue: Linear(cfg.dModel, cfg.numLabels), key: "class_embed")
        self._bboxEmbed = ModuleInfo(wrappedValue: MLPPredictionHead(cfg.dModel, cfg.dModel, 4, numLayers: 3), key: "bbox_embed")
        super.init()
    }

    /// - target: (B, Q, d_model) initial query features.
    /// - value: (B, S, d_model) flattened encoder features.
    /// - referencePointsUnact: (B, Q, 4) pre-sigmoid reference boxes.
    /// - maskFeat: (B, mH, mW, numPrototypes).
    public func callAsFunction(
        target: MLXArray, value: MLXArray, referencePointsUnact: MLXArray,
        spatialShapes: [(Int, Int)],
        orderHead: [Linear], globalPointer: GlobalPointer,
        maskQueryHead: MLPPredictionHead, norm: LayerNorm, maskFeat: MLXArray
    ) -> DecoderOutput {
        let B = target.dim(0), Q = target.dim(1)
        let mH = maskFeat.dim(1), mW = maskFeat.dim(2), P = maskFeat.dim(3)
        let maskFeatFlat = maskFeat.reshaped([B, mH * mW, P]).transposed(0, 2, 1) // (B,P,mH*mW)

        var hidden = target
        var referencePoints = sigmoid(referencePointsUnact)
        var lastLogits = MLXArray.zeros([B, Q, 1])
        var lastRef = referencePoints
        var lastOrder = MLXArray.zeros([B, Q, Q])
        var lastMask = MLXArray.zeros([B, Q, mH, mW])
        var interLogits: [MLXArray] = []
        var interBoxes: [MLXArray] = []

        for (idx, layer) in layers.enumerated() {
            let posEmb = queryPosHead(referencePoints)
            hidden = layer(hidden, posEmb: posEmb, referencePoints: referencePoints, value: value, spatialShapes: spatialShapes)

            let predictedCorners = bboxEmbed(hidden)
            referencePoints = sigmoid(predictedCorners + inverseSigmoid(referencePoints))
            lastRef = referencePoints

            let outQuery = norm(hidden)
            let maskQueryEmbed = maskQueryHead(outQuery) // (B,Q,P)
            lastMask = matmul(maskQueryEmbed, maskFeatFlat).reshaped([B, Q, mH, mW])
            lastLogits = classEmbed(outQuery)
            lastOrder = globalPointer(orderHead[idx](outQuery))

            interLogits.append(lastLogits)
            interBoxes.append(referencePoints)
        }

        return DecoderOutput(logits: lastLogits, predBoxes: lastRef, orderLogits: lastOrder,
                             outMasks: lastMask, interLogits: interLogits, interBoxes: interBoxes)
    }
}
