// Hybrid encoder (AIFI + FPN/PAN) with mask-feature head.
//
// PORT FROM: reference/python/modeling_pp_doclayout_v3.py
//   (PPDocLayoutV3MLP, EncoderLayer, AIFILayer, RepVggBlock, CSPRepLayer,
//    ScaleHead, MaskFeatFPN, EncoderMaskOutput, HybridEncoder, ConvLayer)

import Foundation
import MLX
import MLXNN

// MARK: - small reusable pieces

/// Bilinear 2x upsample (align_corners=false), as a parameter-free UnaryLayer so
/// it can sit in a `[UnaryLayer]` matching HF nn.Sequential indices.
final class BilinearUpsample2x: Module, UnaryLayer {
    override init() { super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { bilinearResize(x, x.dim(1) * 2, x.dim(2) * 2) }
}

/// AIFI transformer encoder layer (post-norm). FFN is flat fc1/fc2 (no wrapper),
/// matching the safetensors key layout.
final class EncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SelfAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm
    let act: (MLXArray) -> MLXArray

    init(_ cfg: PPDocLayoutV3Config) {
        let dim = cfg.encoderHiddenDim
        self._selfAttn = ModuleInfo(
            wrappedValue: SelfAttention(hiddenSize: dim, numHeads: cfg.encoderAttentionHeads), key: "self_attn")
        self._selfAttnLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim, eps: cfg.layerNormEps), key: "self_attn_layer_norm")
        self._fc1 = ModuleInfo(wrappedValue: Linear(dim, cfg.encoderFFNDim), key: "fc1")
        self._fc2 = ModuleInfo(wrappedValue: Linear(cfg.encoderFFNDim, dim), key: "fc2")
        self._finalLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim, eps: cfg.layerNormEps), key: "final_layer_norm")
        self.act = activation(cfg.encoderActivationFunction)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray?) -> MLXArray {
        var h = x + selfAttn(x, positionEmbeddings: posEmb)
        h = selfAttnLayerNorm(h)
        h = h + fc2(act(fc1(h)))
        return finalLayerNorm(h)
    }
}

/// AIFI: flatten an NHWC map, run encoder layers with 2D sine pos emb, reshape.
public final class AIFILayer: Module {
    @ModuleInfo(key: "layers") var layers: [EncoderLayer]
    let embedDim: Int
    let temperature: Float

    init(_ cfg: PPDocLayoutV3Config) {
        self.embedDim = cfg.encoderHiddenDim
        self.temperature = Float(cfg.positionalEncodingTemperature)
        var built: [EncoderLayer] = []
        for _ in 0..<cfg.encoderLayers { built.append(EncoderLayer(cfg)) }
        self._layers = ModuleInfo(wrappedValue: built, key: "layers")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), H = x.dim(1), W = x.dim(2), C = x.dim(3)
        var h = x.reshaped([B, H * W, C])
        let pos = sine2DPositionEmbedding(height: H, width: W, embedDim: embedDim, temperature: temperature)
        for layer in layers { h = layer(h, posEmb: pos) }
        return h.reshaped([B, H, W, C])
    }
}

/// RepVGG block: silu(conv3(x) + conv1(x)).
public final class RepVggBlock: Module {
    @ModuleInfo(key: "conv1") public var conv1: ConvNorm
    @ModuleInfo(key: "conv2") var conv2: ConvNorm
    let act: (MLXArray) -> MLXArray
    init(_ cfg: PPDocLayoutV3Config) {
        let hidden = Int(Float(cfg.encoderHiddenDim) * cfg.hiddenExpansion)
        self._conv1 = ModuleInfo(wrappedValue: ConvNorm(hidden, hidden, kernelSize: 3, padding: 1, activation: nil, bnEps: cfg.batchNormEps), key: "conv1")
        self._conv2 = ModuleInfo(wrappedValue: ConvNorm(hidden, hidden, kernelSize: 1, padding: 0, activation: nil, bnEps: cfg.batchNormEps), key: "conv2")
        self.act = activation(cfg.activationFunction)
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { act(conv1(x) + conv2(x)) }
}

/// CSPRepLayer: cross-stage partial with RepVGG bottlenecks.
public final class CSPRepLayer: Module {
    @ModuleInfo(key: "conv1") var conv1: ConvNorm
    @ModuleInfo(key: "conv2") var conv2: ConvNorm
    @ModuleInfo(key: "bottlenecks") public var bottlenecks: [RepVggBlock]
    // conv3 is Identity here (hidden == out), so omitted.

    init(_ cfg: PPDocLayoutV3Config) {
        let inCh = cfg.encoderHiddenDim * 2
        let out = cfg.encoderHiddenDim
        let hidden = Int(Float(out) * cfg.hiddenExpansion)
        let act = cfg.activationFunction
        self._conv1 = ModuleInfo(wrappedValue: ConvNorm(inCh, hidden, kernelSize: 1, activation: act, bnEps: cfg.batchNormEps), key: "conv1")
        self._conv2 = ModuleInfo(wrappedValue: ConvNorm(inCh, hidden, kernelSize: 1, activation: act, bnEps: cfg.batchNormEps), key: "conv2")
        var bn: [RepVggBlock] = []
        for _ in 0..<3 { bn.append(RepVggBlock(cfg)) }
        self._bottlenecks = ModuleInfo(wrappedValue: bn, key: "bottlenecks")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h1 = conv1(x)
        for b in bottlenecks { h1 = b(h1) }
        let h2 = conv2(x)
        return h1 + h2
    }
}

/// One scale head of the mask-feature FPN: stacked 3x3 convs + bilinear upsamples.
final class ScaleHead: Module {
    @ModuleInfo(key: "layers") var layers: [UnaryLayer]
    init(inChannels: Int, featureChannels: Int, fpnStride: Int, baseStride: Int) {
        let headLength = max(1, Int(log2(Double(fpnStride)) - log2(Double(baseStride))))
        var built: [UnaryLayer] = []
        for k in 0..<headLength {
            let inC = k == 0 ? inChannels : featureChannels
            built.append(ConvNorm(inC, featureChannels, kernelSize: 3, activation: "silu"))
            if fpnStride != baseStride { built.append(BilinearUpsample2x()) }
        }
        self._layers = ModuleInfo(wrappedValue: built, key: "layers")
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for l in layers { h = l(h) }
        return h
    }
}

/// Mask-feature FPN: per-level scale heads summed at the finest resolution.
final class MaskFeatFPN: Module {
    @ModuleInfo(key: "scale_heads") var scaleHeads: [ScaleHead]
    @ModuleInfo(key: "output_conv") var outputConv: ConvNorm
    let strides: [Int]

    init(inChannels: [Int], fpnStrides: [Int], featureChannels: Int, outChannels: Int) {
        // reorder_index = argsort(fpn_strides); our strides [8,16,32] already ascending.
        let order = Array(fpnStrides.indices).sorted { fpnStrides[$0] < fpnStrides[$1] }
        let inCh = order.map { inChannels[$0] }
        let strs = order.map { fpnStrides[$0] }
        self.strides = strs
        var heads: [ScaleHead] = []
        for i in 0..<strs.count {
            heads.append(ScaleHead(inChannels: inCh[i], featureChannels: featureChannels,
                                   fpnStride: strs[i], baseStride: strs[0]))
        }
        self._scaleHeads = ModuleInfo(wrappedValue: heads, key: "scale_heads")
        self._outputConv = ModuleInfo(wrappedValue: ConvNorm(featureChannels, outChannels, kernelSize: 3, activation: "silu"), key: "output_conv")
        super.init()
    }

    // inputs already ordered ascending by stride (pan maps: stride8, 16, 32).
    func callAsFunction(_ inputs: [MLXArray]) -> MLXArray {
        var output = scaleHeads[0](inputs[0])
        let oh = output.dim(1), ow = output.dim(2)
        for i in 1..<strides.count {
            output = output + bilinearResize(scaleHeads[i](inputs[i]), oh, ow)
        }
        return outputConv(output)
    }
}

/// 3x3 conv (silu) + 1x1 conv (bias) producing prototype masks.
final class EncoderMaskOutput: Module {
    @ModuleInfo(key: "base_conv") var baseConv: ConvNorm
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(inChannels: Int, numPrototypes: Int) {
        self._baseConv = ModuleInfo(wrappedValue: ConvNorm(inChannels, inChannels, kernelSize: 3, activation: "silu"), key: "base_conv")
        self._conv = ModuleInfo(wrappedValue: Conv2d(inputChannels: inChannels, outputChannels: numPrototypes, kernelSize: .init(1), bias: true), key: "conv")
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(baseConv(x)) }
}

/// Output of the hybrid encoder.
public struct HybridEncoderOutput {
    public var lastHiddenState: [MLXArray] // 3 PAN maps (NHWC), stride 8/16/32
    public var maskFeat: MLXArray          // (B, 200, 200, numPrototypes)
}

/// RT-DETR hybrid encoder with the PP-DocLayoutV3 mask head.
public final class HybridEncoder: Module {
    // Safetensors names the AIFI module list `encoder` (model.encoder.encoder.*).
    @ModuleInfo(key: "encoder") public var aifi: [AIFILayer]
    @ModuleInfo(key: "lateral_convs") var lateralConvs: [ConvNorm]
    @ModuleInfo(key: "fpn_blocks") public var fpnBlocks: [CSPRepLayer]
    @ModuleInfo(key: "downsample_convs") var downsampleConvs: [ConvNorm]
    @ModuleInfo(key: "pan_blocks") var panBlocks: [CSPRepLayer]
    @ModuleInfo(key: "mask_feature_head") var maskFeatureHead: MaskFeatFPN
    @ModuleInfo(key: "encoder_mask_lateral") var encoderMaskLateral: ConvNorm
    @ModuleInfo(key: "encoder_mask_output") var encoderMaskOutput: EncoderMaskOutput

    let encodeProjLayers: [Int]
    let numFpnStages: Int
    let encoderLayers: Int

    public init(_ cfg: PPDocLayoutV3Config) {
        let dim = cfg.encoderHiddenDim
        let numLevels = cfg.encoderInChannels.count
        self.encodeProjLayers = cfg.encodeProjLayers
        self.numFpnStages = numLevels - 1
        self.encoderLayers = cfg.encoderLayers

        var aifis: [AIFILayer] = []
        for _ in 0..<cfg.encodeProjLayers.count { aifis.append(AIFILayer(cfg)) }
        self._aifi = ModuleInfo(wrappedValue: aifis, key: "encoder")

        var lat: [ConvNorm] = []; var fpn: [CSPRepLayer] = []
        for _ in 0..<numFpnStages {
            lat.append(ConvNorm(dim, dim, kernelSize: 1, activation: cfg.activationFunction, bnEps: cfg.batchNormEps))
            fpn.append(CSPRepLayer(cfg))
        }
        self._lateralConvs = ModuleInfo(wrappedValue: lat, key: "lateral_convs")
        self._fpnBlocks = ModuleInfo(wrappedValue: fpn, key: "fpn_blocks")

        var down: [ConvNorm] = []; var pan: [CSPRepLayer] = []
        for _ in 0..<numFpnStages {
            down.append(ConvNorm(dim, dim, kernelSize: 3, stride: 2, activation: cfg.activationFunction, bnEps: cfg.batchNormEps))
            pan.append(CSPRepLayer(cfg))
        }
        self._downsampleConvs = ModuleInfo(wrappedValue: down, key: "downsample_convs")
        self._panBlocks = ModuleInfo(wrappedValue: pan, key: "pan_blocks")

        self._maskFeatureHead = ModuleInfo(
            wrappedValue: MaskFeatFPN(inChannels: Array(repeating: dim, count: cfg.featStrides.count),
                                      fpnStrides: cfg.featStrides,
                                      featureChannels: cfg.maskFeatureChannels[0],
                                      outChannels: cfg.maskFeatureChannels[1]),
            key: "mask_feature_head")
        self._encoderMaskLateral = ModuleInfo(
            wrappedValue: ConvNorm(cfg.x4FeatDim, cfg.maskFeatureChannels[1], kernelSize: 3, activation: "silu"),
            key: "encoder_mask_lateral")
        self._encoderMaskOutput = ModuleInfo(
            wrappedValue: EncoderMaskOutput(inChannels: cfg.maskFeatureChannels[1], numPrototypes: cfg.numPrototypes),
            key: "encoder_mask_output")
        super.init()
    }

    /// `featureMaps`: 3 projected backbone features (NHWC), stride 8/16/32.
    /// `x4Feat`: stride-4 backbone feature (NHWC), 128 channels.
    public func callAsFunction(_ featureMaps: [MLXArray], x4Feat: MLXArray) -> HybridEncoderOutput {
        var feats = featureMaps

        // AIFI on selected levels.
        if encoderLayers > 0 {
            for (i, encInd) in encodeProjLayers.enumerated() {
                feats[encInd] = aifi[i](feats[encInd])
            }
        }

        // top-down FPN
        var fpnFeatureMaps = [feats[feats.count - 1]]
        for idx in 0..<numFpnStages {
            let backboneFeat = feats[numFpnStages - idx - 1]
            var top = fpnFeatureMaps[fpnFeatureMaps.count - 1]
            top = lateralConvs[idx](top)
            fpnFeatureMaps[fpnFeatureMaps.count - 1] = top
            let upsampled = nearestUpsample2x(top)
            let fused = concatenated([upsampled, backboneFeat], axis: -1)
            fpnFeatureMaps.append(fpnBlocks[idx](fused))
        }
        fpnFeatureMaps.reverse()

        // bottom-up PAN
        var panFeatureMaps = [fpnFeatureMaps[0]]
        for idx in 0..<numFpnStages {
            let top = panFeatureMaps[panFeatureMaps.count - 1]
            let fpnFeat = fpnFeatureMaps[idx + 1]
            let downsampled = downsampleConvs[idx](top)
            let fused = concatenated([downsampled, fpnFeat], axis: -1)
            panFeatureMaps.append(panBlocks[idx](fused))
        }

        // mask feature head
        var maskFeat = maskFeatureHead(panFeatureMaps)
        maskFeat = bilinearResize(maskFeat, maskFeat.dim(1) * 2, maskFeat.dim(2) * 2)
        maskFeat = maskFeat + encoderMaskLateral(x4Feat)
        maskFeat = encoderMaskOutput(maskFeat)

        return HybridEncoderOutput(lastHiddenState: panFeatureMaps, maskFeat: maskFeat)
    }
}
