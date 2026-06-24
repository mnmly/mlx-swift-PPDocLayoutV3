// HGNetV2-L backbone (NHWC). Returns 4 feature maps (strides 4, 8, 16, 32).
//
// PORT FROM: transformers/models/hgnet_v2/modeling_hgnet_v2.py
//   (HGNetV2Embeddings, HGNetV2ConvLayerLight, HGNetV2BasicLayer,
//    HGNetV2Stage, HGNetV2Encoder, HGNetV2Backbone)
// use_learnable_affine_block=False → no `lab`; hidden_act=relu.

import Foundation
import MLX
import MLXNN

/// HGNetV2ConvLayerLight: 1x1 (no act) → depthwise k×k (relu).
final class LightConv: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: ConvNorm
    @ModuleInfo(key: "conv2") var conv2: ConvNorm

    init(_ inCh: Int, _ outCh: Int, kernelSize: Int) {
        self._conv1 = ModuleInfo(wrappedValue: ConvNorm(inCh, outCh, kernelSize: 1, activation: nil), key: "conv1")
        self._conv2 = ModuleInfo(
            wrappedValue: ConvNorm(outCh, outCh, kernelSize: kernelSize, groups: outCh, activation: "relu"),
            key: "conv2")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv2(conv1(x)) }
}

/// HGNetV2BasicLayer: `layer_num` sub-convs whose outputs are concatenated with
/// the input, then squeezed/excited (aggregation), with optional residual.
final class HGBlock: Module {
    @ModuleInfo(key: "layers") var layers: [UnaryLayer]
    @ModuleInfo(key: "aggregation") var aggregation: [ConvNorm]
    let residual: Bool

    init(inCh: Int, midCh: Int, outCh: Int, layerNum: Int, kernelSize: Int,
         residual: Bool, lightBlock: Bool) {
        self.residual = residual
        var built: [UnaryLayer] = []
        for i in 0..<layerNum {
            let tin = (i == 0) ? inCh : midCh
            if lightBlock {
                built.append(LightConv(tin, midCh, kernelSize: kernelSize))
            } else {
                built.append(ConvNorm(tin, midCh, kernelSize: kernelSize, stride: 1, activation: "relu"))
            }
        }
        self._layers = ModuleInfo(wrappedValue: built, key: "layers")
        let total = inCh + layerNum * midCh
        let agg = [
            ConvNorm(total, outCh / 2, kernelSize: 1, activation: "relu"),
            ConvNorm(outCh / 2, outCh, kernelSize: 1, activation: "relu"),
        ]
        self._aggregation = ModuleInfo(wrappedValue: agg, key: "aggregation")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let identity = x
        var h = x
        var outputs = [x]
        for layer in layers {
            h = layer(h)
            outputs.append(h)
        }
        h = concatenated(outputs, axis: -1) // NHWC: channels last
        h = aggregation[1](aggregation[0](h))
        return residual ? h + identity : h
    }
}

/// HGNetV2Stage: optional depthwise downsample, then `num_blocks` HGBlocks.
final class HGStage: Module {
    @ModuleInfo(key: "downsample") var downsample: ConvNorm?
    @ModuleInfo(key: "blocks") var blocks: [HGBlock]
    let hasDownsample: Bool

    init(_ cfg: HGNetV2Config, _ idx: Int) {
        let inCh = cfg.stageInChannels[idx]
        let midCh = cfg.stageMidChannels[idx]
        let outCh = cfg.stageOutChannels[idx]
        let numBlocks = cfg.stageNumBlocks[idx]
        let numLayers = cfg.stageNumLayers[idx]
        let light = cfg.stageLightBlock[idx]
        let k = cfg.stageKernelSize[idx]
        self.hasDownsample = cfg.stageDownsample[idx]
        if hasDownsample {
            self._downsample = ModuleInfo(
                wrappedValue: ConvNorm(inCh, inCh, kernelSize: 3,
                                       stride: cfg.stageDownsampleStrides[idx],
                                       groups: inCh, activation: nil),
                key: "downsample")
        } else {
            self._downsample = ModuleInfo(wrappedValue: nil, key: "downsample")
        }
        var built: [HGBlock] = []
        for i in 0..<numBlocks {
            built.append(HGBlock(
                inCh: i == 0 ? inCh : outCh, midCh: midCh, outCh: outCh,
                layerNum: numLayers, kernelSize: k, residual: i != 0, lightBlock: light))
        }
        self._blocks = ModuleInfo(wrappedValue: built, key: "blocks")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = hasDownsample ? downsample!(x) : x
        for b in blocks { h = b(h) }
        return h
    }
}

/// HGNetV2Embeddings stem.
final class HGStem: Module {
    @ModuleInfo(key: "stem1") var stem1: ConvNorm
    @ModuleInfo(key: "stem2a") var stem2a: ConvNorm
    @ModuleInfo(key: "stem2b") var stem2b: ConvNorm
    @ModuleInfo(key: "stem3") var stem3: ConvNorm
    @ModuleInfo(key: "stem4") var stem4: ConvNorm
    let pool: MaxPool2d

    init(_ cfg: HGNetV2Config) {
        let c = cfg.stemChannels // [3, 32, 48]
        let s = cfg.stemStrides  // [2,1,1,2,1]
        self._stem1 = ModuleInfo(wrappedValue: ConvNorm(c[0], c[1], kernelSize: 3, stride: s[0]), key: "stem1")
        self._stem2a = ModuleInfo(wrappedValue: ConvNorm(c[1], c[1] / 2, kernelSize: 2, stride: s[1], padding: 0), key: "stem2a")
        self._stem2b = ModuleInfo(wrappedValue: ConvNorm(c[1] / 2, c[1], kernelSize: 2, stride: s[2], padding: 0), key: "stem2b")
        self._stem3 = ModuleInfo(wrappedValue: ConvNorm(c[1] * 2, c[1], kernelSize: 3, stride: s[3]), key: "stem3")
        self._stem4 = ModuleInfo(wrappedValue: ConvNorm(c[1], c[2], kernelSize: 1, stride: s[4]), key: "stem4")
        self.pool = MaxPool2d(kernelSize: 2, stride: 1)
        super.init()
    }

    // F.pad(x, (0,1,0,1)) in NHWC pads H (dim1) and W (dim2) by 1 on the bottom/right.
    private func padBR(_ x: MLXArray) -> MLXArray {
        padded(x, widths: [.init(0), .init((0, 1)), .init((0, 1)), .init(0)])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var emb = stem1(x)
        emb = padBR(emb)
        var emb2a = stem2a(emb)
        emb2a = padBR(emb2a)
        emb2a = stem2b(emb2a)
        let pooled = pool(emb)
        emb = concatenated([pooled, emb2a], axis: -1)
        emb = stem3(emb)
        emb = stem4(emb)
        return emb
    }
}

/// HGNetV2Encoder: holds the 4 stages (key path `.encoder.stages`).
final class HGNetV2Encoder: Module {
    @ModuleInfo(key: "stages") var stages: [HGStage]

    init(_ cfg: HGNetV2Config) {
        var built: [HGStage] = []
        for i in 0..<cfg.stageInChannels.count { built.append(HGStage(cfg, i)) }
        self._stages = ModuleInfo(wrappedValue: built, key: "stages")
        super.init()
    }
}

/// HGNetV2 backbone body (key path `backbone.model.*`): embedder + encoder.
final class HGNetV2Model: Module {
    @ModuleInfo(key: "embedder") var embedder: HGStem
    @ModuleInfo(key: "encoder") var encoder: HGNetV2Encoder

    init(_ cfg: HGNetV2Config) {
        self._embedder = ModuleInfo(wrappedValue: HGStem(cfg), key: "embedder")
        self._encoder = ModuleInfo(wrappedValue: HGNetV2Encoder(cfg), key: "encoder")
        super.init()
    }

    /// Returns the 4 stage feature maps (NHWC), strides 4/8/16/32.
    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        var h = embedder(x)
        var feats: [MLXArray] = []
        for stage in encoder.stages {
            h = stage(h)
            feats.append(h)
        }
        return feats
    }
}

/// PPDocLayoutV3ConvEncoder wrapper (key path `backbone.*`): holds `model`.
public final class HGNetV2Backbone: Module {
    @ModuleInfo(key: "model") var model: HGNetV2Model

    public init(_ cfg: HGNetV2Config) {
        self._model = ModuleInfo(wrappedValue: HGNetV2Model(cfg), key: "model")
        super.init()
    }

    /// Returns the 4 stage feature maps (NHWC), strides 4/8/16/32.
    public func callAsFunction(_ x: MLXArray) -> [MLXArray] { model(x) }
}
