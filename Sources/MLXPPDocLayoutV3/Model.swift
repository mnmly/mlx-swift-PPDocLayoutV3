// PP-DocLayoutV3 model trunk + ForObjectDetection head.
//
// PORT FROM: reference/python/modeling_pp_doclayout_v3.py
//   (PPDocLayoutV3Model.forward, generate_anchors, mask_to_box_coordinate,
//    PPDocLayoutV3ForObjectDetection)

import Foundation
import MLX
import MLXNN

private let BIG: Float = 3.4e38

/// enc_output = Linear → LayerNorm.
final class EncOutput: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm
    init(_ dim: Int, eps: Float) {
        self._linear = ModuleInfo(wrappedValue: Linear(dim, dim), key: "linear")
        self._norm = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim, eps: eps), key: "norm")
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(linear(x)) }
}

/// Final model output (last decoder layer).
public struct PPDocLayoutV3Output {
    public var logits: MLXArray      // (B, Q, numLabels)
    public var predBoxes: MLXArray   // (B, Q, 4) cxcywh in [0,1]
    public var orderLogits: MLXArray // (B, Q, Q)
    public var outMasks: MLXArray    // (B, Q, mH, mW)
}

public final class PPDocLayoutV3Model: Module {
    @ModuleInfo(key: "backbone") public var backbone: HGNetV2Backbone
    @ModuleInfo(key: "encoder_input_proj") var encoderInputProj: [ConvNorm]
    @ModuleInfo(key: "encoder") public var encoder: HybridEncoder
    @ModuleInfo(key: "enc_output") var encOutput: EncOutput
    @ModuleInfo(key: "enc_score_head") var encScoreHead: Linear
    @ModuleInfo(key: "enc_bbox_head") var encBboxHead: MLPPredictionHead
    @ModuleInfo(key: "decoder_input_proj") var decoderInputProj: [ConvNorm]
    @ModuleInfo(key: "decoder") var decoder: Decoder
    @ModuleInfo(key: "decoder_order_head") var decoderOrderHead: [Linear]
    @ModuleInfo(key: "decoder_global_pointer") var decoderGlobalPointer: GlobalPointer
    @ModuleInfo(key: "decoder_norm") var decoderNorm: LayerNorm
    @ModuleInfo(key: "mask_query_head") var maskQueryHead: MLPPredictionHead

    let cfg: PPDocLayoutV3Config

    public init(_ cfg: PPDocLayoutV3Config) {
        self.cfg = cfg
        self._backbone = ModuleInfo(wrappedValue: HGNetV2Backbone(cfg.backbone), key: "backbone")

        // encoder_input_proj: drops the x4 level → one per encoder_in_channels.
        var einp: [ConvNorm] = []
        for ch in cfg.encoderInChannels {
            einp.append(ConvNorm(ch, cfg.encoderHiddenDim, kernelSize: 1, activation: nil))
        }
        self._encoderInputProj = ModuleInfo(wrappedValue: einp, key: "encoder_input_proj")

        self._encoder = ModuleInfo(wrappedValue: HybridEncoder(cfg), key: "encoder")
        self._encOutput = ModuleInfo(wrappedValue: EncOutput(cfg.dModel, eps: cfg.layerNormEps), key: "enc_output")
        self._encScoreHead = ModuleInfo(wrappedValue: Linear(cfg.dModel, cfg.numLabels), key: "enc_score_head")
        self._encBboxHead = ModuleInfo(wrappedValue: MLPPredictionHead(cfg.dModel, cfg.dModel, 4, numLayers: 3), key: "enc_bbox_head")

        var dinp: [ConvNorm] = []
        for ch in cfg.decoderInChannels {
            dinp.append(ConvNorm(ch, cfg.dModel, kernelSize: 1, activation: nil, bnEps: cfg.batchNormEps))
        }
        self._decoderInputProj = ModuleInfo(wrappedValue: dinp, key: "decoder_input_proj")

        self._decoder = ModuleInfo(wrappedValue: Decoder(cfg), key: "decoder")
        var orderHeads: [Linear] = []
        for _ in 0..<cfg.decoderLayers { orderHeads.append(Linear(cfg.dModel, cfg.dModel)) }
        self._decoderOrderHead = ModuleInfo(wrappedValue: orderHeads, key: "decoder_order_head")
        self._decoderGlobalPointer = ModuleInfo(wrappedValue: GlobalPointer(dModel: cfg.dModel, headSize: cfg.globalPointerHeadSize), key: "decoder_global_pointer")
        self._decoderNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: cfg.dModel, eps: cfg.layerNormEps), key: "decoder_norm")
        self._maskQueryHead = ModuleInfo(wrappedValue: MLPPredictionHead(cfg.dModel, cfg.dModel, cfg.numPrototypes, numLayers: 3), key: "mask_query_head")
        super.init()
    }

    /// Anchors + valid mask for given per-level spatial shapes. Returns
    /// (anchors (1, S, 4), validMask (1, S, 1) as float 0/1).
    func generateAnchors(_ spatialShapes: [(Int, Int)], gridSize: Float = 0.05) -> (MLXArray, MLXArray) {
        var anchorsList: [MLXArray] = []
        for (level, (H, W)) in spatialShapes.enumerated() {
            let gy = broadcast(MLXArray(0..<H).asType(.float32).reshaped([H, 1]), to: [H, W])
            let gx = broadcast(MLXArray(0..<W).asType(.float32).reshaped([1, W]), to: [H, W])
            var gridXY = stacked([gx, gy], axis: -1) // (H,W,2)
            gridXY = gridXY + 0.5
            let normWH = MLXArray([Float(W), Float(H)]).reshaped([1, 1, 2])
            gridXY = gridXY / normWH
            let wh = MLXArray.ones([H, W, 2]) * (gridSize * pow(2.0, Float(level)))
            let anchor = concatenated([gridXY, wh], axis: -1).reshaped([1, H * W, 4])
            anchorsList.append(anchor)
        }
        var anchors = concatenated(anchorsList, axis: 1) // (1, S, 4)
        let eps: Float = 1e-2
        let valid = (anchors .> eps) .&& (anchors .< (1 - eps))
        let validMask = valid.all(axis: -1, keepDims: true) // (1, S, 1) bool
        anchors = log(anchors / (1 - anchors))
        anchors = MLX.where(validMask, anchors, MLXArray(BIG))
        return (anchors, validMask.asType(.float32))
    }

    /// Derive cxcywh reference boxes from binary masks. mask: (B, Q, H, W) bool.
    func maskToBoxCoordinate(_ maskBool: MLXArray) -> MLXArray {
        let H = maskBool.dim(-2), W = maskBool.dim(-1)
        let m = maskBool.asType(.float32)
        let yc = broadcast(MLXArray(0..<H).asType(.float32).reshaped([H, 1]), to: [H, W])
        let xc = broadcast(MLXArray(0..<W).asType(.float32).reshaped([1, W]), to: [H, W])
        let B = maskBool.dim(0), Q = maskBool.dim(1)
        let xcb = xc.reshaped([1, 1, H, W]); let ycb = yc.reshaped([1, 1, H, W])

        let xMaskedFlat = (xcb * m).reshaped([B, Q, H * W])
        let yMaskedFlat = (ycb * m).reshaped([B, Q, H * W])
        let mFlat = maskBool.reshaped([B, Q, H * W])

        let xMax = xMaskedFlat.max(axis: -1) + 1
        let yMax = yMaskedFlat.max(axis: -1) + 1
        let xMin = MLX.where(mFlat, xMaskedFlat, MLXArray(BIG)).min(axis: -1)
        let yMin = MLX.where(mFlat, yMaskedFlat, MLXArray(BIG)).min(axis: -1)

        var bbox = stacked([xMin, yMin, xMax, yMax], axis: -1) // (B,Q,4)
        let nonEmpty = mFlat.any(axis: -1, keepDims: true).asType(.float32) // (B,Q,1)
        bbox = bbox * nonEmpty
        let norm = MLXArray([Float(W), Float(H), Float(W), Float(H)]).reshaped([1, 1, 4])
        let xyxy = bbox / norm
        let xMinN = xyxy[0..., 0..., 0], yMinN = xyxy[0..., 0..., 1]
        let xMaxN = xyxy[0..., 0..., 2], yMaxN = xyxy[0..., 0..., 3]
        let cx = (xMinN + xMaxN) / 2, cy = (yMinN + yMaxN) / 2
        let bw = xMaxN - xMinN, bh = yMaxN - yMinN
        return stacked([cx, cy, bw, bh], axis: -1)
    }

    /// Intermediates for staged parity checks (encoder + enc proposal heads).
    public struct Trunk {
        public var panMaps: [MLXArray]
        public var maskFeat: MLXArray
        public var encOutputsClass: MLXArray
        public var encOutputsCoord: MLXArray
    }

    /// Run backbone → encoder → enc proposal heads, returning intermediates.
    public func trunk(_ pixelValues: MLXArray) -> Trunk {
        let feats = backbone(pixelValues)
        let x4Feat = feats[0]
        var projFeats: [MLXArray] = []
        for (i, proj) in encoderInputProj.enumerated() { projFeats.append(proj(feats[i + 1])) }
        let enc = encoder(projFeats, x4Feat: x4Feat)
        var sources: [MLXArray] = []
        for (level, src) in enc.lastHiddenState.enumerated() { sources.append(decoderInputProj[level](src)) }
        let B = pixelValues.dim(0)
        var spatialShapes: [(Int, Int)] = []
        var flat: [MLXArray] = []
        for src in sources {
            let H = src.dim(1), W = src.dim(2), C = src.dim(3)
            spatialShapes.append((H, W))
            flat.append(src.reshaped([B, H * W, C]))
        }
        let sourceFlatten = concatenated(flat, axis: 1)
        let (anchors, validMask) = generateAnchors(spatialShapes)
        let outputMemory = encOutput(validMask * sourceFlatten)
        return Trunk(panMaps: enc.lastHiddenState, maskFeat: enc.maskFeat,
                     encOutputsClass: encScoreHead(outputMemory),
                     encOutputsCoord: encBboxHead(outputMemory) + anchors)
    }

    /// Inputs the decoder consumes, produced by the backbone + hybrid encoder +
    /// query-selection trunk. Used by both inference (``callAsFunction(_:)``) and
    /// head-only training, where the trunk is frozen and these are computed once.
    public struct DecoderInputs {
        public var target: MLXArray          // (B, Q, d_model) selected query features
        public var value: MLXArray           // (B, S, d_model) flattened encoder memory
        public var referenceUnact: MLXArray  // (B, Q, 4) pre-sigmoid reference boxes
        public var spatialShapes: [(Int, Int)]
        public var maskFeat: MLXArray         // (B, mH, mW, numPrototypes)
    }

    /// Run backbone → encoder → query selection, returning the decoder inputs.
    /// pixelValues: (B, H, W, 3) NHWC, already preprocessed.
    public func decoderInputs(_ pixelValues: MLXArray) -> DecoderInputs {
        let feats = backbone(pixelValues) // [stage0(x4), stage1, stage2, stage3]
        let x4Feat = feats[0]
        var projFeats: [MLXArray] = []
        for (i, proj) in encoderInputProj.enumerated() { projFeats.append(proj(feats[i + 1])) }

        let enc = encoder(projFeats, x4Feat: x4Feat)

        // decoder input projections (3 levels)
        var sources: [MLXArray] = []
        for (level, src) in enc.lastHiddenState.enumerated() {
            sources.append(decoderInputProj[level](src))
        }

        // flatten sources: (B, S, d_model), record spatial shapes
        let B = pixelValues.dim(0)
        var spatialShapes: [(Int, Int)] = []
        var flat: [MLXArray] = []
        for src in sources {
            let H = src.dim(1), W = src.dim(2), C = src.dim(3)
            spatialShapes.append((H, W))
            flat.append(src.reshaped([B, H * W, C]))
        }
        let sourceFlatten = concatenated(flat, axis: 1) // (B, S, d_model)

        let (anchors, validMask) = generateAnchors(spatialShapes)
        let memory = validMask * sourceFlatten
        let outputMemory = encOutput(memory)

        let encOutputsClass = encScoreHead(outputMemory)               // (B, S, numLabels)
        let encOutputsCoord = encBboxHead(outputMemory) + anchors      // (B, S, 4)

        // top-k queries by max class score
        let maxScores = encOutputsClass.max(axis: -1)                  // (B, S)
        let topkInd = argSort(-maxScores, axis: -1)[0..., ..<cfg.numQueries] // (B, Q) descending

        let coordIdx = broadcast(topkInd.expandedDimensions(axis: -1), to: [B, cfg.numQueries, 4])
        let referenceUnactInit = takeAlong(encOutputsCoord, coordIdx, axis: 1) // (B, Q, 4)

        let memIdx = broadcast(topkInd.expandedDimensions(axis: -1), to: [B, cfg.numQueries, cfg.dModel])
        let target = takeAlong(outputMemory, memIdx, axis: 1)          // (B, Q, d_model)

        // mask-enhanced reference points
        let outQuery = decoderNorm(target)
        let maskQueryEmbed = maskQueryHead(outQuery)                   // (B, Q, P)
        let mH = enc.maskFeat.dim(1), mW = enc.maskFeat.dim(2), P = enc.maskFeat.dim(3)
        let maskFeatFlat = enc.maskFeat.reshaped([B, mH * mW, P]).transposed(0, 2, 1) // (B,P,S)
        let encOutMasks = matmul(maskQueryEmbed, maskFeatFlat).reshaped([B, cfg.numQueries, mH, mW])
        var referenceUnact = referenceUnactInit
        if cfg.maskEnhanced {
            let refBoxes = maskToBoxCoordinate(encOutMasks .> 0)
            referenceUnact = inverseSigmoid(refBoxes)
        }

        return DecoderInputs(
            target: target, value: sourceFlatten, referenceUnact: referenceUnact,
            spatialShapes: spatialShapes, maskFeat: enc.maskFeat)
    }

    /// Run the decoder given precomputed trunk inputs.
    func runDecoder(_ inputs: DecoderInputs) -> DecoderOutput {
        decoder(
            target: inputs.target, value: inputs.value,
            referencePointsUnact: inputs.referenceUnact,
            spatialShapes: inputs.spatialShapes,
            orderHead: decoderOrderHead, globalPointer: decoderGlobalPointer,
            maskQueryHead: maskQueryHead, norm: decoderNorm, maskFeat: inputs.maskFeat)
    }

    /// pixelValues: (B, H, W, 3) NHWC, already preprocessed.
    public func callAsFunction(_ pixelValues: MLXArray) -> PPDocLayoutV3Output {
        let out = runDecoder(decoderInputs(pixelValues))
        return PPDocLayoutV3Output(
            logits: out.logits, predBoxes: out.predBoxes,
            orderLogits: out.orderLogits, outMasks: out.outMasks)
    }
}
