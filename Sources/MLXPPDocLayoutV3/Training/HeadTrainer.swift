// Head-only fine-tuning driver for PP-DocLayoutV3.
//
// The backbone, hybrid encoder, and query-selection trunk are frozen; only the
// decoder's `class_embed` and `bbox_embed` heads are trained. The frozen trunk
// is evaluated once per image (``PPDocLayoutV3Model/decoderInputs(_:)``) and
// reused across steps, so each training step runs only the decoder + loss.
//
// Gradients reach `bbox_embed` through the (frozen) deformable cross-attention,
// whose bilinear grid-sample ships a VJP in `Decoder/GridSampleKernel.swift`.

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Trains the decoder class/box heads with Hungarian-matched detection loss.
public final class HeadTrainer {
    /// Which decoder heads to fine-tune. Everything else stays frozen.
    public struct TrainableHeads: OptionSet, Sendable {
        public let rawValue: Int
        public init(rawValue: Int) { self.rawValue = rawValue }
        /// `decoder.class_embed` + `decoder.bbox_embed`.
        public static let classBox = TrainableHeads(rawValue: 1 << 0)
        /// `mask_query_head` (predicts the per-query mask prototypes).
        public static let mask = TrainableHeads(rawValue: 1 << 1)
        /// `decoder_order_head` + `decoder_global_pointer` (reading order).
        public static let order = TrainableHeads(rawValue: 1 << 2)
        public static let all: TrainableHeads = [.classBox, .mask, .order]
    }

    public let model: PPDocLayoutV3Model
    public let config: DetectionLossConfig
    let optimizer: AdamW

    /// Freezes the whole model, then unfreezes the selected heads.
    /// - Parameters:
    ///   - model: a loaded model (weights already loaded).
    ///   - learningRate: AdamW learning rate.
    ///   - config: loss configuration (class count + weights).
    ///   - heads: which heads to train (default `.classBox`).
    public init(
        model: PPDocLayoutV3Model, learningRate: Float = 1e-4, config: DetectionLossConfig,
        heads: TrainableHeads = .classBox
    ) {
        self.model = model
        self.config = config
        self.optimizer = AdamW(learningRate: learningRate)

        model.freeze()
        if heads.contains(.classBox) {
            model.decoder.classEmbed.unfreeze()
            model.decoder.bboxEmbed.unfreeze()
        }
        if heads.contains(.mask) {
            model.maskQueryHead.unfreeze()
        }
        if heads.contains(.order) {
            for h in model.decoderOrderHead { h.unfreeze() }
            model.decoderGlobalPointer.unfreeze()
        }
    }

    /// One optimizer step on a single image's precomputed trunk inputs.
    /// - Parameters:
    ///   - inputs: frozen-trunk decoder inputs (compute once with `model.decoderInputs`).
    ///   - targets: ground truth, one ``Target`` per batch element of `inputs`.
    ///   - fixedMatches: if provided, reuse this assignment instead of rematching
    ///     (stabilizes the box-loss target; otherwise the match is recomputed from
    ///     the current predictions, as in normal training).
    /// - Returns: the loss components for this step.
    @discardableResult
    public func step(
        inputs: PPDocLayoutV3Model.DecoderInputs, targets: [Target],
        fixedMatches: [[[(Int, Int)]]]? = nil
    ) -> LossBreakdown {
        // Matching is non-differentiable: run the decoder once, match outside grad.
        let matches: [[[(Int, Int)]]]
        if let fixedMatches {
            matches = fixedMatches
        } else {
            let out0 = model.runDecoder(inputs)
            matches = DetectionLoss.matches(output: out0, targets: targets, cost: config.cost)
        }

        let cfg = config
        let lg = valueAndGrad(model: model) {
            (m: PPDocLayoutV3Model, args: (PPDocLayoutV3Model.DecoderInputs, [Target], [[[(Int, Int)]]]))
                -> [MLXArray] in
            let out = m.runDecoder(args.0)
            let lb = DetectionLoss.loss(output: out, targets: args.1, matches: args.2, config: cfg)
            return [lb.total, lb.classLoss, lb.bboxLoss, lb.giouLoss, lb.maskLoss, lb.orderLoss]
        }

        let (values, grads) = lg(model, (inputs, targets, matches))
        optimizer.update(model: model, gradients: grads)
        eval(model, optimizer)

        return LossBreakdown(
            classLoss: values[1], bboxLoss: values[2], giouLoss: values[3],
            maskLoss: values[4], orderLoss: values[5], total: values[0])
    }
}
