// Gradient-correctness smoke test for head-only training: freeze the trunk,
// feed one image + synthetic ground-truth boxes, train decoder.class_embed /
// decoder.bbox_embed, and assert the loss collapses and the matched predicted
// boxes converge toward the targets.
//
// Requires the local checkpoint + fixtures (same as ParityTests); skipped if absent.

import XCTest
import MLX
@testable import MLXPPDocLayoutV3

final class OverfitTests: XCTestCase {
    static let repoRoot = URL(filePath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static var fixturesDir: URL { repoRoot.appending(path: "fixtures") }
    static var snapshot: URL {
        FileManager.default.homeDirectoryForCurrentUser.appending(
            path: ".cache/huggingface/hub/models--PaddlePaddle--PP-DocLayoutV3_safetensors/snapshots/da745593e69e44b5228fb909ec7ac5f1aadeea1f")
    }

    func testOverfitSingleSample() throws {
        guard FileManager.default.fileExists(atPath: Self.snapshot.appending(path: "model.safetensors").path),
              FileManager.default.fileExists(atPath: Self.fixturesDir.appending(path: "stage_tensors.safetensors").path)
        else { throw XCTSkip("checkpoint or fixtures not available") }

        let cfg = try PPDocLayoutV3Config.load(contentsOf: Self.snapshot.appending(path: "config.json"))
        let model = PPDocLayoutV3Model(cfg)
        try loadWeights(url: Self.snapshot.appending(path: "model.safetensors"), into: model, dtype: .float32)
        let fx = try loadArrays(url: Self.fixturesDir.appending(path: "stage_tensors.safetensors"))
        let pixels = fx["pixel_values"]!.transposed(0, 2, 3, 1)

        // Synthetic ground truth: two boxes with distinct labels.
        let targets = [Target(
            labels: [0, 7],
            boxes: MLXArray([
                Float(0.5), 0.5, 0.2, 0.2,
                0.25, 0.3, 0.1, 0.15,
            ], [2, 4]))]

        let lossCfg = DetectionLossConfig(numClasses: cfg.numLabels)
        let trainer = HeadTrainer(model: model, learningRate: 3e-4, config: lossCfg)

        // Freeze the trunk once; reuse its decoder inputs across all steps.
        let inputs = model.decoderInputs(pixels)
        eval(inputs.target, inputs.value, inputs.referenceUnact, inputs.maskFeat)

        // Fix the assignment once. Rematching every step (the default) makes the
        // box target a moving goalpost — the dense query set means tiny logit
        // shifts flip the Hungarian match — so the joint VFL+box overfit only
        // converges deterministically with a stable assignment. The realistic
        // rematching path is exercised by the other tests.
        let fixed = DetectionLoss.matches(
            output: model.runDecoder(inputs), targets: targets, cost: lossCfg.cost)
        let lastPairs = fixed[fixed.count - 1][0]

        func breakdown() -> (Float, Float, Float, Float) {
            let lb = DetectionLoss.loss(
                output: model.runDecoder(inputs), targets: targets, matches: fixed, config: lossCfg)
            return (lb.total.item(Float.self), lb.classLoss.item(Float.self),
                    lb.bboxLoss.item(Float.self), lb.giouLoss.item(Float.self))
        }
        func boxL1() -> Float {
            let out = model.runDecoder(inputs)
            let last = out.interBoxes.count - 1
            let qIdx = MLXArray(lastPairs.map { Int32($0.0) })
            let tIdx = MLXArray(lastPairs.map { Int32($0.1) })
            return abs(out.interBoxes[last][0][qIdx] - targets[0].boxes[tIdx]).mean().item(Float.self)
        }

        let (initialLoss, initClass, initBbox, initGiou) = breakdown()
        let initialBoxL1 = boxL1()

        var lastLoss = initialLoss
        for i in 0..<300 {
            let lb = trainer.step(inputs: inputs, targets: targets, fixedMatches: fixed)
            lastLoss = lb.total.item(Float.self)
            if i % 50 == 0 {
                print("step \(i): total \(lastLoss) cls \(lb.classLoss.item(Float.self)) "
                    + "bbox \(lb.bboxLoss.item(Float.self)) giou \(lb.giouLoss.item(Float.self))")
            }
        }

        let (_, finClass, finBbox, finGiou) = breakdown()
        let finalBoxL1 = boxL1()

        print("total \(initialLoss)→\(lastLoss)  cls \(initClass)→\(finClass)  "
            + "bbox \(initBbox)→\(finBbox)  giou \(initGiou)→\(finGiou)  boxL1 \(initialBoxL1)→\(finalBoxL1)")

        // A single sample is overfit: every loss term drops sharply and the
        // matched boxes converge onto the targets — proving the gradient is
        // correct end-to-end through the decoder and the deformable cross-attention.
        XCTAssertLessThan(lastLoss, initialLoss * 0.25, "total loss should collapse")
        XCTAssertLessThan(finClass, initClass * 0.2, "VFL classification should drop sharply")
        XCTAssertLessThan(finBbox + finGiou, (initBbox + initGiou) * 0.3, "box loss should collapse")
        XCTAssertLessThan(finalBoxL1, initialBoxL1 * 0.2, "matched boxes should converge to GT")
    }

    /// Isolates the box head: with the bipartite match frozen, the matched
    /// query's box has a stable target, so the box loss (L1 + GIoU) must
    /// decrease — proving the gradient through bbox_embed and the (frozen)
    /// deformable cross-attention is correct, not just present.
    func testBoxHeadConvergesWithFixedMatch() throws {
        guard FileManager.default.fileExists(atPath: Self.snapshot.appending(path: "model.safetensors").path),
              FileManager.default.fileExists(atPath: Self.fixturesDir.appending(path: "stage_tensors.safetensors").path)
        else { throw XCTSkip("checkpoint or fixtures not available") }

        let cfg = try PPDocLayoutV3Config.load(contentsOf: Self.snapshot.appending(path: "config.json"))
        let model = PPDocLayoutV3Model(cfg)
        try loadWeights(url: Self.snapshot.appending(path: "model.safetensors"), into: model, dtype: .float32)
        let fx = try loadArrays(url: Self.fixturesDir.appending(path: "stage_tensors.safetensors"))
        let pixels = fx["pixel_values"]!.transposed(0, 2, 3, 1)

        let targets = [Target(
            labels: [0, 7],
            boxes: MLXArray([Float(0.5), 0.5, 0.2, 0.2, 0.25, 0.3, 0.1, 0.15], [2, 4]))]
        // Box-only loss (vfl weight 0) isolates the bbox_embed gradient path.
        let lossCfg = DetectionLossConfig(numClasses: cfg.numLabels, vflWeight: 0)
        let trainer = HeadTrainer(model: model, learningRate: 1e-4, config: lossCfg)

        let inputs = model.decoderInputs(pixels)
        eval(inputs.target, inputs.value, inputs.referenceUnact, inputs.maskFeat)

        // Freeze the assignment to the initial best match and reuse it every step.
        let fixed = DetectionLoss.matches(
            output: model.runDecoder(inputs), targets: targets, cost: lossCfg.cost)

        func boxLoss() -> Float {
            let lb = DetectionLoss.loss(
                output: model.runDecoder(inputs), targets: targets, matches: fixed, config: lossCfg)
            return (lb.bboxLoss + lb.giouLoss).item(Float.self)
        }

        let initBox = boxLoss()
        for i in 0..<200 {
            let lb = trainer.step(inputs: inputs, targets: targets, fixedMatches: fixed)
            if i % 40 == 0 {
                print("step \(i): box \((lb.bboxLoss + lb.giouLoss).item(Float.self))")
            }
        }
        let finBox = boxLoss()
        print("fixed-match box loss \(initBox) → \(finBox)")

        XCTAssertLessThan(finBox, initBox * 0.3, "box loss should drop with a stable match")
    }

    /// Loads the checkpoint + fixtures, or skips. Returns config, model, pixels.
    func loadOrSkip() throws -> (PPDocLayoutV3Config, PPDocLayoutV3Model, MLXArray) {
        guard FileManager.default.fileExists(atPath: Self.snapshot.appending(path: "model.safetensors").path),
              FileManager.default.fileExists(atPath: Self.fixturesDir.appending(path: "stage_tensors.safetensors").path)
        else { throw XCTSkip("checkpoint or fixtures not available") }
        let cfg = try PPDocLayoutV3Config.load(contentsOf: Self.snapshot.appending(path: "config.json"))
        let model = PPDocLayoutV3Model(cfg)
        try loadWeights(url: Self.snapshot.appending(path: "model.safetensors"), into: model, dtype: .float32)
        let fx = try loadArrays(url: Self.fixturesDir.appending(path: "stage_tensors.safetensors"))
        return (cfg, model, fx["pixel_values"]!.transposed(0, 2, 3, 1))
    }

    /// The mask head learns synthetic per-instance masks. The class/box heads are
    /// frozen, so the bipartite match is stable and the mask loss must drop —
    /// proving the gradient reaches `mask_query_head`.
    func testMaskHeadConverges() throws {
        let (cfg, model, pixels) = try loadOrSkip()
        let lossCfg = DetectionLossConfig(numClasses: cfg.numLabels, vflWeight: 0, orderWeight: 0)
        let trainer = HeadTrainer(model: model, learningRate: 5e-4, config: lossCfg, heads: .mask)

        let inputs = model.decoderInputs(pixels)
        eval(inputs.target, inputs.value, inputs.referenceUnact, inputs.maskFeat)

        // Synthetic GT masks at mask-feature resolution: target 0 = top half,
        // target 1 = left half.
        let probe = model.runDecoder(inputs)
        let mH = probe.outMasks.dim(2), mW = probe.outMasks.dim(3)
        var g = [Float](repeating: 0, count: 2 * mH * mW)
        for y in 0..<mH {
            for x in 0..<mW {
                if y < mH / 2 { g[y * mW + x] = 1 }
                if x < mW / 2 { g[mH * mW + y * mW + x] = 1 }
            }
        }
        let targets = [Target(
            labels: [0, 7],
            boxes: MLXArray([Float(0.5), 0.5, 0.2, 0.2, 0.25, 0.3, 0.1, 0.15], [2, 4]),
            masks: MLXArray(g, [2, mH, mW]))]

        func maskLossNow() -> Float {
            DetectionLoss.loss(output: model.runDecoder(inputs), targets: targets, config: lossCfg)
                .maskLoss.item(Float.self)
        }
        let initMask = maskLossNow()
        for i in 0..<150 {
            let lb = trainer.step(inputs: inputs, targets: targets)
            if i % 30 == 0 { print("step \(i): mask \(lb.maskLoss.item(Float.self))") }
        }
        let finMask = maskLossNow()
        print("mask loss \(initMask) → \(finMask)")
        XCTAssertLessThan(finMask, initMask * 0.5, "mask loss should drop")
    }

    /// The reading-order head learns a synthetic precedence. With class/box frozen
    /// the match is stable, so the global-pointer loss must collapse — proving the
    /// gradient reaches `decoder_order_head` + `decoder_global_pointer`.
    func testOrderHeadConverges() throws {
        let (cfg, model, pixels) = try loadOrSkip()
        let lossCfg = DetectionLossConfig(numClasses: cfg.numLabels, vflWeight: 0, maskWeight: 0)
        let trainer = HeadTrainer(model: model, learningRate: 1e-3, config: lossCfg, heads: .order)

        let inputs = model.decoderInputs(pixels)
        eval(inputs.target, inputs.value, inputs.referenceUnact, inputs.maskFeat)

        // Two targets with an explicit reading order (target 0 read first).
        let targets = [Target(
            labels: [0, 7],
            boxes: MLXArray([Float(0.5), 0.5, 0.2, 0.2, 0.25, 0.3, 0.1, 0.15], [2, 4]),
            orderRanks: [0, 1])]

        func orderLossNow() -> Float {
            DetectionLoss.loss(output: model.runDecoder(inputs), targets: targets, config: lossCfg)
                .orderLoss.item(Float.self)
        }
        let initOrder = orderLossNow()
        for i in 0..<150 {
            let lb = trainer.step(inputs: inputs, targets: targets)
            if i % 30 == 0 { print("step \(i): order \(lb.orderLoss.item(Float.self))") }
        }
        let finOrder = orderLossNow()
        print("order loss \(initOrder) → \(finOrder)")
        XCTAssertLessThan(finOrder, initOrder * 0.2, "order loss should collapse")
    }
}
