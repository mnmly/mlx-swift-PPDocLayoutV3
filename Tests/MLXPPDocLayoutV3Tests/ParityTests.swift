// Parity tests against Python (transformers) fixtures.
//
// Fixtures live in the repo `fixtures/` dir (located via #filePath, not bundled).
// HF weights are read from the local huggingface cache.
//
// NOTE: per-tensor encoder parity is intentionally NOT asserted — fp32 GPU-vs-CPU
// conv accumulation drift propagates (same as the mlx-swift-PaddleOCR layout port).
// We gate on DETECTION-level parity: labels exact, scores ≤5e-3, boxes ≤2px.

import XCTest
import MLX
import MLXNN
@testable import MLXPPDocLayoutV3

final class ParityTests: XCTestCase {
    static let repoRoot = URL(filePath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static var fixturesDir: URL { repoRoot.appending(path: "fixtures") }
    static var snapshot: URL {
        FileManager.default.homeDirectoryForCurrentUser.appending(
            path: ".cache/huggingface/hub/models--PaddlePaddle--PP-DocLayoutV3_safetensors/snapshots/da745593e69e44b5228fb909ec7ac5f1aadeea1f")
    }

    struct Golden: Decodable { let score: Float; let label: Int; let box: [Float] }

    func loadModelAndFixtures() throws -> (PPDocLayoutV3Model, PPDocLayoutV3Config, [String: MLXArray], MLXArray) {
        let cfg = try PPDocLayoutV3Config.load(contentsOf: Self.snapshot.appending(path: "config.json"))
        let model = PPDocLayoutV3Model(cfg)
        try loadWeights(url: Self.snapshot.appending(path: "model.safetensors"), into: model, dtype: .float32)
        let fx = try loadArrays(url: Self.fixturesDir.appending(path: "stage_tensors.safetensors"))
        let pixels = fx["pixel_values"]!.transposed(0, 2, 3, 1)
        return (model, cfg, fx, pixels)
    }

    func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Backbone is fp32-exact (no BN-fusion drift) — assert tight.
    func testBackboneParity() throws {
        let (model, _, fx, pixels) = try loadModelAndFixtures()
        let feats = model.backbone(pixels)
        for i in 0..<4 {
            let ref = fx["bb_feat_\(i)"]!.transposed(0, 2, 3, 1)
            XCTAssertEqual(feats[i].shape, ref.shape, "backbone[\(i)] shape")
            XCTAssertLessThan(maxAbs(feats[i], ref), 1e-2, "backbone[\(i)] parity")
        }
    }

    /// End-to-end detection parity vs the transformers oracle.
    func testDetectionParity() throws {
        let (model, cfg, _, pixels) = try loadModelAndFixtures()
        let out = model(pixels)
        let dets = PostProcess.decode(output: out, targetSize: (2339, 1654), threshold: 0.5,
                                      numClasses: cfg.numLabels, polygons: false)

        let goldenData = try Data(contentsOf: Self.fixturesDir.appending(path: "golden_detections.json"))
        let golden = try JSONDecoder().decode([Golden].self, from: goldenData)

        XCTAssertEqual(dets.count, golden.count, "detection count")

        // Match each golden box to the nearest swift box (order-independent).
        for g in golden {
            let gb = g.box
            var best: Detection?
            var bestD = Float.greatestFiniteMagnitude
            for d in dets {
                let c = abs(d.box.0 - gb[0]) + abs(d.box.1 - gb[1]) + abs(d.box.2 - gb[2]) + abs(d.box.3 - gb[3])
                if c < bestD { bestD = c; best = d }
            }
            guard let m = best else { XCTFail("no match for golden box \(gb)"); continue }
            XCTAssertEqual(m.label, g.label, "label for box \(gb)")
            XCTAssertLessThan(abs(m.score - g.score), 5e-3, "score for box \(gb)")
            XCTAssertLessThan(abs(m.box.0 - gb[0]), 2.0, "x1 for box \(gb)")
            XCTAssertLessThan(abs(m.box.1 - gb[1]), 2.0, "y1 for box \(gb)")
            XCTAssertLessThan(abs(m.box.2 - gb[2]), 2.0, "x2 for box \(gb)")
            XCTAssertLessThan(abs(m.box.3 - gb[3]), 2.0, "y2 for box \(gb)")
        }
    }
}
