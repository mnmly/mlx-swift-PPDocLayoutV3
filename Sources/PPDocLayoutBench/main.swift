// Staged parity harness + benchmark for the PP-DocLayoutV3 port.
//
// Usage:
//   PPDocLayoutBench parity   — compare per-stage tensors vs Python fixtures
//   PPDocLayoutBench bench    — loop the pipeline, watch active memory

import Foundation
import MLX
import MLXNN
import MLXPPDocLayoutV3

// Resolve repo-relative paths from this source file location.
let thisFile = URL(filePath: #filePath)
let repoRoot = thisFile.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
let fixturesDir = repoRoot.appending(path: "fixtures")

guard let snapshot = ModelStore.resolveExistingModelDirectory() else {
    FileHandle.standardError.write(Data("error: model not found. Download \(ModelStore.repo) into ~/.cache/huggingface or run the app's Download button.\n".utf8))
    exit(1)
}
let configURL = snapshot.appending(path: "config.json")
let weightsURL = snapshot.appending(path: "model.safetensors")

func loadModel() throws -> PPDocLayoutV3Model {
    let cfg = try PPDocLayoutV3Config.load(contentsOf: configURL)
    let model = PPDocLayoutV3Model(cfg)
    try loadWeights(url: weightsURL, into: model, dtype: .float32)
    return model
}

/// Max abs / mean abs diff between two arrays (cast to fp32).
func diff(_ a: MLXArray, _ b: MLXArray) -> (Float, Float) {
    let d = abs(a.asType(.float32) - b.asType(.float32))
    return (d.max().item(Float.self), d.mean().item(Float.self))
}

func report(_ name: String, _ swift: MLXArray, _ ref: MLXArray) {
    eval(swift)
    if swift.shape != ref.shape {
        print("  ✘ \(name): SHAPE Swift \(swift.shape) vs ref \(ref.shape)")
        return
    }
    let (mx, mean) = diff(swift, ref)
    let flag = mx < 1e-2 ? "✔" : "✘"
    print("  \(flag) \(name): maxAbs=\(String(format: "%.2e", mx)) meanAbs=\(String(format: "%.2e", mean)) shape=\(swift.shape)")
}

let args = CommandLine.arguments
let cmd = args.count > 1 ? args[1] : "parity"

let fixtures = try loadArrays(url: fixturesDir.appending(path: "stage_tensors.safetensors"))
// pixel_values fixture is NCHW (1,3,800,800) → NHWC for MLX.
let pixelNCHW = fixtures["pixel_values"]!
let pixelValues = pixelNCHW.transposed(0, 2, 3, 1)

func toNHWC(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 3, 1) }

switch cmd {
case "keys":
    let cfg = try PPDocLayoutV3Config.load(contentsOf: configURL)
    let model = PPDocLayoutV3Model(cfg)
    let modelKeys = Set(model.parameters().flattened().map { $0.0 })
    let weights = try loadArrays(url: weightsURL)
    var mapped = Set<String>()
    for (k, v) in weights {
        if let (mk, _) = sanitize(key: k, value: v, dtype: .float32) { mapped.insert(mk) }
    }
    print("model params: \(modelKeys.count), mapped weights: \(mapped.count)")
    let missingInModel = mapped.subtracting(modelKeys).sorted()   // provided but no slot
    let missingInWeights = modelKeys.subtracting(mapped).sorted()  // slot but no weight
    print("--- mapped-but-not-in-model (\(missingInModel.count)) ---")
    for k in missingInModel.prefix(40) { print("  \(k)") }
    print("--- in-model-but-not-mapped (\(missingInWeights.count)) ---")
    for k in missingInWeights.prefix(40) { print("  \(k)") }

case "parity":
    print("Loading model…")
    let model = try loadModel()
    print("Stage parity vs Python fixtures:")

    // --- backbone ---
    let feats = model.backbone(pixelValues)
    for i in 0..<4 {
        report("backbone[\(i)]", feats[i], toNHWC(fixtures["bb_feat_\(i)"]!))
    }

    // --- encoder + proposal heads ---
    // NOTE: encoder maps drift ~0.1-0.2 maxAbs — inherent fp32 GPU-vs-CPU conv
    // accumulation divergence (see README). Detection-level parity is the oracle.
    let tr = model.trunk(pixelValues)
    for i in 0..<3 { report("enc_pan[\(i)]", tr.panMaps[i], toNHWC(fixtures["enc_pan_\(i)"]!)) }
    report("mask_feat", tr.maskFeat, toNHWC(fixtures["mask_feat"]!))
    report("enc_outputs_class", tr.encOutputsClass, fixtures["enc_outputs_class"]!)
    print("(encoder drift above is expected fp32 divergence — gate on `detect`)")

case "detect":
    let cfg = try PPDocLayoutV3Config.load(contentsOf: configURL)
    let model = try loadModel()
    let out = model(pixelValues)
    // target size from meta.json image_size_wh = [W,H]
    let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: fixturesDir.appending(path: "meta.json"))) as! [String: Any]
    let wh = meta["image_size_wh"] as! [Int]
    let dets = PostProcess.decode(output: out, targetSize: (wh[1], wh[0]), threshold: 0.5, numClasses: cfg.numLabels)
    let id2label = cfg.id2label
    print("Swift detections: \(dets.count)  (golden: \(meta["num_detections"] ?? "?"))")
    for (i, d) in dets.enumerated() {
        let name = id2label[String(d.label)] ?? "?"
        let b = d.box
        let pv = d.polygon.first.map { String(format: "[%.0f,%.0f]", $0.0, $0.1) } ?? "-"
        print(String(format: "  Order %2d: %-16@ %.2f  [%.1f, %.1f, %.1f, %.1f]  poly=%d verts first=%@", i + 1, name as NSString, d.score, b.0, b.1, b.2, b.3, d.polygon.count, pv as NSString))
    }

case "pipeline":
    let cfg = try PPDocLayoutV3Config.load(contentsOf: configURL)
    let pipeline = try PPDocLayoutV3Pipeline(modelDirectory: snapshot)
    let imgURL = fixturesDir.appending(path: "input_image.png")
    let dets = try pipeline.detect(imageURL: imgURL)
    print("Pipeline (real preprocessing) detections: \(dets.count)")
    for (i, d) in dets.enumerated() {
        print(String(format: "  Order %2d: %-16@ %.2f  [%.1f,%.1f,%.1f,%.1f]  poly=%d", i + 1, pipeline.label(d.label) as NSString, d.score, d.box.0, d.box.1, d.box.2, d.box.3, d.polygon.count))
    }
    _ = cfg

case "bench":
    let model = try loadModel()
    let iters = args.count > 2 ? Int(args[2]) ?? 20 : 20
    // warmup
    eval(model(pixelValues).logits)
    let start = Date()
    for i in 0..<iters {
        let o = model(pixelValues)
        eval(o.logits, o.predBoxes, o.orderLogits, o.outMasks)
        if i % 5 == 0 {
            let active = Double(MLX.Memory.activeMemory) / 1e6
            let cache = Double(MLX.Memory.cacheMemory) / 1e6
            print("  iter \(i)  active=\(String(format: "%.0f", active)) MB  cache=\(String(format: "%.0f", cache)) MB")
        }
    }
    let dt = Date().timeIntervalSince(start) / Double(iters)
    print("avg \(String(format: "%.3f", dt))s / image  (\(iters) iters), peak=\(MLX.Memory.peakMemory / 1_000_000) MB")

default:
    print("unknown command \(cmd)")
}
