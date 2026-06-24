// Weight loading + HF→Swift key remapping for PP-DocLayoutV3.
//
// HF safetensors keys are prefixed `model.` and use `convolution`/`normalization`
// for conv blocks and `nn.Sequential` integer indices for the conv+bn projection
// stacks. We remap those to the Swift module tree and transpose Conv2d weights
// from PyTorch NCHW (O,I,kH,kW) to MLX NHWC (O,kH,kW,I).

import Foundation
import MLX
import MLXNN

/// Remap a single HF key (already stripped of the leading `model.`) to the Swift
/// module-tree key. Returns nil for keys that should be dropped.
func remapKey(_ key: String) -> String? {
    if key.hasSuffix("num_batches_tracked") { return nil }
    // Training-only embedding; unused at inference and not present in Swift tree.
    if key.hasPrefix("denoising_class_embed") { return nil }

    var parts = key.components(separatedBy: ".")

    // nn.Sequential[Conv2d, BatchNorm2d] projection stacks: prefix.<list>.<0|1>.rest
    // Rename only the Sequential index (the one after the list index).
    if (parts.first == "encoder_input_proj" || parts.first == "decoder_input_proj"),
       parts.count >= 3, Int(parts[1]) != nil {
        if parts[2] == "0" { parts[2] = "conv" } else if parts[2] == "1" { parts[2] = "norm" }
    }
    // enc_output = nn.Sequential[Linear, LayerNorm]
    if parts.first == "enc_output", parts.count >= 2 {
        if parts[1] == "0" { parts[1] = "linear" } else if parts[1] == "1" { parts[1] = "norm" }
    }

    var k = parts.joined(separator: ".")
    // Canonical conv/bn sub-keys used by ConvNorm everywhere.
    k = k.replacingOccurrences(of: ".convolution.", with: ".conv.")
    k = k.replacingOccurrences(of: ".normalization.", with: ".norm.")
    return k
}

/// True if a 4D weight at this Swift key is a Conv2d/ConvNorm kernel needing the
/// NCHW→NHWC transpose. (All 4D weights in this model are standard convs.)
private func isConvWeight(_ key: String) -> Bool {
    key.hasSuffix(".weight")
}

/// Sanitize a single HF key/value into the Swift representation.
public func sanitize(key: String, value: MLXArray, dtype: DType) -> (String, MLXArray)? {
    guard key.hasPrefix("model.") else { return nil }
    let stripped = String(key.dropFirst("model.".count))
    guard let mapped = remapKey(stripped) else { return nil }

    var v = value.asType(dtype)
    if v.ndim == 4 && isConvWeight(mapped) {
        v = v.transposed(0, 2, 3, 1) // (O,I,kH,kW) → (O,kH,kW,I)
    }
    return (mapped, v)
}

/// Load PP-DocLayoutV3 weights from a safetensors file into `model`.
public func loadWeights(
    url: URL, into model: PPDocLayoutV3Model, dtype: DType = .float32,
    verify: Module.VerifyUpdate = [.noUnusedKeys]
) throws {
    let weights = try loadArrays(url: url)
    var remapped: [(String, MLXArray)] = []
    remapped.reserveCapacity(weights.count)
    for (key, value) in weights {
        if let pair = sanitize(key: key, value: value, dtype: dtype) {
            remapped.append(pair)
            // Tied weights: decoder.class_embed ← enc_score_head,
            //               decoder.bbox_embed  ← enc_bbox_head.
            if pair.0.hasPrefix("enc_score_head.") {
                remapped.append(("decoder.class_embed." + pair.0.dropFirst("enc_score_head.".count), pair.1))
            } else if pair.0.hasPrefix("enc_bbox_head.") {
                remapped.append(("decoder.bbox_embed." + pair.0.dropFirst("enc_bbox_head.".count), pair.1))
            }
        }
    }
    try model.update(parameters: ModuleParameters.unflattened(remapped), verify: verify)
    eval(model)
}
