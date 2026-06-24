// High-level PP-DocLayoutV3 pipeline: image file → layout detections.

import Foundation
import MLX

#if canImport(CoreGraphics)
import CoreGraphics
public final class PPDocLayoutV3Pipeline {
    public let config: PPDocLayoutV3Config
    public let model: PPDocLayoutV3Model
    let processor: ImageProcessor

    /// Load from a directory containing `config.json` and `model.safetensors`.
    public init(modelDirectory: URL, dtype: DType = .float32) throws {
        let cfg = try PPDocLayoutV3Config.load(contentsOf: modelDirectory.appending(path: "config.json"))
        self.config = cfg
        self.model = PPDocLayoutV3Model(cfg)
        try loadWeights(url: modelDirectory.appending(path: "model.safetensors"), into: model, dtype: dtype)
        self.processor = ImageProcessor(size: cfg.featStrides.isEmpty ? 800 : 800)
    }

    /// Detect layout regions in an image file.
    public func detect(imageURL: URL, threshold: Float = 0.5, polygons: Bool = true) throws -> [Detection] {
        let (pixels, w, h) = try processor.preprocess(url: imageURL)
        return decode(pixels, w, h, threshold: threshold, polygons: polygons)
    }

    /// Detect layout regions in an already-decoded image (e.g. a rendered PDF page).
    public func detect(cgImage: CGImage, threshold: Float = 0.5, polygons: Bool = true) throws -> [Detection] {
        let (pixels, w, h) = try processor.preprocess(cgImage: cgImage)
        return decode(pixels, w, h, threshold: threshold, polygons: polygons)
    }

    private func decode(_ pixels: MLXArray, _ w: Int, _ h: Int, threshold: Float, polygons: Bool) -> [Detection] {
        let out = model(pixels)
        eval(out.logits, out.predBoxes, out.orderLogits, out.outMasks)
        return PostProcess.decode(output: out, targetSize: (h, w), threshold: threshold,
                                  numClasses: config.numLabels, inputSize: processor.size,
                                  polygons: polygons)
    }

    /// Human-readable label for a class id.
    public func label(_ id: Int) -> String { config.id2label[String(id)] ?? "class_\(id)" }
}
#endif
