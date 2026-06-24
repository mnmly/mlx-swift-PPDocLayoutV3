// Actor owning the (non-Sendable) MLX pipeline. Serializes model load + detection
// off the main actor, so the @MainActor view model never captures the pipeline
// across an isolation boundary — Swift-6 data-race-clean without @unchecked.

import Foundation
import CoreGraphics
import MLXPPDocLayoutV3

actor LayoutEngine {
    private var pipeline: PPDocLayoutV3Pipeline?
    private var labels: [Int: String] = [:]

    var isLoaded: Bool { pipeline != nil }

    /// Load weights from a resolved model directory. Returns the id→label map
    /// so the view model can render names synchronously.
    func load(modelDirectory: URL) throws -> [Int: String] {
        let p = try PPDocLayoutV3Pipeline(modelDirectory: modelDirectory)
        pipeline = p
        labels = Dictionary(uniqueKeysWithValues: p.config.id2label.compactMap { k, v in
            Int(k).map { ($0, v) }
        })
        return labels
    }

    /// Detect layout regions in a rendered page. `Detection` is a value type, so
    /// the result crosses back to the caller's isolation safely.
    func detect(cgImage: CGImage, threshold: Float = 0.5) throws -> [Detection] {
        guard let pipeline else { return [] }
        return try pipeline.detect(cgImage: cgImage, threshold: threshold)
    }
}
