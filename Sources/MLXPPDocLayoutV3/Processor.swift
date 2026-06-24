// Image preprocessing: load → resize 800x800 (bicubic) → rescale 1/255 → NHWC.
//
// PORT FROM: preprocessor_config.json (size 800, resample BICUBIC, rescale 1/255,
//            normalize mean[0,0,0] std[1,1,1] = identity).

import Foundation
import MLX
import MLXNN

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

public struct ImageProcessor: Sendable {
    public let size: Int
    public init(size: Int = 800) { self.size = size }

    /// Decode a `CGImage` into an NHWC RGB MLXArray at native resolution, [0,255].
    public func loadRGB(cgImage cg: CGImage) throws -> (MLXArray, Int, Int) {
        let w = cg.width, h = cg.height
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw NSError(domain: "ImageProcessor", code: 2)
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        let rgba = MLXArray(buf, [h, w, 4]).asType(.float32)
        let rgb = rgba[0..., 0..., 0..<3].reshaped([1, h, w, 3])
        return (rgb, w, h)
    }

    /// Load an image file into an NHWC RGB MLXArray at native resolution, [0,255].
    public func loadRGB(url: URL) throws -> (MLXArray, Int, Int) {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw NSError(domain: "ImageProcessor", code: 1, userInfo: [NSLocalizedDescriptionKey: "cannot decode \(url.path)"])
        }
        return try loadRGB(cgImage: cg)
    }

    /// Preprocess a file into model input: (1, 800, 800, 3) float, /255.
    public func preprocess(url: URL) throws -> (pixelValues: MLXArray, width: Int, height: Int) {
        let (rgb, w, h) = try loadRGB(url: url)
        return (bicubicResize(rgb, size, size) / 255.0, w, h)
    }

    /// Preprocess a `CGImage` (e.g. a rendered PDF page) into model input.
    public func preprocess(cgImage cg: CGImage) throws -> (pixelValues: MLXArray, width: Int, height: Int) {
        let (rgb, w, h) = try loadRGB(cgImage: cg)
        return (bicubicResize(rgb, size, size) / 255.0, w, h)
    }

    private func bicubicResize(_ x: MLXArray, _ th: Int, _ tw: Int) -> MLXArray {
        let H = x.dim(1), W = x.dim(2)
        if H == th && W == tw { return x }
        let up = Upsample(scaleFactor: .array([Float(th) / Float(H), Float(tw) / Float(W)]),
                          mode: .cubic(alignCorners: false))
        return up(x)
    }
}
#endif
