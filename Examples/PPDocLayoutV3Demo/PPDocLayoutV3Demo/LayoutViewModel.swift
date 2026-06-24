// Presentation-side driver. All compute is in MLXPPDocLayoutV3 (ModelStore +
// PPDocLayoutV3Pipeline). This view model owns only UI state + Task hops.

import SwiftUI
import AppKit
import PDFKit
import Observation
import UniformTypeIdentifiers
import MLXPPDocLayoutV3

/// One renderable page (a still image, or a rendered PDF page) + its detections.
struct DocPage: Identifiable {
    let id = UUID()
    let nsImage: NSImage
    let cgImage: CGImage
    let pixelSize: CGSize
    var detections: [Detection] = []
    var detected = false
}

@MainActor
@Observable
final class LayoutViewModel {
    enum ModelState: Equatable {
        case checking, missing, downloading(Double), loading, ready, error(String)
    }

    var modelState: ModelState = .checking
    var modelDirText: String = ""
    var pages: [DocPage] = []
    var currentPage: Int = 0
    var documentName: String = ""
    var isDetecting = false
    var detectProgress: String = ""
    /// Live display threshold. Detection runs at `detectFloor`; the UI filters up.
    var confidence: Double = 0.5
    static let detectFloor: Float = 0.25

    /// Current page's detections filtered to the live confidence threshold.
    var visibleDetections: [Detection] {
        (page?.detections ?? []).filter { $0.score >= Float(confidence) }
    }

    @ObservationIgnored private let engine = LayoutEngine()
    @ObservationIgnored private var labels: [Int: String] = [:]
    @ObservationIgnored private var modelDirectory: URL?

    var page: DocPage? { pages.indices.contains(currentPage) ? pages[currentPage] : nil }

    // MARK: - Model resolution

    func checkModel() {
        modelState = .checking
        if let dir = ModelStore.resolveExistingModelDirectory() {
            modelDirectory = dir
            modelDirText = dir.path
            loadPipeline(from: dir)
        } else {
            modelState = .missing
            modelDirText = "Not found in ~/.cache/huggingface"
        }
    }

    private func loadPipeline(from dir: URL) {
        modelState = .loading
        Task {
            do {
                labels = try await engine.load(modelDirectory: dir)
                modelState = .ready
            } catch {
                modelState = .error("load failed: \(error.localizedDescription)")
            }
        }
    }

    // MARK: - Download to a user-picked, bookmarked directory

    func chooseDownloadLocationAndDownload() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Download Here"
        panel.message = "Choose a folder to store the PP-DocLayoutV3 model"
        guard panel.runModal() == .OK, let dir = panel.url else { return }
        ModelStore.saveBookmark(for: dir)
        let target = dir.appending(path: "PP-DocLayoutV3")
        modelState = .downloading(0)
        Task {
            do {
                let didAccess = dir.startAccessingSecurityScopedResource()
                defer { if didAccess { dir.stopAccessingSecurityScopedResource() } }
                let populated = try await ModelStore.download(into: target) { frac in
                    Task { @MainActor in self.modelState = .downloading(frac) }
                }
                ModelStore.saveBookmark(for: populated)
                modelDirectory = populated
                modelDirText = populated.path
                loadPipeline(from: populated)
            } catch {
                modelState = .error("download failed: \(error.localizedDescription)")
            }
        }
    }

    /// Point at an already-downloaded model dir (or a HF `models--…` cache folder).
    func chooseExistingModelDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Use Folder"
        guard panel.runModal() == .OK, let picked = panel.url else { return }
        guard let dir = ModelStore.resolveModelDirectory(pickedAt: picked) else {
            modelState = .error("no config.json / model.safetensors found (also checked snapshots/)")
            return
        }
        ModelStore.saveBookmark(for: picked)
        modelDirectory = dir
        modelDirText = dir.path
        loadPipeline(from: dir)
    }

    // MARK: - Document import (image or PDF)

    func pickDocument() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.pdf, .png, .jpeg, .tiff, .bmp, .image]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        documentName = url.lastPathComponent
        currentPage = 0
        if url.pathExtension.lowercased() == "pdf" {
            pages = Self.renderPDF(url: url)
        } else if let img = NSImage(contentsOf: url),
                  let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) {
            pages = [DocPage(nsImage: img, cgImage: cg,
                             pixelSize: CGSize(width: cg.width, height: cg.height))]
        } else {
            pages = []
        }
    }

    /// Render every PDF page to an upright CGImage at 2x.
    private static func renderPDF(url: URL, scale: CGFloat = 2.0) -> [DocPage] {
        guard let doc = PDFDocument(url: url) else { return [] }
        var out: [DocPage] = []
        for i in 0..<doc.pageCount {
            guard let page = doc.page(at: i) else { continue }
            let rect = page.bounds(for: .mediaBox)
            let w = Int(rect.width * scale), h = Int(rect.height * scale)
            guard w > 0, h > 0,
                  let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                      bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { continue }
            ctx.setFillColor(CGColor(gray: 1, alpha: 1))
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            // macOS CGBitmapContext is bottom-left origin (matches PDF space) — no
            // vertical flip needed; just scale and offset the page's mediaBox.
            ctx.scaleBy(x: scale, y: scale)
            ctx.translateBy(x: -rect.minX, y: -rect.minY)
            page.draw(with: .mediaBox, to: ctx)
            guard let cg = ctx.makeImage() else { continue }
            let ns = NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
            out.append(DocPage(nsImage: ns, cgImage: cg,
                               pixelSize: CGSize(width: cg.width, height: cg.height)))
        }
        return out
    }

    // MARK: - Detection (all pages)

    func detectAll() {
        guard !pages.isEmpty, !isDetecting else { return }
        isDetecting = true
        let cgImages = pages.map { $0.cgImage }
        Task {
            for (i, cg) in cgImages.enumerated() {
                if Task.isCancelled { break }
                detectProgress = "Detecting page \(i + 1)/\(cgImages.count)…"
                let dets = (try? await engine.detect(cgImage: cg, threshold: Self.detectFloor)) ?? []
                if pages.indices.contains(i) {
                    pages[i].detections = dets
                    pages[i].detected = true
                }
            }
            isDetecting = false
            detectProgress = ""
        }
    }

    func label(_ id: Int) -> String { labels[id] ?? "class_\(id)" }
}
