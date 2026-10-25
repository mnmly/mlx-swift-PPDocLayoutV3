// Model resolution + download, shared by the CLI and the SwiftUI app.
//
// A "model directory" is any folder containing `config.json` + `model.safetensors`.
// Resolution order: (1) a user-picked directory persisted via a security-scoped
// bookmark, (2) the local HuggingFace cache. If neither exists, `download(into:)`
// fetches the files from the Hub.

import Foundation

public enum ModelStore {
    public static let repo = "PaddlePaddle/PP-DocLayoutV3_safetensors"
    public static let requiredFiles = ["config.json", "model.safetensors"]
    public static let allFiles = ["config.json", "model.safetensors", "preprocessor_config.json"]
    static let bookmarkKey = "PPDocLayoutV3.modelDirBookmark"

    /// True if `dir` contains the required model files.
    public static func isValidModelDirectory(_ dir: URL) -> Bool {
        requiredFiles.allSatisfy { FileManager.default.fileExists(atPath: dir.appending(path: $0).path) }
    }

    /// Resolve a user-picked folder to the actual model directory. Accepts the
    /// model dir itself, OR a HuggingFace cache `models--…` folder (whose files
    /// live under `snapshots/<hash>/`).
    public static func resolveModelDirectory(pickedAt url: URL) -> URL? {
        if isValidModelDirectory(url) { return url }
        // HF cache layout: <picked>/snapshots/<hash>/{config.json,model.safetensors}
        let snapshots = url.appending(path: "snapshots")
        if let entries = try? FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil) {
            if let match = entries.first(where: { isValidModelDirectory($0) }) { return match }
        }
        return nil
    }

    /// Locate an already-available model directory, or nil.
    public static func resolveExistingModelDirectory() -> URL? {
        if let bookmarked = bookmarkedDirectory(), let m = resolveModelDirectory(pickedAt: bookmarked) { return m }
        if let cache = huggingFaceCacheSnapshot(), isValidModelDirectory(cache) { return cache }
        return nil
    }

    /// The newest HF-cache snapshot dir for this repo that has the weights.
    public static func huggingFaceCacheSnapshot() -> URL? {
        #if os(macOS)
        let home = FileManager.default.homeDirectoryForCurrentUser
        #else
        let home = URL(filePath: NSHomeDirectory())
        #endif
        let snaps = home.appending(path: ".cache/huggingface/hub/models--PaddlePaddle--PP-DocLayoutV3_safetensors/snapshots")
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: snaps, includingPropertiesForKeys: [.contentModificationDateKey]) else { return nil }
        return entries.first { isValidModelDirectory($0) }
    }

    // MARK: - Download

    /// Download the model files from the Hub into `dir`. Reports 0...1 progress.
    /// Returns the populated directory.
    @discardableResult
    public static func download(
        into dir: URL, progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) async throws -> URL {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // Weight the model.safetensors as the bulk of the work.
        let weights: [String: Double] = ["config.json": 0.005, "preprocessor_config.json": 0.005, "model.safetensors": 0.99]
        var done = 0.0
        for file in allFiles {
            let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(file)")!
            let dest = dir.appending(path: file)
            let base = done                          // immutable snapshot for the closure
            let weight = weights[file] ?? 0
            try await downloadFile(from: url, to: dest) { frac in
                progress(base + weight * frac)
            }
            done += weight
            progress(done)
        }
        progress(1.0)
        return dir
    }

    private static func downloadFile(
        from url: URL, to dest: URL, progress: @Sendable @escaping (Double) -> Void
    ) async throws {
        let (bytes, response) = try await URLSession.shared.bytes(from: url)
        let total = response.expectedContentLength
        FileManager.default.createFile(atPath: dest.path, contents: nil)
        let handle = try FileHandle(forWritingTo: dest)
        defer { try? handle.close() }
        var buffer = Data()
        buffer.reserveCapacity(1 << 20)
        var received: Int64 = 0
        for try await byte in bytes {
            buffer.append(byte)
            received += 1
            if buffer.count >= (1 << 20) {
                try handle.write(contentsOf: buffer)
                buffer.removeAll(keepingCapacity: true)
                if total > 0 { progress(Double(received) / Double(total)) }
            }
        }
        if !buffer.isEmpty { try handle.write(contentsOf: buffer) }
        progress(1.0)
    }

    // MARK: - Security-scoped bookmark (sandbox)

    /// Persist access to a user-chosen directory for subsequent launches.
    public static func saveBookmark(for url: URL) {
        #if os(macOS)
        let didAccess = url.startAccessingSecurityScopedResource()
        defer { if didAccess { url.stopAccessingSecurityScopedResource() } }
        if let data = try? url.bookmarkData(options: [.withSecurityScope],
                                            includingResourceValuesForKeys: nil, relativeTo: nil) {
            UserDefaults.standard.set(data, forKey: bookmarkKey)
        }
        #else
        if let data = try? url.bookmarkData() { UserDefaults.standard.set(data, forKey: bookmarkKey) }
        #endif
    }

    /// Resolve the bookmarked directory and begin security-scoped access.
    /// Caller should `stopAccessingSecurityScopedResource()` when done.
    public static func bookmarkedDirectory() -> URL? {
        guard let data = UserDefaults.standard.data(forKey: bookmarkKey) else { return nil }
        var stale = false
        #if os(macOS)
        guard let url = try? URL(resolvingBookmarkData: data, options: [.withSecurityScope],
                                 relativeTo: nil, bookmarkDataIsStale: &stale) else { return nil }
        _ = url.startAccessingSecurityScopedResource()
        #else
        guard let url = try? URL(resolvingBookmarkData: data, relativeTo: nil,
                                 bookmarkDataIsStale: &stale) else { return nil }
        #endif
        return url
    }
}
