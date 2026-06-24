import SwiftUI
import MLXPPDocLayoutV3

struct ContentView: View {
    @State private var vm = LayoutViewModel()

    var body: some View {
        VStack(spacing: 0) {
            modelBar
            Divider()
            HStack(spacing: 0) {
                VStack(spacing: 0) {
                    imagePane
                    if vm.pages.count > 1 { Divider(); pageBar }
                }
                Divider()
                detectionList.frame(width: 260)
            }
        }
        .onAppear { vm.checkModel() }
    }

    // MARK: - Model status / actions bar

    @ViewBuilder private var modelBar: some View {
        HStack(spacing: 10) {
            switch vm.modelState {
            case .checking, .loading:
                ProgressView().controlSize(.small)
                Text(vm.modelState == .loading ? "Loading model…" : "Checking…")
            case .missing:
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                Text("Model not found").bold()
                Button("Download…") { vm.chooseDownloadLocationAndDownload() }
                    .buttonStyle(.borderedProminent)
                Button("Choose Folder…") { vm.chooseExistingModelDirectory() }
            case .downloading(let p):
                ProgressView(value: p).frame(width: 160)
                Text("Downloading \(Int(p * 100))%")
            case .ready:
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                Button("Open Image / PDF…") { vm.pickDocument() }
                Button("Detect") { vm.detectAll() }
                    .buttonStyle(.borderedProminent)
                    .disabled(vm.pages.isEmpty || vm.isDetecting)
                if vm.isDetecting {
                    ProgressView().controlSize(.small)
                    Text(vm.detectProgress).font(.caption)
                }
                if vm.page?.detected == true {
                    Divider().frame(height: 16)
                    Image(systemName: "slider.horizontal.3").foregroundStyle(.secondary)
                    Slider(value: $vm.confidence, in: Double(LayoutViewModel.detectFloor)...0.95)
                        .frame(width: 120)
                    Text(String(format: "≥ %.2f", vm.confidence)).font(.caption).monospacedDigit()
                }
            case .error(let msg):
                Image(systemName: "xmark.octagon.fill").foregroundStyle(.red)
                Text(msg).lineLimit(1)
                Button("Retry") { vm.checkModel() }
            }
            Spacer()
            Text(vm.documentName.isEmpty ? vm.modelDirText : vm.documentName)
                .font(.caption).foregroundStyle(.secondary).lineLimit(1)
        }
        .padding(10)
    }

    // MARK: - Image + overlay

    @ViewBuilder private var imagePane: some View {
        ZStack {
            Color(nsColor: .textBackgroundColor)
            if let page = vm.page {
                Image(nsImage: page.nsImage)
                    .resizable().aspectRatio(contentMode: .fit)
                    .overlay(DetectionOverlay(detections: vm.visibleDetections,
                                              imagePixelSize: page.pixelSize,
                                              label: vm.label))
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "doc.text.image").font(.system(size: 48)).foregroundStyle(.secondary)
                    Text("Open a document image or PDF to detect its layout").foregroundStyle(.secondary)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Page navigation (PDF)

    @ViewBuilder private var pageBar: some View {
        HStack(spacing: 14) {
            Button { vm.currentPage = max(0, vm.currentPage - 1) } label: {
                Image(systemName: "chevron.left")
            }.disabled(vm.currentPage == 0)
            Text("Page \(vm.currentPage + 1) / \(vm.pages.count)").monospacedDigit()
            Button { vm.currentPage = min(vm.pages.count - 1, vm.currentPage + 1) } label: {
                Image(systemName: "chevron.right")
            }.disabled(vm.currentPage >= vm.pages.count - 1)
            if let p = vm.page, p.detected {
                Text("· \(vm.visibleDetections.count) regions").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(.horizontal, 12).padding(.vertical, 7)
    }

    // MARK: - Detection list (reading order, current page)

    @ViewBuilder private var detectionList: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Reading order (\(vm.visibleDetections.count))")
                .font(.headline).padding(10)
            Divider()
            List(Array(vm.visibleDetections.enumerated()), id: \.offset) { idx, d in
                HStack {
                    Text("\(idx + 1)").bold().frame(width: 22, alignment: .trailing)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(vm.label(d.label)).font(.callout)
                        Text(String(format: "%.0f, %.0f, %.0f, %.0f", d.box.0, d.box.1, d.box.2, d.box.3))
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text(String(format: "%.2f", d.score)).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }
}

#Preview { ContentView() }
