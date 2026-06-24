import SwiftUI
import MLXPPDocLayoutV3

/// Draws boxes, polygons, labels and reading-order badges over the image,
/// scaling from original-image pixel coordinates to the displayed frame.
struct DetectionOverlay: View {
    let detections: [Detection]
    let imagePixelSize: CGSize
    let label: (Int) -> String

    private func palette(_ id: Int) -> Color {
        let hues: [Double] = [0.0, 0.58, 0.33, 0.08, 0.78, 0.5, 0.92, 0.15]
        return Color(hue: hues[id % hues.count], saturation: 0.8, brightness: 0.95)
    }

    var body: some View {
        GeometryReader { geo in
            // Aspect-fit the image inside the frame.
            let scale = min(geo.size.width / imagePixelSize.width,
                            geo.size.height / imagePixelSize.height)
            let drawW = imagePixelSize.width * scale
            let drawH = imagePixelSize.height * scale
            let ox = (geo.size.width - drawW) / 2
            let oy = (geo.size.height - drawH) / 2
            let px: (Float, Float) -> CGPoint = { x, y in
                CGPoint(x: ox + CGFloat(x) * scale, y: oy + CGFloat(y) * scale)
            }

            ZStack(alignment: .topLeading) {
                ForEach(Array(detections.enumerated()), id: \.offset) { idx, d in
                    let color = palette(d.label)
                    // polygon (if present) else box
                    if d.polygon.count >= 3 {
                        Path { p in
                            p.move(to: px(d.polygon[0].0, d.polygon[0].1))
                            for pt in d.polygon.dropFirst() { p.addLine(to: px(pt.0, pt.1)) }
                            p.closeSubpath()
                        }
                        .stroke(color, lineWidth: 1.5)
                        .background(Path { p in
                            p.move(to: px(d.polygon[0].0, d.polygon[0].1))
                            for pt in d.polygon.dropFirst() { p.addLine(to: px(pt.0, pt.1)) }
                            p.closeSubpath()
                        }.fill(color.opacity(0.12)))
                    }
                    let tl = px(d.box.0, d.box.1)
                    let br = px(d.box.2, d.box.3)
                    Rectangle()
                        .stroke(color, lineWidth: 1.5)
                        .frame(width: br.x - tl.x, height: br.y - tl.y)
                        .position(x: (tl.x + br.x) / 2, y: (tl.y + br.y) / 2)
                    // order badge + label
                    HStack(spacing: 3) {
                        Text("\(idx + 1)").bold()
                        Text(label(d.label))
                        Text(String(format: "%.2f", d.score)).opacity(0.85)
                    }
                    .font(.system(size: 10, weight: .semibold))
                    .padding(.horizontal, 4).padding(.vertical, 1)
                    .background(color).foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 3))
                    .position(x: tl.x + 2, y: tl.y - 8)
                    .fixedSize()
                }
            }
        }
    }
}
