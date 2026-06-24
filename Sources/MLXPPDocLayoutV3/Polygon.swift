// Mask → polygon extraction (Swift replacement for cv2 findContours/approxPolyDP).
//
// PORT FROM: reference/python/modular_pp_doclayout_v3.py
//   (_extract_polygon_points_by_masks, _mask2polygon, extract_custom_vertices)
// cv2 is replaced with: largest-connected-component boundary trace (Moore) +
// Douglas–Peucker simplification. Functionally equivalent, not bit-identical.

import Foundation

enum Polygon {
    typealias Pt = (Float, Float)

    /// Largest 8-connected component as a binary grid (row-major, w×h).
    private static func largestComponent(_ mask: [Bool], _ w: Int, _ h: Int) -> [Bool]? {
        var label = [Int](repeating: 0, count: w * h)
        var best = -1, bestSize = 0, cur = 0
        var stack: [Int] = []
        let nbr8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        for start in 0..<(w * h) where mask[start] && label[start] == 0 {
            cur += 1
            var size = 0
            stack.removeAll(keepingCapacity: true)
            stack.append(start); label[start] = cur
            while let p = stack.popLast() {
                size += 1
                let px = p % w, py = p / w
                for (dx, dy) in nbr8 {
                    let nx = px + dx, ny = py + dy
                    if nx < 0 || ny < 0 || nx >= w || ny >= h { continue }
                    let np = ny * w + nx
                    if mask[np] && label[np] == 0 { label[np] = cur; stack.append(np) }
                }
            }
            if size > bestSize { bestSize = size; best = cur }
        }
        if best < 0 { return nil }
        return (0..<(w * h)).map { label[$0] == best }
    }

    /// Moore-neighbor boundary trace of a single 8-connected blob. Clockwise.
    private static func traceBoundary(_ blob: [Bool], _ w: Int, _ h: Int) -> [Pt] {
        // find start: first foreground in raster order
        guard let startIdx = (0..<(w * h)).first(where: { blob[$0] }) else { return [] }
        let sx = startIdx % w, sy = startIdx / w
        func fg(_ x: Int, _ y: Int) -> Bool { x >= 0 && y >= 0 && x < w && y < h && blob[y * w + x] }
        // 8 directions clockwise starting from E
        let dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        var contour: [Pt] = []
        var cx = sx, cy = sy
        var backDir = 4 // came from W initially
        let maxSteps = 4 * (w * h) + 10
        var steps = 0
        repeat {
            contour.append((Float(cx), Float(cy)))
            var found = false
            // search neighbors clockwise starting just after the back direction
            for k in 0..<8 {
                let d = (backDir + 1 + k) % 8
                let nx = cx + dirs[d].0, ny = cy + dirs[d].1
                if fg(nx, ny) {
                    backDir = (d + 4) % 8 // direction back to current from neighbor
                    cx = nx; cy = ny
                    found = true
                    break
                }
            }
            if !found { break } // isolated pixel
            steps += 1
        } while !(cx == sx && cy == sy) && steps < maxSteps
        return contour
    }

    /// Perimeter of a closed polyline.
    private static func arcLength(_ pts: [Pt]) -> Float {
        guard pts.count > 1 else { return 0 }
        var total: Float = 0
        for i in 0..<pts.count {
            let a = pts[i], b = pts[(i + 1) % pts.count]
            total += (Float(pow(Double(b.0 - a.0), 2) + pow(Double(b.1 - a.1), 2))).squareRoot()
        }
        return total
    }

    /// Douglas–Peucker simplification of an open polyline.
    private static func douglasPeucker(_ pts: [Pt], _ eps: Float) -> [Pt] {
        if pts.count < 3 { return pts }
        var keep = [Bool](repeating: false, count: pts.count)
        keep[0] = true; keep[pts.count - 1] = true
        var stack: [(Int, Int)] = [(0, pts.count - 1)]
        while let (s, e) = stack.popLast() {
            var maxD: Float = 0, idx = -1
            let a = pts[s], b = pts[e]
            let dx = b.0 - a.0, dy = b.1 - a.1
            let denom = (dx * dx + dy * dy).squareRoot()
            for i in (s + 1)..<e {
                let p = pts[i]
                let d: Float
                if denom < 1e-6 {
                    d = ((p.0 - a.0) * (p.0 - a.0) + (p.1 - a.1) * (p.1 - a.1)).squareRoot()
                } else {
                    d = abs(dy * p.0 - dx * p.1 + b.0 * a.1 - b.1 * a.0) / denom
                }
                if d > maxD { maxD = d; idx = i }
            }
            if maxD > eps && idx >= 0 {
                keep[idx] = true
                stack.append((s, idx)); stack.append((idx, e))
            }
        }
        return (0..<pts.count).filter { keep[$0] }.map { pts[$0] }
    }

    /// extract_custom_vertices: adjust ~45° sharp concave corners.
    private static func customVertices(_ poly: [Pt], sharpAngleThresh: Float = 45) -> [Pt] {
        let n = poly.count
        if n < 3 { return poly }
        var res: [Pt] = []
        for i in 0..<n {
            let prev = poly[(i - 1 + n) % n], cur = poly[i], next = poly[(i + 1) % n]
            let v1 = (prev.0 - cur.0, prev.1 - cur.1)
            let v2 = (next.0 - cur.0, next.1 - cur.1)
            let cross = v1.1 * v2.0 - v1.0 * v2.1
            if cross < 0 {
                let n1 = (v1.0 * v1.0 + v1.1 * v1.1).squareRoot()
                let n2 = (v2.0 * v2.0 + v2.1 * v2.1).squareRoot()
                if n1 < 1e-6 || n2 < 1e-6 { res.append(cur); continue }
                let cosA = max(-1, min(1, (v1.0 * v2.0 + v1.1 * v2.1) / (n1 * n2)))
                let angle = acos(cosA) * 180 / .pi
                if abs(angle - sharpAngleThresh) < 1 {
                    var dir = (v1.0 / n1 + v2.0 / n2, v1.1 / n1 + v2.1 / n2)
                    let dn = (dir.0 * dir.0 + dir.1 * dir.1).squareRoot()
                    if dn > 1e-6 {
                        dir = (dir.0 / dn, dir.1 / dn)
                        let step = (n1 + n2) / 2
                        res.append((cur.0 + dir.0 * step, cur.1 + dir.1 * step))
                    } else { res.append(cur) }
                } else { res.append(cur) }
            }
        }
        return res
    }

    /// Convert a binary mask (row-major w×h) to polygon points; nil if empty.
    static func mask2polygon(_ mask: [Bool], _ w: Int, _ h: Int, epsilonRatio: Float = 0.004) -> [Pt]? {
        guard let blob = largestComponent(mask, w, h) else { return nil }
        let contour = traceBoundary(blob, w, h)
        if contour.count < 3 { return nil }
        let eps = epsilonRatio * arcLength(contour)
        let approx = douglasPeucker(contour, eps)
        return customVertices(approx)
    }

    /// Full per-box polygon extraction (mirrors _extract_polygon_points_by_masks).
    /// - boxes: xyxy in target-image pixels.
    /// - masks: per-detection binary masks at (maskH, maskW).
    /// - scaleRatio: (size_w/target_w, size_h/target_h).
    static func extractPolygons(
        boxes: [(Float, Float, Float, Float)], masks: [[Bool]], maskW: Int, maskH: Int,
        scaleRatio: (Float, Float)
    ) -> [[Pt]] {
        let scaleW = scaleRatio.0 / 4, scaleH = scaleRatio.1 / 4
        var out: [[Pt]] = []
        for (i, box) in boxes.enumerated() {
            let xMin = Int(box.0), yMin = Int(box.1), xMax = Int(box.2), yMax = Int(box.3)
            let boxW = xMax - xMin, boxH = yMax - yMin
            let rect: [Pt] = [(Float(xMin), Float(yMin)), (Float(xMax), Float(yMin)),
                              (Float(xMax), Float(yMax)), (Float(xMin), Float(yMax))]
            if boxW <= 0 || boxH <= 0 { out.append(rect); continue }
            let xs = max(0, min(maskW, Int((Float(xMin) * scaleW).rounded())))
            let xe = max(0, min(maskW, Int((Float(xMax) * scaleW).rounded())))
            let ys = max(0, min(maskH, Int((Float(yMin) * scaleH).rounded())))
            let ye = max(0, min(maskH, Int((Float(yMax) * scaleH).rounded())))
            if xe <= xs || ye <= ys { out.append(rect); continue }
            // crop
            let cw = xe - xs, ch = ye - ys
            var crop = [Bool](repeating: false, count: cw * ch)
            let full = masks[i]
            var any = false
            for y in 0..<ch {
                for x in 0..<cw {
                    let v = full[(ys + y) * maskW + (xs + x)]
                    crop[y * cw + x] = v
                    any = any || v
                }
            }
            if !any { out.append(rect); continue }
            // nearest-resize crop to (boxW, boxH)
            var resized = [Bool](repeating: false, count: boxW * boxH)
            for y in 0..<boxH {
                let sy = min(ch - 1, Int(Float(y) / Float(boxH) * Float(ch)))
                for x in 0..<boxW {
                    let sx = min(cw - 1, Int(Float(x) / Float(boxW) * Float(cw)))
                    resized[y * boxW + x] = crop[sy * cw + sx]
                }
            }
            guard var poly = mask2polygon(resized, boxW, boxH), poly.count >= 4 else {
                out.append(rect); continue
            }
            poly = poly.map { ($0.0 + Float(xMin), $0.1 + Float(yMin)) }
            out.append(poly)
        }
        return out
    }
}
