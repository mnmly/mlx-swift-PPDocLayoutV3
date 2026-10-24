// Finite-difference gradient check for the bilinear grid_sample VJP.
//
// The forward kernel is exercised by the parity tests, but gradients flow
// through grid_sample only during training. This pins the VJP: the analytic
// gradient of L = sum(gridSample(x, grid) * cot) w.r.t. both x and grid must
// match central finite differences.

import XCTest
import MLX
@testable import MLXPPDocLayoutV3

final class GridSampleGradTests: XCTestCase {
    func testGridSampleVJPMatchesFiniteDifference() {
        // Small deterministic inputs. NHWC feature map and a [-1,1] sampling grid.
        let B = 1, H = 4, W = 5, C = 3, gH = 2, gW = 2
        let xVals = (0..<(B * H * W * C)).map { Float($0) * 0.017 - 0.5 }
        let x = MLXArray(xVals, [B, H, W, C])
        // 4 grid cells × 2 coords, kept away from borders so bilinear is smooth.
        let gridVals: [Float] = [-0.3, 0.2, 0.55, -0.4, 0.1, 0.7, -0.65, -0.15]
        let grid = MLXArray(gridVals, [B, gH, gW, 2])
        let cotVals = (0..<(B * gH * gW * C)).map { Float(($0 % 7)) * 0.31 - 1.0 }
        let cot = MLXArray(cotVals, [B, gH, gW, C])

        func sample(_ p: [MLXArray]) -> [MLXArray] { [gridSample([p[0], p[1]])[0]] }
        func lossAt(_ xa: MLXArray, _ ga: MLXArray) -> Float {
            (gridSample([xa, ga])[0] * cot).sum().item(Float.self)
        }

        // Analytic gradients via the custom VJP (cotangent = cot → grads of the scalar L).
        let (_, grads) = vjp(sample, primals: [x, grid], cotangents: [cot])
        let gx = grads[0].asArray(Float.self)
        let gGrid = grads[1].asArray(Float.self)

        let eps: Float = 1e-3

        // Finite differences w.r.t. every grid coordinate.
        for i in 0..<gridVals.count {
            var plus = gridVals, minus = gridVals
            plus[i] += eps; minus[i] -= eps
            let lp = lossAt(x, MLXArray(plus, [B, gH, gW, 2]))
            let lm = lossAt(x, MLXArray(minus, [B, gH, gW, 2]))
            let fd = (lp - lm) / (2 * eps)
            XCTAssertEqual(gGrid[i], fd, accuracy: 5e-2, "grid grad[\(i)]")
        }

        // Finite differences w.r.t. every feature-map entry.
        for i in 0..<xVals.count {
            var plus = xVals, minus = xVals
            plus[i] += eps; minus[i] -= eps
            let lp = lossAt(MLXArray(plus, [B, H, W, C]), grid)
            let lm = lossAt(MLXArray(minus, [B, H, W, C]), grid)
            let fd = (lp - lm) / (2 * eps)
            XCTAssertEqual(gx[i], fd, accuracy: 5e-3, "x grad[\(i)]")
        }
    }
}
