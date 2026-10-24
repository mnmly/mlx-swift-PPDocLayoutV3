// Numeric unit tests for the training set criterion: each loss component and
// the matcher are checked against hand-computed values on tiny fixed inputs.
// These need no model checkpoint.

import XCTest
import MLX
@testable import MLXPPDocLayoutV3

final class LossNumericTests: XCTestCase {
    func scalar(_ a: MLXArray) -> Float { a.item(Float.self) }

    // MARK: box conversions

    func testCxcywhXyxyRoundTrip() {
        let cxcywh = MLXArray([Float(0.5), 0.5, 0.4, 0.6], [1, 4])
        let xyxy = BoxOps.cxcywhToXyxy(cxcywh)
        let got = xyxy.reshaped([4]).asArray(Float.self)
        XCTAssertEqual(got[0], 0.3, accuracy: 1e-5)
        XCTAssertEqual(got[1], 0.2, accuracy: 1e-5)
        XCTAssertEqual(got[2], 0.7, accuracy: 1e-5)
        XCTAssertEqual(got[3], 0.8, accuracy: 1e-5)

        let back = BoxOps.xyxyToCxcywh(xyxy).reshaped([4]).asArray(Float.self)
        for (a, b) in zip(back, [Float(0.5), 0.5, 0.4, 0.6]) {
            XCTAssertEqual(a, b, accuracy: 1e-5)
        }
    }

    // MARK: GIoU

    func testGeneralizedIoU() {
        // identical → 1
        let b = MLXArray([Float(0), 0, 1, 1], [1, 4])
        XCTAssertEqual(scalar(BoxOps.generalizedBoxIoU(b, b)), 1.0, accuracy: 1e-5)

        // disjoint unit boxes, enclosing 3x3: giou = -7/9
        let b1 = MLXArray([Float(0), 0, 1, 1], [1, 4])
        let b2 = MLXArray([Float(2), 2, 3, 3], [1, 4])
        XCTAssertEqual(scalar(BoxOps.generalizedBoxIoU(b1, b2)), -7.0 / 9.0, accuracy: 1e-5)

        // partial overlap: iou=1/7, enclosing 3x3 → giou = 1/7 - 2/9
        let c1 = MLXArray([Float(0), 0, 2, 2], [1, 4])
        let c2 = MLXArray([Float(1), 1, 3, 3], [1, 4])
        XCTAssertEqual(scalar(BoxOps.generalizedBoxIoU(c1, c2)), 1.0 / 7.0 - 2.0 / 9.0, accuracy: 1e-5)
    }

    // MARK: BCE / focal

    func testBCEWithLogits() {
        // x=0, y=1 → log(2)
        let l0 = MLXArray([Float(0)]), t1 = MLXArray([Float(1)])
        XCTAssertEqual(scalar(DetectionLoss.bceWithLogits(l0, t1)), Foundation.log(2.0), accuracy: 1e-5)
        // x=2, y=1 → 2 - 2 + log(1+e^-2) = log(1+e^-2)
        let l2 = MLXArray([Float(2)])
        XCTAssertEqual(scalar(DetectionLoss.bceWithLogits(l2, t1)),
                       Foundation.log(1 + Foundation.exp(-2.0)), accuracy: 1e-5)
    }

    func testFocalPerElement() {
        let alpha: Float = 0.25, gamma: Float = 2.0
        // x=0 (p=0.5), y=1: 0.25 * 0.25 * log(2)
        let pos = DetectionLoss.focalPerElement(
            MLXArray([Float(0)]), MLXArray([Float(1)]), alpha: alpha, gamma: gamma)
        XCTAssertEqual(scalar(pos), 0.25 * 0.25 * Foundation.log(2.0), accuracy: 1e-5)
        // x=0, y=0: 0.75 * 0.25 * log(2)
        let neg = DetectionLoss.focalPerElement(
            MLXArray([Float(0)]), MLXArray([Float(0)]), alpha: alpha, gamma: gamma)
        XCTAssertEqual(scalar(neg), 0.75 * 0.25 * Foundation.log(2.0), accuracy: 1e-5)
    }

    // MARK: Hungarian

    func testHungarianSquare() {
        // [[4,1],[2,3]] → row0->col1, row1->col0 (cost 3); returns rowToCol.
        let assigned = HungarianMatcher.hungarian(cost: [4, 1, 2, 3], rows: 2, cols: 2)
        XCTAssertEqual(assigned, [1, 0])
    }

    func testHungarianRectangular() {
        // rows=2, cols=3: [[1,5,3],[4,2,6]] → row0->col0, row1->col1 (cost 3).
        let assigned = HungarianMatcher.hungarian(cost: [1, 5, 3, 4, 2, 6], rows: 2, cols: 3)
        XCTAssertEqual(assigned, [0, 1])
    }

    func testHungarianEmpty() {
        XCTAssertEqual(HungarianMatcher.hungarian(cost: [], rows: 0, cols: 5), [])
    }

    // MARK: VarifocalLoss

    func testVFL() {
        // Q=2, C=1. q0 matched (pred box == GT → IoU 1), q1 background. Both logits 0.
        let logits = MLXArray([Float(0), 0], [2, 1])
        let boxes = MLXArray([Float(0.5), 0.5, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1], [2, 4])
        let target = Target(labels: [0], boxes: MLXArray([Float(0.5), 0.5, 0.2, 0.2], [1, 4]))
        let v = DetectionLoss.vfl(
            logits: logits, boxes: boxes, target: target, pairs: [(0, 0)],
            numClasses: 1, alpha: 0.25, gamma: 2.0)
        // matched: weight 1 → bce(0,1)=log2; background: weight 0.25*0.5^2=0.0625 → 0.0625*log2.
        XCTAssertEqual(scalar(v), Float(Foundation.log(2.0)) * 1.0625, accuracy: 1e-5)
    }

    // MARK: mask loss (BCE + Dice)

    func testMaskLoss() {
        // 1 matched instance, single pixel, logit 0 (p=0.5), GT mask 1.
        let predMasks = MLXArray([Float(0)], [1, 1, 1])
        let target = Target(
            labels: [0], boxes: MLXArray([Float(0.5), 0.5, 0.2, 0.2], [1, 4]),
            masks: MLXArray([Float(1)], [1, 1, 1]))
        let m = DetectionLoss.maskLoss(predMasks: predMasks, target: target, pairs: [(0, 0)])
        // bce = log2; dice = 1 - (2*0.5+1)/(0.5+1+1) = 1 - 2/2.5 = 0.2
        XCTAssertEqual(scalar(m!), Float(Foundation.log(2.0)) + 0.2, accuracy: 1e-5)
    }

    func testMaskLossNilWithoutGT() {
        let predMasks = MLXArray([Float(0)], [1, 1, 1])
        let target = Target(labels: [0], boxes: MLXArray([Float(0.5), 0.5, 0.2, 0.2], [1, 4]))
        XCTAssertNil(DetectionLoss.maskLoss(predMasks: predMasks, target: target, pairs: [(0, 0)]))
    }

    // MARK: order loss (global-pointer multilabel CCE)

    func testOrderLoss() {
        // 2 matched queries, target 0 read before target 1 → cell (0,1) positive.
        // Set that logit to 2.0 → loss = softplus(-2) = log(1+e^-2).
        let orderLogits = MLXArray([Float(0), 2, 0, 0], [2, 2])
        let target = Target(
            labels: [0, 1],
            boxes: MLXArray([Float(0.5), 0.5, 0.2, 0.2, 0.25, 0.3, 0.1, 0.15], [2, 4]),
            orderRanks: [0, 1])
        let o = DetectionLoss.orderLoss(orderLogits: orderLogits, target: target, pairs: [(0, 0), (1, 1)])
        XCTAssertEqual(scalar(o!), Float(Foundation.log(1 + Foundation.exp(-2.0))), accuracy: 1e-4)
    }

    func testOrderLossNilWithoutGT() {
        let orderLogits = MLXArray.zeros([2, 2])
        let target = Target(
            labels: [0, 1],
            boxes: MLXArray([Float(0.5), 0.5, 0.2, 0.2, 0.25, 0.3, 0.1, 0.15], [2, 4]))
        XCTAssertNil(DetectionLoss.orderLoss(orderLogits: orderLogits, target: target, pairs: [(0, 0), (1, 1)]))
    }

    // MARK: matcher end-to-end on a tiny case

    func testMatcherPicksBestQuery() {
        // 3 queries, 4 classes, 1 target (label 2 at box matching query 1).
        let C = 4
        var logitVals = [Float](repeating: -2, count: 3 * C)
        logitVals[1 * C + 2] = 8 // query 1 strongly predicts class 2
        let logits = MLXArray(logitVals, [3, C])

        let tgtBox: [Float] = [0.5, 0.5, 0.2, 0.2]
        let boxes = MLXArray([
            Float(0.1), 0.1, 0.1, 0.1,   // query 0 far
            0.5, 0.5, 0.2, 0.2,          // query 1 == target
            0.9, 0.9, 0.1, 0.1,          // query 2 far
        ], [3, 4])

        let target = Target(labels: [2], boxes: MLXArray(tgtBox, [1, 4]))
        let pairs = HungarianMatcher.match(logits: logits, boxes: boxes, target: target, cost: MatchCost())
        XCTAssertEqual(pairs.count, 1)
        XCTAssertEqual(pairs.first?.0, 1, "best query should be index 1")
        XCTAssertEqual(pairs.first?.1, 0)
    }
}
