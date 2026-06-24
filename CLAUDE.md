# mlx-swift-PPDocLayoutV3 — agent guide

MLX-Swift port of PaddlePaddle **PP-DocLayoutV3** (RT-DETR + HGNetV2-L + mask /
reading-order heads). Library: `Sources/MLXPPDocLayoutV3`. Demo app:
`Examples/PPDocLayoutV3Demo` (Xcode project). CLI/bench: `Sources/PPDocLayoutBench`.

## Build & test

- Use **`xcodebuild`**, never `swift build`/`swift test` (they can't load mlx-swift's
  Metal library).
- Tests: `xcodebuild -scheme mlx-swift-PPDocLayoutV3-Package -destination 'platform=macOS' test`
- App: `xcodebuild -project Examples/PPDocLayoutV3Demo/PPDocLayoutV3Demo.xcodeproj -scheme PPDocLayoutV3Demo -destination 'platform=macOS' build`

## Parity invariant

Gate on **detection-level** parity (labels exact, scores ≤5e-3, boxes ≤2px), NOT
exact intermediate tensors — encoder feature maps drift ~0.1–0.2 from inherent fp32
GPU-vs-CPU conv accumulation (the conv weights load bit-exact). Regenerate fixtures
with `reference/dump_fixtures.py`.

## Conventions

- **State/observation:** SwiftUI view models use the **Observation framework**
  (`@Observable`, `@State`), never Combine / `ObservableObject` / `@Published`.
- **Concurrency:** both targets build under Swift 6 strict concurrency. Keep it that
  way — the non-Sendable MLX pipeline lives behind the `LayoutEngine` actor; don't
  introduce `@unchecked Sendable` or capture the model into `Task.detached`.
- **Tensor layout:** NHWC throughout; PyTorch conv weights transpose `(0,2,3,1)` at load.

## Documentation

Every `public`/`open` symbol gets a `///` doc comment. New top-level public symbols
get filed under a `## Topics` group in
`Sources/MLXPPDocLayoutV3/Documentation.docc/MLXPPDocLayoutV3.md`. DocC link syntax is
signature-sensitive (`` ``foo(_:)`` ``); document parameters by their *internal* name.
Verify with `Scripts/build_docs.sh` (no unresolved-link warnings on authored prose).
