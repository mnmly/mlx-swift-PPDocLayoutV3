// swift-tools-version: 6.0
// Port of PP-DocLayoutV3 (PaddlePaddle / transformers) → mlx-swift.
// RT-DETR (HGNetV2-L backbone + hybrid encoder + deformable decoder) with
// mask-feature head and reading-order global pointer.
// PORT FROM: transformers/models/pp_doclayout_v3 (reference/python/)

import PackageDescription

let package = Package(
    name: "mlx-swift-PPDocLayoutV3",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "MLXPPDocLayoutV3", targets: ["MLXPPDocLayoutV3"]),
        .executable(name: "PPDocLayoutBench", targets: ["PPDocLayoutBench"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.32.3")),
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3"),
    ],
    targets: [
        .target(
            name: "MLXPPDocLayoutV3",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXPPDocLayoutV3Tests",
            dependencies: ["MLXPPDocLayoutV3"]
        ),
        .executableTarget(
            name: "PPDocLayoutBench",
            dependencies: [
                "MLXPPDocLayoutV3",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
    ]
)
