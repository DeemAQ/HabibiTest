// swift-tools-version: 5.10
// F5-TTS v1 + Vocos with MLX. Based on lucasnewman/f5-tts-swift and vocos-swift (MIT)

import PackageDescription

let package = Package(
    name: "F5Engine",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "F5Engine", targets: ["F5Engine"]),
    ],
    dependencies: [
        // 0.29.x still builds with Xcode 16
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.29.1"),
    ],
    targets: [
        .target(
            name: "F5Engine",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
    ]
)
