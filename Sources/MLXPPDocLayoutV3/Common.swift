// Shared building blocks for PP-DocLayoutV3 (NHWC tensors throughout).
//
// PORT FROM: reference/python/modeling_pp_doclayout_v3.py
//   (PPDocLayoutV3FrozenBatchNorm2d, PPDocLayoutV3ConvNormLayer,
//    PPDocLayoutV3ConvLayer, PPDocLayoutV3MLPPredictionHead)

import Foundation
import MLX
import MLXNN

// MARK: - Activations

/// Map a transformers activation name to a unary closure.
func activation(_ name: String?) -> (MLXArray) -> MLXArray {
    switch name {
    case nil, "identity": return { $0 }
    case "relu": return { relu($0) }
    case "silu", "swish": return { silu($0) }
    case "gelu": return { gelu($0) }            // exact (erf) gelu — matches torch nn.GELU()
    default: fatalError("unsupported activation \(name!)")
    }
}

// MARK: - Frozen BatchNorm (eval-mode affine over channels-last)

/// BatchNorm2d with fixed running statistics, applied channel-last:
/// `out = x * (weight / sqrt(var + eps)) + (bias - mean * scale)`.
public final class FrozenBatchNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray
    let eps: Float

    public init(_ channels: Int, eps: Float = 1e-5) {
        self.eps = eps
        self._weight = ParameterInfo(wrappedValue: MLXArray.ones([channels]), key: "weight")
        self._bias = ParameterInfo(wrappedValue: MLXArray.zeros([channels]), key: "bias")
        self._runningMean = ParameterInfo(wrappedValue: MLXArray.zeros([channels]), key: "running_mean")
        self._runningVar = ParameterInfo(wrappedValue: MLXArray.ones([channels]), key: "running_var")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Center first: (x - mean) * scale + bias. The algebraically-equivalent
        // `x*scale + (bias - mean*scale)` catastrophically cancels in fp32 when
        // x ≈ running_mean (the BN-centered case), which diverges from PyTorch.
        let scale = weight * rsqrt(runningVar + eps)
        return (x - runningMean) * scale + bias
    }
}

// MARK: - Conv + BN + activation

/// Conv2d (NHWC, no bias) → FrozenBatchNorm → activation.
/// Swift sub-keys `conv` / `norm`; HF `convolution`/`normalization` are remapped
/// to these in the weight loader.
public final class ConvNorm: Module, UnaryLayer {
    @ModuleInfo(key: "conv") public var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: FrozenBatchNorm
    let act: (MLXArray) -> MLXArray
    let groups: Int

    public init(
        _ inCh: Int, _ outCh: Int, kernelSize: Int, stride: Int = 1,
        padding: Int? = nil, groups: Int = 1, activation actName: String? = "relu",
        bnEps: Float = 1e-5
    ) {
        let pad = padding ?? ((kernelSize - 1) / 2)
        self.groups = groups
        self._conv = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: inCh, outputChannels: outCh,
                kernelSize: .init(kernelSize), stride: .init(stride),
                padding: .init(pad), groups: groups, bias: false),
            key: "conv")
        self._norm = ModuleInfo(wrappedValue: FrozenBatchNorm(outCh, eps: bnEps), key: "norm")
        self.act = activation(actName)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        act(norm(conv(x)))
    }
}

// MARK: - MLP prediction head

/// Multi-layer perceptron with ReLU between layers (last layer linear).
/// Mirrors `PPDocLayoutV3MLPPredictionHead`.
public final class MLPPredictionHead: Module {
    @ModuleInfo(key: "layers") var layers: [Linear]
    let numLayers: Int

    public init(_ inputDim: Int, _ hiddenDim: Int, _ outputDim: Int, numLayers: Int) {
        self.numLayers = numLayers
        let dims = [inputDim] + Array(repeating: hiddenDim, count: numLayers - 1)
        let outs = Array(repeating: hiddenDim, count: numLayers - 1) + [outputDim]
        var built: [Linear] = []
        for i in 0..<numLayers { built.append(Linear(dims[i], outs[i])) }
        self._layers = ModuleInfo(wrappedValue: built, key: "layers")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for (i, layer) in layers.enumerated() {
            h = (i < numLayers - 1) ? relu(layer(h)) : layer(h)
        }
        return h
    }
}

// MARK: - Spatial helpers (NHWC)

/// Bilinear resize an NHWC tensor to (targetH, targetW), align_corners=false.
func bilinearResize(_ x: MLXArray, _ targetH: Int, _ targetW: Int) -> MLXArray {
    let H = x.dim(1), W = x.dim(2)
    if H == targetH && W == targetW { return x }
    let up = Upsample(
        scaleFactor: .array([Float(targetH) / Float(H), Float(targetW) / Float(W)]),
        mode: .linear(alignCorners: false))
    return up(x)
}

/// Nearest-neighbour 2x upsample of an NHWC tensor.
func nearestUpsample2x(_ x: MLXArray) -> MLXArray {
    let up = Upsample(scaleFactor: .float(2.0), mode: .nearest)
    return up(x)
}
