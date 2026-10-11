import Foundation
import MLX
import MLXNN

// Vocos (charactr/vocos-mel-24khz), based on lucasnewman/vocos-swift (MIT).

final class VocosConvNeXtBlock: Module {
    let dwconv: Conv1d
    let norm: LayerNorm
    let pwconv1: Linear
    let pwconv2: Linear
    let gamma: MLXArray

    init(dim: Int, intermediateDim: Int) {
        dwconv = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: 3, groups: dim)
        norm = LayerNorm(dimensions: dim, eps: 1e-6)
        pwconv1 = Linear(dim, intermediateDim)
        pwconv2 = Linear(intermediateDim, dim)
        gamma = MLXArray.ones([dim])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = dwconv(x)
        h = norm(h)
        h = pwconv1(h)
        h = gelu(h)
        h = pwconv2(h)
        return x + gamma * h
    }
}

final class VocosBackbone: Module {
    let embed: Conv1d
    let norm: LayerNorm
    let convnext: [VocosConvNeXtBlock]
    let final_layer_norm: LayerNorm

    init(inputChannels: Int, dim: Int, intermediateDim: Int, numLayers: Int) {
        embed = Conv1d(inputChannels: inputChannels, outputChannels: dim, kernelSize: 7, padding: 3)
        norm = LayerNorm(dimensions: dim, eps: 1e-6)
        convnext = (0 ..< numLayers).map { _ in VocosConvNeXtBlock(dim: dim, intermediateDim: intermediateDim) }
        final_layer_norm = LayerNorm(dimensions: dim, eps: 1e-6)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = embed(x)
        h = norm(h)
        for block in convnext {
            h = block(h)
        }
        return final_layer_norm(h)
    }
}

final class ISTFTHead: Module {
    let out: Linear
    let _nFFT: Int

    init(dim: Int, nFFT: Int) {
        out = Linear(dim, nFFT + 2)
        _nFFT = nFFT
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = out(x)[0]
        let bins = _nFFT / 2 + 1
        let magnitude = MLX.minimum(MLX.exp(h[0..., 0 ..< bins]), MLXArray(Float(100)))
        let phase = h[0..., bins...]
        let real = magnitude * MLX.cos(phase)
        let imaginary = magnitude * MLX.sin(phase)
        let spectrum = real + imaginary.asImaginary()
        let frames = MLXFFT.irfft(spectrum, n: _nFFT, axis: -1)
        return DSP.overlapAdd(frames)
    }
}

public final class Vocos: Module {
    let backbone: VocosBackbone
    let head: ISTFTHead
    // also used for the F5 reference mel
    let _filterbank: MLXArray

    init(backbone: VocosBackbone, head: ISTFTHead, filterbank: MLXArray) {
        self.backbone = backbone
        self.head = head
        self._filterbank = filterbank
        super.init()
    }

    var filterbank: MLXArray { _filterbank }

    func decode(_ mel: MLXArray) -> MLXArray {
        head(backbone(mel.asType(.float32)))
    }

    static func load(directory: URL) throws -> Vocos {
        var weights = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        guard let filterbank = weights["feature_extractor.filterbank"] else {
            throw F5EngineError.badWeights("Vocos: feature_extractor.filterbank missing")
        }
        weights = weights.filter { !$0.key.hasPrefix("feature_extractor.") }
            .mapValues { $0.asType(.float32) }

        // vocos-mel-24khz defaults
        var inputChannels = 100, dim = 512, intermediateDim = 1536, numLayers = 8, nFFT = 1024
        let configURL = directory.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configURL),
           let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            if let backbone = (config["backbone"] as? [String: Any])?["init_args"] as? [String: Any] {
                inputChannels = backbone["input_channels"] as? Int ?? inputChannels
                dim = backbone["dim"] as? Int ?? dim
                intermediateDim = backbone["intermediate_dim"] as? Int ?? intermediateDim
                numLayers = backbone["num_layers"] as? Int ?? numLayers
            }
            if let head = (config["head"] as? [String: Any])?["init_args"] as? [String: Any] {
                nFFT = head["n_fft"] as? Int ?? nFFT
            }
        }

        let vocos = Vocos(
            backbone: VocosBackbone(
                inputChannels: inputChannels, dim: dim, intermediateDim: intermediateDim, numLayers: numLayers),
            head: ISTFTHead(dim: dim, nFFT: nFFT),
            filterbank: filterbank.asType(.float32)
        )
        _ = try WeightLoading.apply(weights, to: vocos, name: "Vocos")
        return vocos
    }
}
