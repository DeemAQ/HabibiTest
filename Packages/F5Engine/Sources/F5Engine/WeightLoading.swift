import Foundation
import MLX
import MLXNN

public enum F5EngineError: Error, CustomStringConvertible {
    case badWeights(String)
    case badVocab(String)
    case badInput(String)

    public var description: String {
        switch self {
        case .badWeights(let s): return "Weights: \(s)"
        case .badVocab(let s): return "Vocab: \(s)"
        case .badInput(let s): return "Input: \(s)"
        }
    }
}

enum WeightLoading {
    // Returns the keys that were in the file but not used
    @discardableResult
    static func apply(_ weights: [String: MLXArray], to module: Module, name: String) throws -> [String] {
        let expected = Set(module.parameters().flattened().map { $0.0 })
        let provided = Set(weights.keys)
        let missing = expected.subtracting(provided).sorted()
        guard missing.isEmpty else {
            let sample = provided.sorted().prefix(8).joined(separator: ", ")
            throw F5EngineError.badWeights(
                "\(name): \(missing.count) missing, e.g. \(missing.prefix(6).joined(separator: ", ")). File keys look like: \(sample)")
        }
        let used = weights.filter { expected.contains($0.key) }
        do {
            try module.update(parameters: ModuleParameters.unflattened(used), verify: [.all])
        } catch {
            throw F5EngineError.badWeights("\(name): \(error)")
        }
        return provided.subtracting(expected).sorted()
    }

    // PyTorch F5-TTS checkpoint -> DiT parameters (keys, conv layout, dtype)
    static func convertTorchDiT(_ raw: [String: MLXArray], dtype: DType) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        let hasTransformerPrefix = raw.keys.contains { $0.contains("transformer.") }
        for (originalKey, value) in raw {
            var key = originalKey
            if key.hasPrefix("ema_model.") { key.removeFirst("ema_model.".count) }
            if key.hasPrefix("transformer.") {
                key.removeFirst("transformer.".count)
            } else if hasTransformerPrefix {
                continue
            }

            key = key
                .replacingOccurrences(of: "time_embed.time_mlp.2.", with: "time_embed.time_mlp.1.")
                .replacingOccurrences(of: "conv_pos_embed.conv1d.2.", with: "conv_pos_embed.conv1d.1.")
                .replacingOccurrences(of: ".ff.ff.0.0.", with: ".ff.ff.0.")
                .replacingOccurrences(of: ".ff.ff.2.", with: ".ff.ff.1.")

            var array = value
            if array.ndim == 3 && key.hasSuffix(".weight") {
                array = array.transposed(0, 2, 1)
            }
            if array.dtype.isFloatingPoint {
                array = array.asType(dtype)
            }
            out[key] = array
        }
        return out
    }
}
