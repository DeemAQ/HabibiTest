import Foundation
import MLX
import MLXNN

public enum Precision: String, CaseIterable, Sendable, Codable {
    case float16
    case float32

    var dtype: DType { self == .float16 ? .float16 : .float32 }
}

/// 24 kHz mono samples and the exact transcript of what is said in them.
public struct ReferenceVoice: Sendable {
    public var samples: [Float]
    public var text: String

    public init(samples: [Float], text: String) {
        self.samples = samples
        self.text = text
    }

    public var seconds: Double { Double(samples.count) / Double(F5TTS.sampleRate) }
}

public struct GenerationOptions: Sendable, Codable {
    // Habibi uses 32
    public var steps = 32
    public var cfgStrength = 2.0
    public var swayCoefficient: Double? = -1.0
    public var speed = 1.0
    public var seed: UInt64? = nil
    public var targetRMS: Float = 0.1

    public init() {}
}

public struct GenerationResult: Sendable {
    public var samples: [Float]
    public var audioSeconds: Double
    public var prepareSeconds: Double
    public var flowSeconds: Double
    public var vocoderSeconds: Double
    public var totalSeconds: Double
    // reference + generated mel frames
    public var totalFrames: Int
    public var generatedFrames: Int
    public var textTokens: Int
    public var unknownCharacters: [String]
}

public struct LoadReport: Sendable {
    public var seconds: Double
    public var convertedFromCheckpoint: Bool
    public var unusedCheckpointKeys: [String]
    public var vocabSize: Int
    public var embeddingRows: Int
    public var notes: [String]
}

/// F5-TTS v1 inference, following Habibi-TTS `infer_process`, with Vocos as the vocoder.
public final class F5TTS {
    public static let sampleRate = DSP.sampleRate
    static let hop = DSP.hopLength

    let dit: DiT
    let vocos: Vocos
    let vocab: [String: Int]
    public let precision: Precision

    init(dit: DiT, vocos: Vocos, vocab: [String: Int], precision: Precision) {
        self.dit = dit
        self.vocos = vocos
        self.vocab = vocab
        self.precision = precision
    }

    // MARK: Loading

    /// `checkpoint` is the original PyTorch safetensors file (e.g. Habibi
    /// `Specialized/SAU/model_200000.safetensors`). If `convertedCache` is set, the converted
    /// weights are saved there and reused next time.
    public static func load(
        checkpoint: URL, vocab vocabURL: URL, vocosDirectory: URL, precision: Precision,
        convertedCache: URL?, config: DiTConfig = DiTConfig()
    ) throws -> (F5TTS, LoadReport) {
        let start = Date()
        var notes = [String]()
        let vocab = try parseVocab(Data(contentsOf: vocabURL))
        if vocab[" "] != 0 { notes.append("vocab: space is not id 0") }

        var weights: [String: MLXArray]
        var converted = false
        var cached: [String: MLXArray]? = nil
        if let cache = convertedCache, FileManager.default.fileExists(atPath: cache.path),
           let loaded = try? loadArraysAndMetadata(url: cache),
           loaded.1["format"] == cacheFormat, loaded.1["dtype"] == precision.rawValue
        {
            cached = loaded.0
        }
        if let cached {
            weights = cached
        } else {
            let raw = try loadArrays(url: checkpoint)
            weights = WeightLoading.convertTorchDiT(raw, dtype: precision.dtype)
            converted = true
            if let cache = convertedCache {
                MLX.eval(Array(weights.values))
                try? FileManager.default.removeItem(at: cache)
                do {
                    try MLX.save(
                        arrays: weights, metadata: ["format": cacheFormat, "dtype": precision.rawValue], url: cache)
                } catch {
                    notes.append("could not save converted weights: \(error)")
                }
            }
        }

        var c = config
        guard let embedding = weights["text_embed.text_embed.weight"] else {
            throw F5EngineError.badWeights(
                "text_embed.text_embed.weight not found. Keys look like: \(weights.keys.sorted().prefix(8).joined(separator: ", "))")
        }
        let embeddingRows = embedding.shape[0]
        c.textNumEmbeds = embeddingRows - 1
        c.textDim = embedding.shape[1]
        if c.textNumEmbeds != vocab.count {
            notes.append("vocab.txt has \(vocab.count) entries, checkpoint embedding \(c.textNumEmbeds)")
        }
        let blockIndices = Set(weights.keys.compactMap { key -> Int? in
            guard key.hasPrefix("transformer_blocks.") else { return nil }
            return Int(key.split(separator: ".")[1])
        })
        if !blockIndices.isEmpty { c.depth = blockIndices.count }
        if let projOut = weights["proj_out.weight"] { c.dim = projOut.shape[1]; c.melDim = projOut.shape[0] }

        let dit = DiT(c)
        let unused = try WeightLoading.apply(weights, to: dit, name: "DiT")
        weights.removeAll()
        MLX.eval(dit.parameters())

        let vocos = try Vocos.load(directory: vocosDirectory)
        MLX.eval(vocos.parameters())

        let report = LoadReport(
            seconds: Date().timeIntervalSince(start), convertedFromCheckpoint: converted,
            unusedCheckpointKeys: unused, vocabSize: vocab.count, embeddingRows: embeddingRows, notes: notes)
        return (F5TTS(dit: dit, vocos: vocos, vocab: vocab, precision: precision), report)
    }

    static let cacheFormat = "f5engine-dit-v1"

    // Same as get_tokenizer(..., "custom"): line i has id i
    static func parseVocab(_ data: Data) throws -> [String: Int] {
        guard let text = String(data: data, encoding: .utf8) else {
            throw F5EngineError.badVocab("not UTF-8")
        }
        var map = [String: Int]()
        var line = String.UnicodeScalarView()
        var index = 0
        for scalar in text.unicodeScalars {
            if scalar == "\n" {
                map[String(line)] = index
                index += 1
                line = String.UnicodeScalarView()
            } else {
                line.append(scalar)
            }
        }
        guard !map.isEmpty else { throw F5EngineError.badVocab("empty") }
        return map
    }

    // MARK: Text

    // Python iterates a str by code point, so tokenize by unicode scalar, not Character
    // (a letter with harakat is one Character but several tokens). Unknown chars map to 0.
    func tokenize(_ text: String) -> (ids: [Int32], unknown: [String]) {
        var ids = [Int32]()
        var unknown = Set<String>()
        for scalar in text.unicodeScalars {
            let s = String(Character(scalar))
            if let id = vocab[s] {
                ids.append(Int32(id))
            } else {
                ids.append(0)
                unknown.insert(s)
            }
        }
        return (ids, unknown.sorted())
    }

    // Same as preprocess_ref_audio_text + infer_batch_process in F5-TTS
    static func preparedReferenceText(_ text: String) -> String {
        var t = text
        if !t.hasSuffix(". ") && !t.hasSuffix("。") {
            t += t.hasSuffix(".") ? " " : ". "
        }
        if let last = t.unicodeScalars.last, last.utf8.count == 1 {
            t += " "
        }
        return t
    }

    // EPSS timesteps where F5-TTS has a table for the step count, uniform otherwise, then sway
    static func timeGrid(steps: Int, sway: Double?) -> [Double] {
        let epss: [Int: [Int]] = [
            5: [0, 2, 4, 8, 16, 32],
            6: [0, 2, 4, 6, 8, 16, 32],
            7: [0, 2, 4, 6, 8, 16, 24, 32],
            10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
            12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
            16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        ]
        var t: [Double]
        if let table = epss[steps] {
            t = table.map { Double($0) / 32 }
        } else {
            t = (0 ... steps).map { Double($0) / Double(steps) }
        }
        if let s = sway {
            t = t.map { $0 + s * (cos(Double.pi / 2 * $0) - 1 + $0) }
        }
        return t
    }

    // MARK: Generation

    /// Generates one sentence. The specialised SAU model takes plain text (no dialect tag).
    public func generate(
        text: String, reference: ReferenceVoice, options: GenerationOptions = GenerationOptions(),
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> GenerationResult {
        let t0 = Date()
        let genText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !genText.isEmpty else { throw F5EngineError.badInput("empty text") }
        guard reference.samples.count > Self.hop * 8 else { throw F5EngineError.badInput("reference too short") }
        let refText = Self.preparedReferenceText(reference.text.trimmingCharacters(in: .whitespacesAndNewlines))

        // quiet references are normalised, the output is scaled back at the end
        var audio = MLXArray(reference.samples)
        let rms = (audio * audio).mean().sqrt().item(Float.self)
        if rms < options.targetRMS && rms > 0 {
            audio = audio * (options.targetRMS / rms)
        }

        // duration estimate from the reference speaking rate (utf-8 bytes, as in F5-TTS)
        let refAudioLen = reference.samples.count / Self.hop
        let refBytes = max(refText.utf8.count, 1)
        let genBytes = genText.utf8.count
        let localSpeed = genBytes < 10 ? 0.3 : options.speed
        var duration = refAudioLen + Int(Double(refAudioLen) / Double(refBytes) * Double(genBytes) / localSpeed)

        let cond = DSP.logMel(audio, filterbank: vocos.filterbank)  // [1, condLen, 100]
        let condLen = cond.shape[1]
        let (ids, unknown) = tokenize(refText + genText)
        duration = max(max(ids.count, condLen) + 1, duration)
        duration = min(duration, 4096)  // ~43 s

        let textIDs = MLXArray(ids).reshaped([1, ids.count])
        let condPadded = MLX.padded(
            cond, widths: [IntOrPair((0, 0)), IntOrPair((0, duration - condLen)), IntOrPair((0, 0))])

        let useCFG = options.cfgStrength > 1e-5
        let textEmbed: MLXArray
        let condInput: MLXArray
        if useCFG {
            // cond + uncond in one batch (cfg_infer)
            textEmbed = MLX.concatenated(
                [dit.embedText(textIDs, seqLen: duration, dropText: false),
                 dit.embedText(textIDs, seqLen: duration, dropText: true)], axis: 0)
            condInput = MLX.concatenated([condPadded, MLX.zeros(like: condPadded)], axis: 0).asType(dit.dtype)
        } else {
            textEmbed = dit.embedText(textIDs, seqLen: duration, dropText: false)
            condInput = condPadded.asType(dit.dtype)
        }
        MLX.eval(textEmbed, condInput)
        let t1 = Date()

        // Euler ODE from noise to mel
        if let seed = options.seed { MLXRandom.seed(seed) }
        var y = MLXRandom.normal([1, duration, 100], dtype: .float32)
        let grid = Self.timeGrid(steps: options.steps, sway: options.swayCoefficient)
        let batch = useCFG ? 2 : 1
        let cfg = Float(options.cfgStrength)
        for i in 0 ..< (grid.count - 1) {
            let t = Float(grid[i])
            let dt = Float(grid[i + 1] - grid[i])
            let xIn = useCFG ? MLX.concatenated([y, y], axis: 0) : y
            let out = dit(x: xIn, cond: condInput, textEmbed: textEmbed, time: MLXArray([Float](repeating: t, count: batch)))
            let flow: MLXArray
            if useCFG {
                let pred = out[0 ..< 1]
                let nullPred = out[1 ..< 2]
                flow = pred + (pred - nullPred) * cfg
            } else {
                flow = out
            }
            y = y + dt * flow
            MLX.eval(y)
            progress?(i + 1, grid.count - 1)
        }
        let t2 = Date()

        let isCond = (MLXArray(0 ..< duration) .< MLXArray(Int32(condLen))).reshaped([1, duration, 1])
        let mel = MLX.where(isCond, condPadded, y)
        let generated = mel[0..., refAudioLen..., 0...]
        var wave = vocos.decode(generated)
        if rms < options.targetRMS && rms > 0 {
            wave = wave * (rms / options.targetRMS)
        }
        MLX.eval(wave)
        let samples = wave.asArray(Float.self)
        let t3 = Date()

        return GenerationResult(
            samples: samples,
            audioSeconds: Double(samples.count) / Double(Self.sampleRate),
            prepareSeconds: t1.timeIntervalSince(t0),
            flowSeconds: t2.timeIntervalSince(t1),
            vocoderSeconds: t3.timeIntervalSince(t2),
            totalSeconds: t3.timeIntervalSince(t0),
            totalFrames: duration,
            generatedFrames: duration - refAudioLen,
            textTokens: ids.count,
            unknownCharacters: unknown
        )
    }
}

// MARK: - Memory

public enum EngineMemory {
    public struct Snapshot: Sendable {
        public var activeMB: Double
        public var peakMB: Double
        public var cacheMB: Double
    }

    public static func snapshot() -> Snapshot {
        Snapshot(
            activeMB: Double(GPU.activeMemory) / 1_048_576,
            peakMB: Double(GPU.peakMemory) / 1_048_576,
            cacheMB: Double(GPU.cacheMemory) / 1_048_576)
    }

    public static func resetPeak() { GPU.resetPeakMemory() }
    public static func clearCache() { GPU.clearCache() }
    public static func setCacheLimit(megabytes: Int) { GPU.set(cacheLimit: megabytes * 1_048_576) }
}
