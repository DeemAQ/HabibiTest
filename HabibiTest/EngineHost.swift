import F5Engine
import Foundation

// All MLX work runs on one serial queue.
final class EngineHost: @unchecked Sendable {
    private let queue = DispatchQueue(label: "habibi.engine", qos: .userInitiated)
    private var tts: F5TTS?

    func load(precision: Precision) async throws -> LoadReport {
        try await run {
            self.tts = nil
            EngineMemory.clearCache()
            let (tts, report) = try F5TTS.load(
                checkpoint: ModelStore.localURL(ModelStore.habibiCheckpoint),
                vocab: ModelStore.localURL(ModelStore.habibiVocab),
                vocosDirectory: ModelStore.localURL(ModelStore.vocosWeights).deletingLastPathComponent(),
                precision: precision,
                // only the f16 conversion is cached
                convertedCache: precision == .float16 ? ModelStore.f16Cache : nil
            )
            self.tts = tts
            EngineMemory.clearCache()
            return report
        }
    }

    func unload() async {
        _ = try? await run {
            self.tts = nil
            EngineMemory.clearCache()
        }
    }

    func generate(
        text: String, reference: ReferenceVoice, options: GenerationOptions,
        progress: @escaping @Sendable (Int, Int) -> Void
    ) async throws -> GenerationResult {
        try await run {
            guard let tts = self.tts else {
                throw NSError(domain: "Engine", code: 1, userInfo: [NSLocalizedDescriptionKey: "النموذج غير جاهز"])
            }
            return try tts.generate(text: text, reference: reference, options: options, progress: progress)
        }
    }

    private func run<T>(_ work: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                continuation.resume(with: Result { try work() })
            }
        }
    }
}
