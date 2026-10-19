import F5Engine
import Foundation
import UIKit

struct LoadInfo: Codable {
    var precision: String
    var seconds: Double
    var convertedFromCheckpoint: Bool
    var footprintMB: Double
    var gpuActiveMB: Double
    var availableMB: Double
    var notes: [String]
    var unusedCheckpointKeys: Int
}

struct SentenceResult: Identifiable, Codable {
    var id: Int
    var text: String
    var audioSeconds: Double
    var totalSeconds: Double
    var prepareSeconds: Double
    var flowSeconds: Double
    var vocoderSeconds: Double
    // generation time / audio length
    var realTimeFactor: Double
    var totalFrames: Int
    var generatedFrames: Int
    var textTokens: Int
    var unknownCharacters: [String]
    var peakFootprintMB: Double
    var footprintAfterMB: Double
    var gpuPeakMB: Double
    var gpuActiveMB: Double
    var minAvailableMB: Double
    var thermal: String
    var wavFile: String
}

struct RunSummary: Codable {
    var date: String
    var device: String
    var iOS: String
    var ramGB: Double
    var precision: String
    var steps: Int
    var cfgStrength: Double
    var sway: Double?
    var seed: UInt64?
    var reference: String
    var referenceSeconds: Double
    var referenceTranscript: String
    var warmUpSeconds: Double?
    var load: LoadInfo?
    var thermalStart: String
    var thermalEnd: String
    var meanTotalSeconds: Double
    var meanRealTimeFactor: Double
    var results: [SentenceResult]
}

@MainActor
final class AppModel: ObservableObject {
    static let defaultSentences = [
        "نادوا الممرضة لو سمحتوا",
        "عطوني موية",
        "ألم في الرجل اليسرى، لا يُحتمل",
        "الحمد لله، أنا بخير اليوم",
        "ودي أطلع أقعد برا شوي، الجو حلو والشمس طالعة",
    ]

    enum ReferenceSource: Hashable {
        case preset(String)
        case custom
    }

    let engine = EngineHost()
    let player = ClipPlayer()
    private let sampler = PeakSampler()
    private var downloader: FileDownloader?

    @Published var filesReady = ModelStore.allPresent
    @Published var downloading = false
    @Published var downloadLabel = ""
    @Published var downloadFraction: Double = 0
    @Published var downloadError: String?
    @Published var diskUsedBytes: Int64 = ModelStore.bytesOnDisk()

    @Published var precision: Precision = .float16
    @Published var loadedPrecision: Precision?
    @Published var loading = false
    @Published var loadInfo: LoadInfo?
    @Published var loadError: String?

    @Published var referenceSource: ReferenceSource = .preset("najdi")
    @Published var sourceName = ""
    @Published var sourceSamples: [Float] = []
    @Published var clipStart: Double = 0
    @Published var clipLength: Double = 8
    // F5-TTS clips the reference to 12 s anyway
    static let maxClipSeconds: Double = 12
    var clipEnd: Double { clipStart + clipLength }

    func setClipEnd(_ end: Double) {
        clipLength = max(1, min(end - clipStart, Self.maxClipSeconds, sourceSeconds - clipStart))
    }
    @Published var transcript = ""
    @Published var referenceSamples: [Float] = []
    @Published var referenceError: String?

    @Published var sentences = AppModel.defaultSentences
    @Published var steps = 32
    @Published var seedText = "42"
    @Published var warmUp = true

    @Published var running = false
    @Published var status = ""
    @Published var stepFraction: Double = 0
    @Published var results: [SentenceResult] = []
    @Published var summary: RunSummary?
    @Published var runFolder: URL?
    @Published var runError: String?
    @Published var log: [String] = []
    private(set) var resultSamples: [Int: [Float]] = [:]

    @Published var footprintMB = Metrics.footprintMB()
    @Published var availableMB = Metrics.availableMB()

    init() {
        EngineMemory.setCacheLimit(megabytes: 512)
    }

    var sourceSeconds: Double { Double(sourceSamples.count) / Double(AudioIO.sampleRate) }
    var referenceSeconds: Double { Double(referenceSamples.count) / Double(AudioIO.sampleRate) }
    var canRun: Bool {
        loadedPrecision != nil && !referenceSamples.isEmpty
            && !transcript.trimmingCharacters(in: .whitespaces).isEmpty && !running && !loading
    }

    func refreshMemory() {
        footprintMB = Metrics.footprintMB()
        availableMB = Metrics.availableMB()
    }

    func note(_ s: String) {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "HH:mm:ss"
        let time = f.string(from: Date())
        log.append("[\(time)] \(s)")
    }


    func downloadAll() async {
        downloadError = nil
        let missing = ModelStore.all.filter { !ModelStore.isPresent($0) }
        let needed = missing.reduce(Int64(0)) { $0 + $1.approxBytes } + 700_000_000  // + converted f16 copy
        let free = ModelStore.freeDiskBytes()
        if free > 0 && free < needed {
            downloadError = "المساحة غير كافية: يلزم \(Fmt.bytes(needed)) والمتاح \(Fmt.bytes(free))"
            return
        }

        downloading = true
        UIApplication.shared.isIdleTimerDisabled = true
        defer {
            downloading = false
            UIApplication.shared.isIdleTimerDisabled = running
            filesReady = ModelStore.allPresent
            diskUsedBytes = ModelStore.bytesOnDisk()
        }

        let total = Double(ModelStore.totalBytes)
        var done: Int64 = 0
        let downloader = FileDownloader()
        self.downloader = downloader
        for file in ModelStore.all {
            if ModelStore.isPresent(file) {
                done += file.approxBytes
                continue
            }
            downloadLabel = file.label
            let base = done
            downloader.onProgress = { [weak self] written, _ in
                Task { @MainActor in self?.downloadFraction = min(1, Double(base + written) / total) }
            }
            do {
                note("تنزيل \(file.label)…")
                try await downloader.download(file.url, to: ModelStore.localURL(file))
                note("تم \(file.label)")
            } catch {
                downloadError = "تعذّر تنزيل \(file.label): \(error.localizedDescription)"
                note(downloadError ?? "")
                return
            }
            done += file.approxBytes
            downloadFraction = min(1, Double(done) / total)
        }
        if referenceSamples.isEmpty, case .preset(let id) = referenceSource {
            await selectPreset(id)
        }
    }

    func cancelDownload() {
        downloader?.cancel()
    }

    func deleteFiles() async {
        await unloadModel()
        ModelStore.deleteAll()
        filesReady = false
        diskUsedBytes = ModelStore.bytesOnDisk()
        note("حُذفت ملفات النموذج")
    }

    
    func loadModel() async {
        guard filesReady, !loading else { return }
        loading = true
        loadError = nil
        status = precision == .float16 && !FileManager.default.fileExists(atPath: ModelStore.f16Cache.path)
            ? "تحويل النموذج أول مرة… (دقيقة تقريباً)" : "تجهيز النموذج…"
        UIApplication.shared.isIdleTimerDisabled = true
        defer {
            loading = false
            status = ""
            UIApplication.shared.isIdleTimerDisabled = false
            refreshMemory()
            diskUsedBytes = ModelStore.bytesOnDisk()
        }
        loadedPrecision = nil
        do {
            let report = try await engine.load(precision: precision)
            loadedPrecision = precision
            let gpu = EngineMemory.snapshot()
            loadInfo = LoadInfo(
                precision: precision.rawValue, seconds: report.seconds,
                convertedFromCheckpoint: report.convertedFromCheckpoint,
                footprintMB: Metrics.footprintMB(), gpuActiveMB: gpu.activeMB, availableMB: Metrics.availableMB(),
                notes: report.notes, unusedCheckpointKeys: report.unusedCheckpointKeys.count)
            note("جُهّز النموذج (\(precision.rawValue)) في \(Fmt.s(report.seconds))"
                + (report.convertedFromCheckpoint ? " (مع التحويل)" : ""))
            if !report.unusedCheckpointKeys.isEmpty {
                note("مفاتيح غير مستخدمة: \(report.unusedCheckpointKeys.prefix(10).joined(separator: ", "))")
            }
            for n in report.notes { note(n) }
        } catch {
            loadError = "\(error)"
            note("فشل التجهيز: \(error)")
        }
    }

    func unloadModel() async {
        await engine.unload()
        loadedPrecision = nil
        loadInfo = nil
        refreshMemory()
    }


    func selectPreset(_ id: String) async {
        guard let preset = ModelStore.presetClips.first(where: { $0.id == id }) else { return }
        referenceSource = .preset(id)
        referenceError = nil
        let url = ModelStore.localURL(preset.file)
        guard FileManager.default.fileExists(atPath: url.path) else {
            sourceSamples = []
            referenceSamples = []
            transcript = preset.transcript
            sourceName = preset.name
            return
        }
        do {
            sourceSamples = try await AudioIO.decodeMono24k(url: url)
            sourceName = preset.name
            clipStart = 0
            clipLength = min(12, sourceSeconds)
            transcript = preset.transcript
            prepareReference()
        } catch {
            referenceError = error.localizedDescription
        }
    }

    func importClip(from url: URL) async {
        referenceError = nil
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            let copy = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + "." + url.pathExtension)
            try FileManager.default.copyItem(at: url, to: copy)
            try await useCustomSource(copy, name: url.lastPathComponent)
        } catch {
            referenceError = error.localizedDescription
        }
    }

    func useCustomSource(_ url: URL, name: String) async throws {
        let samples = try await AudioIO.decodeMono24k(url: url)
        guard samples.count > AudioIO.sampleRate else {
            referenceError = "المقطع قصير جداً"
            return
        }
        referenceSource = .custom
        sourceSamples = samples
        sourceName = name
        clipStart = 0
        clipLength = min(10, sourceSeconds)
        transcript = ""
        prepareReference()
    }

    func prepareReference() {
        guard !sourceSamples.isEmpty else { return }
        clipStart = max(0, min(clipStart, max(0, sourceSeconds - 1)))
        clipLength = max(1, min(clipLength, Self.maxClipSeconds, sourceSeconds - clipStart))
        referenceSamples = AudioIO.prepareReference(sourceSamples, start: clipStart, length: clipLength)
    }


    func runAll() async {
        guard canRun, let precision = loadedPrecision else { return }
        running = true
        runError = nil
        results = []
        resultSamples = [:]
        summary = nil
        runFolder = nil
        player.stop()
        UIApplication.shared.isIdleTimerDisabled = true
        defer {
            running = false
            status = ""
            stepFraction = 0
            UIApplication.shared.isIdleTimerDisabled = false
            refreshMemory()
        }

        let reference = ReferenceVoice(samples: referenceSamples, text: transcript)
        var options = GenerationOptions()
        options.steps = steps
        options.seed = UInt64(seedText.trimmingCharacters(in: .whitespaces))
        let thermalStart = Metrics.thermal
        let progress: @Sendable (Int, Int) -> Void = { [weak self] done, total in
            Task { @MainActor in self?.stepFraction = Double(done) / Double(total) }
        }

        var warmUpSeconds: Double?
        do {
            if warmUp {
                status = "تسخين (لا يحسب)…"
                let r = try await engine.generate(
                    text: "مرحبا، كيف الحال؟", reference: reference, options: options, progress: progress)
                warmUpSeconds = r.totalSeconds
                note("تسخين: \(Fmt.s(r.totalSeconds))")
            }

            for (index, sentence) in sentences.enumerated() {
                let text = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else { continue }
                status = "الجملة \(index + 1) من \(sentences.count)"
                stepFraction = 0
                EngineMemory.resetPeak()
                sampler.start()
                let r = try await engine.generate(text: text, reference: reference, options: options, progress: progress)
                sampler.stop()
                let gpu = EngineMemory.snapshot()
                let result = SentenceResult(
                    id: index + 1, text: text, audioSeconds: r.audioSeconds, totalSeconds: r.totalSeconds,
                    prepareSeconds: r.prepareSeconds, flowSeconds: r.flowSeconds, vocoderSeconds: r.vocoderSeconds,
                    realTimeFactor: r.audioSeconds > 0 ? r.totalSeconds / r.audioSeconds : 0,
                    totalFrames: r.totalFrames, generatedFrames: r.generatedFrames, textTokens: r.textTokens,
                    unknownCharacters: r.unknownCharacters,
                    peakFootprintMB: sampler.peakFootprintMB, footprintAfterMB: Metrics.footprintMB(),
                    gpuPeakMB: gpu.peakMB, gpuActiveMB: gpu.activeMB, minAvailableMB: sampler.minAvailableMB,
                    thermal: Metrics.thermal, wavFile: "s\(index + 1).wav")
                results.append(result)
                resultSamples[result.id] = r.samples
                refreshMemory()
                if !r.unknownCharacters.isEmpty {
                    note("الجملة \(index + 1): حروف غير معروفة للنموذج: \(r.unknownCharacters.joined(separator: " "))")
                }
            }
        } catch {
            sampler.stop()
            runError = "\(error)"
            note("خطأ: \(error)")
        }

        guard !results.isEmpty else { return }
        let mean = results.map(\.totalSeconds).reduce(0, +) / Double(results.count)
        let meanRTF = results.map(\.realTimeFactor).reduce(0, +) / Double(results.count)
        summary = RunSummary(
            date: ISO8601DateFormatter().string(from: Date()), device: Metrics.deviceModel,
            iOS: Metrics.systemVersion, ramGB: Metrics.physicalMemoryGB, precision: precision.rawValue,
            steps: options.steps, cfgStrength: options.cfgStrength, sway: options.swayCoefficient, seed: options.seed,
            reference: sourceName, referenceSeconds: referenceSeconds, referenceTranscript: transcript,
            warmUpSeconds: warmUpSeconds, load: loadInfo, thermalStart: thermalStart, thermalEnd: Metrics.thermal,
            meanTotalSeconds: mean, meanRealTimeFactor: meanRTF, results: results)
        saveRun()
    }

    // Documents/Runs/<date>/ (visible in the Files app)
    private func saveRun() {
        guard let summary else { return }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let stamp = DateFormatter()
        stamp.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        stamp.locale = Locale(identifier: "en_US_POSIX")
        let name = "\(stamp.string(from: Date()))_\(summary.precision)_\(summary.steps)steps"
        let folder = docs.appendingPathComponent("Runs", isDirectory: true).appendingPathComponent(name, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try encoder.encode(summary).write(to: folder.appendingPathComponent("results.json"))
            try AudioIO.wavData(referenceSamples).write(to: folder.appendingPathComponent("reference.wav"))
            for r in results {
                if let s = resultSamples[r.id] {
                    try AudioIO.wavData(s).write(to: folder.appendingPathComponent(r.wavFile))
                }
            }
            runFolder = folder
            note("تم حفظ النتائج في: Files ▸ On My iPad ▸ HabibiTest ▸ Runs ▸ \(name)")
        } catch {
            note("تعذر حفظ النتائج: \(error.localizedDescription)")
        }
    }

    var exportFiles: [URL] {
        guard let runFolder,
              let items = try? FileManager.default.contentsOfDirectory(at: runFolder, includingPropertiesForKeys: nil)
        else { return [] }
        return items.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    func playResult(_ id: Int) {
        guard let s = resultSamples[id] else { return }
        player.toggle(id: "s\(id)", samples: s)
    }
}

enum Fmt {
    static func s(_ v: Double) -> String { String(format: "%.2f s", v) }
    static func clock(_ v: Double) -> String {
        let t = max(0, v)
        return String(format: "%d:%04.1f", Int(t) / 60, t.truncatingRemainder(dividingBy: 60))
    }
    static func mb(_ v: Double) -> String { String(format: "%.0f MB", v) }
    static func x(_ v: Double) -> String { String(format: "%.2f×", v) }
    static func bytes(_ b: Int64) -> String {
        let gb = Double(b) / 1_000_000_000
        return gb >= 1 ? String(format: "%.2f GB", gb) : String(format: "%.0f MB", Double(b) / 1_000_000)
    }
}
