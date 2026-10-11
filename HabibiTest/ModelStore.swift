import Foundation

// The SAU weights are CC BY-NC-SA, so they're downloaded on request instead of bundled.
struct RemoteFile: Identifiable, Hashable {
    let url: URL
    let localPath: String
    let approxBytes: Int64
    let label: String

    var id: String { localPath }
}

enum ModelStore {
    static let habibiCheckpoint = RemoteFile(
        url: URL(string: "https://huggingface.co/SWivid/Habibi-TTS/resolve/main/Specialized/SAU/model_200000.safetensors")!,
        localPath: "habibi-sau/model_200000.safetensors", approxBytes: 1_350_000_000, label: "Habibi-TTS SAU")
    static let habibiVocab = RemoteFile(
        url: URL(string: "https://huggingface.co/SWivid/Habibi-TTS/resolve/main/Specialized/SAU/vocab.txt")!,
        localPath: "habibi-sau/vocab.txt", approxBytes: 12_000, label: "vocab.txt")
    static let vocosWeights = RemoteFile(
        url: URL(string: "https://huggingface.co/lucasnewman/vocos-mel-24khz-mlx/resolve/main/model.safetensors")!,
        localPath: "vocos/model.safetensors", approxBytes: 54_300_000, label: "Vocos")
    static let vocosConfig = RemoteFile(
        url: URL(string: "https://huggingface.co/lucasnewman/vocos-mel-24khz-mlx/resolve/main/config.json")!,
        localPath: "vocos/config.json", approxBytes: 403, label: "Vocos config")

    // Saudi samples from the Habibi-TTS repo
    static let presetClips: [PresetClip] = [
        PresetClip(
            id: "najdi", name: "نجدي",
            file: RemoteFile(
                url: URL(string: "https://raw.githubusercontent.com/SWivid/Habibi-TTS/main/src/habibi_tts/assets/Najdi.wav")!,
                localPath: "ref/Najdi.wav", approxBytes: 177_450, label: "Najdi.wav"),
            transcript: "تكفى طمني انا اليوم ماني بنايم ولا هو بداخل عيني النوم الين اتطمن عليه."),
        PresetClip(
            id: "hijazi", name: "حجازي",
            file: RemoteFile(
                url: URL(string: "https://raw.githubusercontent.com/SWivid/Habibi-TTS/main/src/habibi_tts/assets/Hijazi.wav")!,
                localPath: "ref/Hijazi.wav", approxBytes: 175_758, label: "Hijazi.wav"),
            transcript: "ابغاك تحقق معاه بس بشكل ودي لانه سلطان يمر بظروف صعبة شوية."),
        PresetClip(
            id: "gulf", name: "خليجي",
            file: RemoteFile(
                url: URL(string: "https://raw.githubusercontent.com/SWivid/Habibi-TTS/main/src/habibi_tts/assets/Gulf.wav")!,
                localPath: "ref/Gulf.wav", approxBytes: 193_678, label: "Gulf.wav"),
            transcript: "وين تو الناس متى تصحى ومتى تفطر وتغير يبيلك ساعة يعني بالله تروح الشغل الساعة عشره."),
    ]

    static var all: [RemoteFile] {
        [habibiVocab, vocosConfig, vocosWeights] + presetClips.map(\.file) + [habibiCheckpoint]
    }

    static var totalBytes: Int64 { all.reduce(0) { $0 + $1.approxBytes } }

    // excluded from iCloud backup
    static var directory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        var dir = base.appendingPathComponent("Models", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try? dir.setResourceValues(values)
        }
        return dir
    }

    static func localURL(_ file: RemoteFile) -> URL {
        directory.appendingPathComponent(file.localPath)
    }

    static func isPresent(_ file: RemoteFile) -> Bool {
        let path = localURL(file).path
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? NSNumber
        else { return false }
        return file.approxBytes > 100_000_000 ? size.int64Value > file.approxBytes * 9 / 10 : size.int64Value > 0
    }

    static var allPresent: Bool { all.allSatisfy(isPresent) }

    static var f16Cache: URL { directory.appendingPathComponent("habibi-sau/converted-f16.safetensors") }

    static func bytesOnDisk() -> Int64 {
        guard let e = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let url as URL in e {
            total += Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    static func freeDiskBytes() -> Int64 {
        let values = try? URL(fileURLWithPath: NSHomeDirectory())
            .resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage ?? 0
    }

    static func deleteAll() {
        try? FileManager.default.removeItem(at: directory)
    }
}

struct PresetClip: Identifiable, Hashable {
    let id: String
    let name: String
    let file: RemoteFile
    let transcript: String
}

final class FileDownloader: NSObject, URLSessionDownloadDelegate {
    private var continuation: CheckedContinuation<Void, Error>?
    private var destination: URL?
    private var moveError: Error?
    private var session: URLSession?
    var onProgress: ((Int64, Int64) -> Void)?

    func download(_ url: URL, to destination: URL) async throws {
        self.destination = destination
        moveError = nil
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
            continuation = c
            let config = URLSessionConfiguration.default
            config.timeoutIntervalForRequest = 60
            config.timeoutIntervalForResource = 6 * 3600
            config.waitsForConnectivity = true
            let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
            self.session = session
            session.downloadTask(with: url).resume()
        }
    }

    func cancel() {
        session?.invalidateAndCancel()
    }

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64
    ) {
        onProgress?(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        if let http = downloadTask.response as? HTTPURLResponse, !(200 ..< 300).contains(http.statusCode) {
            moveError = NSError(
                domain: "Download", code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode) for \(downloadTask.originalRequest?.url?.lastPathComponent ?? "")"])
            return
        }
        guard let destination else { return }
        do {
            let fm = FileManager.default
            try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }
            try fm.moveItem(at: location, to: destination)
        } catch {
            moveError = error
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let result = error ?? moveError
        session.finishTasksAndInvalidate()
        self.session = nil
        let c = continuation
        continuation = nil
        if let result { c?.resume(throwing: result) } else { c?.resume() }
    }
}
