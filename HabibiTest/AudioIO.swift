import AVFoundation
import Foundation

enum AudioIO {
    static let sampleRate = 24_000

    enum AudioError: LocalizedError {
        case noAudioTrack
        case readFailed(String)

        var errorDescription: String? {
            switch self {
            case .noAudioTrack: return "لا يوجد صوت في هذا الملف"
            case .readFailed(let s): return "تعذّرت قراءة الصوت: \(s)"
            }
        }
    }

    // audio track of any audio/video file -> 24 kHz mono float
    static func decodeMono24k(url: URL) async throws -> [Float] {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
            throw AudioError.noAudioTrack
        }
        let reader = try AVAssetReader(asset: asset)
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else {
            throw AudioError.readFailed(reader.error?.localizedDescription ?? "?")
        }
        var samples = [Float]()
        while let buffer = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            var chunk = [Float](repeating: 0, count: length / MemoryLayout<Float>.size)
            chunk.withUnsafeMutableBytes { raw in
                _ = CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: raw.baseAddress!)
            }
            samples.append(contentsOf: chunk)
        }
        if reader.status == .failed {
            throw AudioError.readFailed(reader.error?.localizedDescription ?? "?")
        }
        return samples
    }

    // cut, trim silent edges, add 50 ms of silence (like F5's preprocess_ref_audio_text)
    static func prepareReference(_ samples: [Float], start: Double, length: Double) -> [Float] {
        let a = max(0, min(samples.count, Int(start * Double(sampleRate))))
        let b = max(a, min(samples.count, a + Int(length * Double(sampleRate))))
        var clip = trimSilence(Array(samples[a ..< b]))
        clip.append(contentsOf: [Float](repeating: 0, count: sampleRate / 20))
        return clip
    }

    static func trimSilence(_ x: [Float], thresholdDB: Float = -50, window: Int = 240) -> [Float] {
        guard x.count > window * 2 else { return x }
        let threshold = powf(10, thresholdDB / 20)
        func loud(_ i: Int) -> Bool {
            let end = min(x.count, i + window)
            var sum: Float = 0
            for k in i ..< end { sum += x[k] * x[k] }
            return (sum / Float(end - i)).squareRoot() > threshold
        }
        var first = 0
        while first < x.count - window, !loud(first) { first += window }
        var last = x.count - window
        while last > first, !loud(last) { last -= window }
        return Array(x[first ..< min(x.count, last + window)])
    }

    static func rms(_ x: [Float]) -> Float {
        guard !x.isEmpty else { return 0 }
        var s: Float = 0
        for v in x { s += v * v }
        return (s / Float(x.count)).squareRoot()
    }

    static func wavData(_ samples: [Float], sampleRate: Int = sampleRate) -> Data {
        var data = Data()
        func u32(_ v: UInt32) { var le = v.littleEndian; data.append(Data(bytes: &le, count: 4)) }
        func u16(_ v: UInt16) { var le = v.littleEndian; data.append(Data(bytes: &le, count: 2)) }
        let byteCount = UInt32(samples.count * 2)
        data.append("RIFF".data(using: .ascii)!)
        u32(36 + byteCount)
        data.append("WAVEfmt ".data(using: .ascii)!)
        u32(16); u16(1); u16(1)
        u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2))
        u16(2); u16(16)
        data.append("data".data(using: .ascii)!)
        u32(byteCount)
        var pcm = [Int16](repeating: 0, count: samples.count)
        for i in 0 ..< samples.count {
            pcm[i] = Int16(max(-1, min(1, samples[i])) * 32767)
        }
        pcm.withUnsafeBytes { data.append(contentsOf: $0) }
        return data
    }
}

@MainActor
final class ClipPlayer: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published private(set) var playingID: String?
    private var player: AVAudioPlayer?

    func toggle(id: String, samples: [Float]) {
        if playingID == id {
            stop()
            return
        }
        stop()
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            let p = try AVAudioPlayer(data: AudioIO.wavData(samples))
            p.delegate = self
            p.play()
            player = p
            playingID = id
        } catch {
            playingID = nil
        }
    }

    func stop() {
        player?.stop()
        player = nil
        playingID = nil
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            self.playingID = nil
            self.player = nil
        }
    }
}
