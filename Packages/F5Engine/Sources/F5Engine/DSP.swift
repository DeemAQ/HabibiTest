import Foundation
import MLX

enum DSP {
    static let sampleRate = 24_000
    static let nFFT = 1024
    static let hopLength = 256

    // periodic, like torch.hann_window
    static func hann(_ n: Int) -> MLXArray {
        MLXArray((0 ..< n).map { Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double($0) / Double(n))) })
    }

    // Same as the vocos mel in F5-TTS: torchaudio MelSpectrogram(power=1, center=True), then log.
    // Returns [1, frames, nMels].
    static func logMel(_ audio: MLXArray, filterbank: MLXArray) -> MLXArray {
        let n = audio.shape[0]
        let pad = nFFT / 2
        precondition(n > pad + 1, "audio too short for STFT")

        // reflect padding
        var index = [Int32]()
        index.reserveCapacity(n + 2 * pad)
        for i in stride(from: pad, through: 1, by: -1) { index.append(Int32(i)) }
        for i in 0 ..< n { index.append(Int32(i)) }
        for i in 1 ... pad { index.append(Int32(n - 1 - i)) }
        let padded = audio.take(MLXArray(index), axis: 0)

        let frames = (padded.shape[0] - nFFT) / hopLength + 1
        let starts = MLXArray(0 ..< frames).expandedDimensions(axis: 1) * Int32(hopLength)
        let offsets = MLXArray(0 ..< nFFT).expandedDimensions(axis: 0)
        let framed = padded.take(starts + offsets, axis: 0)

        let spectrum = MLXFFT.rfft(framed * hann(nFFT), axis: -1)
        let magnitude = MLX.abs(spectrum)
        let mel = MLX.matmul(magnitude, filterbank.T)
        let logMel = MLX.log(MLX.maximum(mel, MLXArray(Float(1e-5))))
        return logMel.expandedDimensions(axis: 0)
    }

    // Overlap-add for the vocos ISTFT with "same" padding. `frames` is [T, nFFT] after irfft.
    static func overlapAdd(_ frames: MLXArray) -> MLXArray {
        let t = frames.shape[0]
        let parts = nFFT / hopLength
        let window = hann(nFFT)
        let windowed = frames * window
        let windowSquared = (window * window).expandedDimensions(axis: 0)

        var signal = MLXArray.zeros([t + parts - 1, hopLength], dtype: .float32)
        var envelope = MLXArray.zeros([t + parts - 1, hopLength], dtype: .float32)
        let envelopeFrames = MLX.broadcast(windowSquared, to: [t, nFFT])
        for k in 0 ..< parts {
            let range = (k * hopLength) ..< ((k + 1) * hopLength)
            let widths: [IntOrPair] = [IntOrPair((k, parts - 1 - k)), IntOrPair((0, 0))]
            signal = signal + MLX.padded(windowed[0..., range], widths: widths)
            envelope = envelope + MLX.padded(envelopeFrames[0..., range], widths: widths)
        }
        let y = signal.reshaped([-1])
        let env = envelope.reshaped([-1])
        let normalised = MLX.where(env .> MLXArray(Float(1e-11)), y / env, y)

        let trim = (nFFT - hopLength) / 2
        let length = normalised.shape[0]
        return normalised[trim ..< (length - trim)]
    }
}
