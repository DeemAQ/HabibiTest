import Foundation
import MLX
import MLXNN

// DiT backbone, F5TTS_v1_Base config. See SWivid/F5-TTS model/backbones/dit.py and modules.py.
// Property names match the PyTorch modules so the checkpoint keys load directly.

public struct DiTConfig: Sendable {
    public var dim = 1024
    public var depth = 22
    public var heads = 16
    public var dimHead = 64
    public var ffMult = 2
    public var melDim = 100
    public var textNumEmbeds = 256
    public var textDim = 512
    public var convLayers = 4

    public init() {}
}

// MARK: - Embeddings

final class SinusPositionEmbedding: Module {
    let _dim: Int

    init(dim: Int) {
        _dim = dim
        super.init()
    }

    func callAsFunction(_ x: MLXArray, scale: Float = 1000) -> MLXArray {
        let half = _dim / 2
        let factor = Foundation.log(10000.0) / Double(half - 1)
        let freqs = MLXArray((0 ..< half).map { Float(Foundation.exp(-Double($0) * factor)) })
        let args = scale * x.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        return MLX.concatenated([MLX.sin(args), MLX.cos(args)], axis: -1)
    }
}

final class TimestepEmbedding: Module {
    let time_embed: SinusPositionEmbedding
    let time_mlp: [Linear]

    init(dim: Int, freqEmbedDim: Int = 256) {
        time_embed = SinusPositionEmbedding(dim: freqEmbedDim)
        time_mlp = [Linear(freqEmbedDim, dim), Linear(dim, dim)]
        super.init()
    }

    func callAsFunction(_ timestep: MLXArray, dtype: DType) -> MLXArray {
        var h = time_embed(timestep.asType(.float32)).asType(dtype)
        h = time_mlp[0](h)
        h = silu(h)
        return time_mlp[1](h)
    }
}

// computed in float32, the sum over the sequence can overflow float16
final class GRN: Module {
    let gamma: MLXArray
    let beta: MLXArray

    init(dim: Int) {
        gamma = MLXArray.zeros([1, 1, dim])
        beta = MLXArray.zeros([1, 1, dim])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let gx = xf.square().sum(axis: 1, keepDims: true).sqrt()
        let nx = gx / (gx.mean(axis: -1, keepDims: true) + Float(1e-6))
        let y = gamma.asType(.float32) * (xf * nx) + beta.asType(.float32) + xf
        return y.asType(x.dtype)
    }
}

final class ConvNeXtV2Block: Module {
    let dwconv: Conv1d
    let norm: LayerNorm
    let pwconv1: Linear
    let grn: GRN
    let pwconv2: Linear

    init(dim: Int, intermediateDim: Int) {
        dwconv = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: 3, groups: dim)
        norm = LayerNorm(dimensions: dim, eps: 1e-6)
        pwconv1 = Linear(dim, intermediateDim)
        grn = GRN(dim: intermediateDim)
        pwconv2 = Linear(intermediateDim, dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = dwconv(x)
        h = norm(h)
        h = pwconv1(h)
        h = gelu(h)
        h = grn(h)
        h = pwconv2(h)
        return x + h
    }
}

final class TextEmbedding: Module {
    let text_embed: Embedding
    let text_blocks: [ConvNeXtV2Block]
    let _textDim: Int

    init(textNumEmbeds: Int, textDim: Int, convLayers: Int, convMult: Int = 2) {
        text_embed = Embedding(embeddingCount: textNumEmbeds + 1, dimensions: textDim)
        text_blocks = (0 ..< convLayers).map { _ in
            ConvNeXtV2Block(dim: textDim, intermediateDim: textDim * convMult)
        }
        _textDim = textDim
        super.init()
    }

    func callAsFunction(_ text: MLXArray, seqLen: Int, dropText: Bool) -> MLXArray {
        var t = text + Int32(1)  // 0 is the filler token
        let nt = t.shape[1]
        if nt > seqLen {
            t = t[0..., 0 ..< seqLen]
        } else if nt < seqLen {
            t = MLX.padded(t, widths: [IntOrPair((0, 0)), IntOrPair((0, seqLen - nt))], value: MLXArray(Int32(0)))
        }
        // text_mask_padding: keep filler positions at zero
        let fillerMask = (t .== MLXArray(Int32(0))).expandedDimensions(axis: -1)
        if dropText {
            t = MLX.zeros(like: t)
        }

        var h = text_embed(t)
        let zero = MLXArray(Float(0)).asType(h.dtype)
        if !text_blocks.isEmpty {
            h = h + Self.sinusoidalPositions(dim: _textDim, length: seqLen).asType(h.dtype)
            h = MLX.where(fillerMask, zero, h)
            for block in text_blocks {
                h = block(h)
                h = MLX.where(fillerMask, zero, h)
            }
        }
        return h
    }

    // precompute_freqs_cis
    static func sinusoidalPositions(dim: Int, length: Int, theta: Double = 10000) -> MLXArray {
        let half = dim / 2
        let freqs = MLXArray((0 ..< half).map { Float(1.0 / Foundation.pow(theta, Double(2 * $0) / Double(dim))) })
        let positions = MLXArray(0 ..< length).asType(.float32)
        let angles = positions.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        return MLX.concatenated([MLX.cos(angles), MLX.sin(angles)], axis: -1)
    }
}

final class ConvPositionEmbedding: Module {
    let conv1d: [Conv1d]

    init(dim: Int, kernelSize: Int = 31, groups: Int = 16) {
        conv1d = (0 ..< 2).map { _ in
            Conv1d(
                inputChannels: dim, outputChannels: dim, kernelSize: kernelSize,
                padding: kernelSize / 2, groups: groups)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = mish(conv1d[0](x))
        return mish(conv1d[1](h))
    }
}

final class InputEmbedding: Module {
    let proj: Linear
    let conv_pos_embed: ConvPositionEmbedding

    init(melDim: Int, textDim: Int, outDim: Int) {
        proj = Linear(melDim * 2 + textDim, outDim)
        conv_pos_embed = ConvPositionEmbedding(dim: outDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cond: MLXArray, textEmbed: MLXArray) -> MLXArray {
        let h = proj(MLX.concatenated([x, cond, textEmbed], axis: -1))
        return conv_pos_embed(h) + h
    }
}

// MARK: - Transformer

final class AdaLayerNorm: Module {
    let linear: Linear

    init(dim: Int) {
        linear = Linear(dim, dim * 6)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, emb: MLXArray)
        -> (x: MLXArray, gateMSA: MLXArray, shiftMLP: MLXArray, scaleMLP: MLXArray, gateMLP: MLXArray)
    {
        let p = linear(silu(emb)).split(parts: 6, axis: 1)
        let normed = MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6)
        let out = normed * (p[1].expandedDimensions(axis: 1) + 1) + p[0].expandedDimensions(axis: 1)
        return (out, p[2], p[3], p[4], p[5])
    }
}

final class AdaLayerNormFinal: Module {
    let linear: Linear

    init(dim: Int) {
        linear = Linear(dim, dim * 2)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, emb: MLXArray) -> MLXArray {
        let p = linear(silu(emb)).split(parts: 2, axis: 1)  // scale, shift
        let normed = MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6)
        return normed * (p[0].expandedDimensions(axis: 1) + 1) + p[1].expandedDimensions(axis: 1)
    }
}

final class Attention: Module {
    let to_q: Linear
    let to_k: Linear
    let to_v: Linear
    let to_out: [Linear]
    let _heads: Int
    let _dimHead: Int

    init(dim: Int, heads: Int, dimHead: Int) {
        let inner = heads * dimHead
        to_q = Linear(dim, inner)
        to_k = Linear(dim, inner)
        to_v = Linear(dim, inner)
        to_out = [Linear(inner, dim)]
        _heads = heads
        _dimHead = dimHead
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.shape[0]
        let n = x.shape[1]

        func heads(_ y: MLXArray) -> MLXArray {
            y.reshaped([b, n, _heads, _dimHead]).transposed(0, 2, 1, 3)
        }
        var q = heads(to_q(x))
        var k = heads(to_k(x))
        let v = heads(to_v(x))

        // rope on all heads; x-transformers rotates interleaved pairs, i.e. traditional RoPE
        q = MLXFast.RoPE(q, dimensions: _dimHead, traditional: true, base: 10000, scale: 1, offset: 0)
        k = MLXFast.RoPE(k, dimensions: _dimHead, traditional: true, base: 10000, scale: 1, offset: 0)

        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(_dimHead).squareRoot(), mask: nil)
        return to_out[0](o.transposed(0, 2, 1, 3).reshaped([b, n, _heads * _dimHead]))
    }
}

final class FeedForward: Module {
    let ff: [Linear]

    init(dim: Int, mult: Int) {
        ff = [Linear(dim, dim * mult), Linear(dim * mult, dim)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        ff[1](geluApproximate(ff[0](x)))
    }
}

final class DiTBlock: Module {
    let attn_norm: AdaLayerNorm
    let attn: Attention
    let ff: FeedForward

    init(dim: Int, heads: Int, dimHead: Int, ffMult: Int) {
        attn_norm = AdaLayerNorm(dim: dim)
        attn = Attention(dim: dim, heads: heads, dimHead: dimHead)
        ff = FeedForward(dim: dim, mult: ffMult)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, t: MLXArray) -> MLXArray {
        let n = attn_norm(x, emb: t)
        var h = x + n.gateMSA.expandedDimensions(axis: 1) * attn(n.x)
        let normed = MLXFast.layerNorm(h, weight: nil, bias: nil, eps: 1e-6)
        let ffIn = normed * (n.scaleMLP.expandedDimensions(axis: 1) + 1) + n.shiftMLP.expandedDimensions(axis: 1)
        h = h + n.gateMLP.expandedDimensions(axis: 1) * ff(ffIn)
        return h
    }
}

public final class DiT: Module {
    let time_embed: TimestepEmbedding
    let text_embed: TextEmbedding
    let input_embed: InputEmbedding
    let transformer_blocks: [DiTBlock]
    let norm_out: AdaLayerNormFinal
    let proj_out: Linear

    init(_ c: DiTConfig) {
        time_embed = TimestepEmbedding(dim: c.dim)
        text_embed = TextEmbedding(textNumEmbeds: c.textNumEmbeds, textDim: c.textDim, convLayers: c.convLayers)
        input_embed = InputEmbedding(melDim: c.melDim, textDim: c.textDim, outDim: c.dim)
        transformer_blocks = (0 ..< c.depth).map { _ in
            DiTBlock(dim: c.dim, heads: c.heads, dimHead: c.dimHead, ffMult: c.ffMult)
        }
        norm_out = AdaLayerNormFinal(dim: c.dim)
        proj_out = Linear(c.dim, c.melDim)
        super.init()
    }

    var dtype: DType { proj_out.weight.dtype }

    func embedText(_ text: MLXArray, seqLen: Int, dropText: Bool) -> MLXArray {
        text_embed(text, seqLen: seqLen, dropText: dropText)
    }

    // returns the predicted flow in float32
    func callAsFunction(x: MLXArray, cond: MLXArray, textEmbed: MLXArray, time: MLXArray) -> MLXArray {
        let dt = dtype
        let t = time_embed(time, dtype: dt)
        var h = input_embed(x.asType(dt), cond: cond.asType(dt), textEmbed: textEmbed.asType(dt))
        for block in transformer_blocks {
            h = block(h, t: t)
        }
        h = norm_out(h, emb: t)
        return proj_out(h).asType(.float32)
    }
}
