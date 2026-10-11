# HabibiTest

Can [Habibi-TTS](https://github.com/SWivid/Habibi-TTS) speak Saudi Arabic in a cloned voice
directly on an iPad, with no server? This app tries it.

It runs the Saudi (SAU) model on-device through a Swift/[MLX](https://github.com/ml-explore/mlx-swift)
port of F5-TTS. You give it a short clip of someone speaking plus the exact words in the clip, and it
says five sentences in that voice. For each sentence it shows how long it took and how much memory
it used. Tested on an iPad Air (M2).

I made it to see whether on-device voice cloning is realistic on an iPad.

## Running it

You need Xcode 16 or newer and an iPad with Apple silicon (M1 or later). The simulator won't work,
MLX needs the real GPU.

1. Copy `Config/Signing.local.example.xcconfig` to `Config/Signing.local.xcconfig` and put in your
   team ID and a bundle ID. A free Apple account is enough.
2. Open `HabibiTest.xcodeproj`. Xcode fetches mlx-swift the first time, and may ask to install the
   Metal Toolchain.
3. Pick your iPad and press Run. The scheme builds Release so the timings mean something.

On the iPad:

1. **تنزيل النموذج** downloads the model files (about 1.4 GB) from Hugging Face.
2. **تجهيز النموذج للتشغيل** loads them into memory. The first time it also converts the weights
   to float16 and keeps that copy, so it takes longer.
3. Choose a reference voice: one of the three Saudi samples from the Habibi repo, or your own audio
   or video file. Pick 5 to 10 seconds where only that person is talking, and type exactly what they
   say.
4. **ابدأ الاختبار**. One warm-up sentence runs first and isn't counted.

Results (a JSON file plus WAVs of every sentence) are saved to Files › On My iPad › HabibiTest › Runs.

## What's in here

```
HabibiTest/           the app (SwiftUI)
Packages/F5Engine/    F5-TTS v1 + Vocos in Swift/MLX
```

`F5Engine` loads the original PyTorch checkpoint directly (`model_200000.safetensors` + `vocab.txt`),
so no conversion step on a computer is needed. It's based on Lucas Newman's
[f5-tts-swift](https://github.com/lucasnewman/f5-tts-swift) and
[vocos-swift](https://github.com/lucasnewman/vocos-swift), updated for the F5-TTS v1 architecture
that Habibi uses, with tokenisation that works for Arabic.

## License

The code is MIT, see [LICENSE](LICENSE).

The model weights are not part of this repo. The app downloads them when you ask it to. The Habibi
SAU model is **CC BY-NC-SA 4.0, non-commercial only**. Please check the
[model card](https://huggingface.co/SWivid/Habibi-TTS) before using it for anything.

## Credits

- [Habibi-TTS](https://github.com/SWivid/Habibi-TTS) (SJTU X-LANCE) and
  [F5-TTS](https://github.com/SWivid/F5-TTS)
- [f5-tts-swift](https://github.com/lucasnewman/f5-tts-swift) and
  [vocos-swift](https://github.com/lucasnewman/vocos-swift) by Lucas Newman
- [Vocos](https://github.com/gemelo-ai/vocos), [MLX](https://github.com/ml-explore/mlx-swift)
