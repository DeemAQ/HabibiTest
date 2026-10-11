import F5Engine
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var model: AppModel
    @ObservedObject private var playerState: ClipPlayer
    @State private var showImporter = false
    @State private var confirmDelete = false

    init(player: ClipPlayer) {
        _playerState = ObservedObject(wrappedValue: player)
    }

    var body: some View {
        NavigationStack {
            Form {
                modelSection
                referenceSection
                sentencesSection
                settingsSection
                runSection
                if !model.results.isEmpty { resultsSection }
                logSection
                creditsSection
            }
            .navigationTitle("اختبار صوت حبيبي")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Text("ذاكرة التطبيق \(Fmt.mb(model.footprintMB)) · المتاح \(Fmt.mb(model.availableMB))")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
        }
        .environment(\.layoutDirection, .rightToLeft)
        .task {
            if model.filesReady && model.referenceSamples.isEmpty {
                await model.selectPreset("najdi")
            }
        }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.audio, .movie]) { result in
            switch result {
            case .success(let url):
                Task { await model.importClip(from: url) }
            case .failure(let error):
                model.referenceError = error.localizedDescription
            }
        }
    }

    // Model

    private var modelSection: some View {
        Section {
            if model.filesReady {
                LabeledContent("الملفات", value: "جاهزة · \(Fmt.bytes(model.diskUsedBytes))")
            } else if model.downloading {
                VStack(alignment: .leading, spacing: 6) {
                    ProgressView(value: model.downloadFraction)
                    Text("\(model.downloadLabel) · \(Int(model.downloadFraction * 100))٪ من \(Fmt.bytes(ModelStore.totalBytes))")
                        .font(.caption.monospacedDigit())
                    Button("إيقاف", role: .cancel) { model.cancelDownload() }
                }
            } else {
                Button {
                    Task { await model.downloadAll() }
                } label: {
                    Label("تنزيل النموذج (\(Fmt.bytes(ModelStore.totalBytes)))", systemImage: "arrow.down.circle")
                }
                Text("يُنزَّل نموذج Habibi-TTS السعودي من Hugging Face مرة واحدة. استخدمي الواي فاي وأبقي الآيباد على الشاحن.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let e = model.downloadError {
                Text(e).foregroundStyle(.red).font(.caption)
            }

            if model.filesReady {
                Picker("الدقة", selection: $model.precision) {
                    Text("نصف (float16)").tag(Precision.float16)
                    Text("كاملة (float32)").tag(Precision.float32)
                }
                .pickerStyle(.segmented)
                .disabled(model.loading || model.running)

                HStack {
                    Button {
                        Task { await model.loadModel() }
                    } label: {
                        Label(model.loadedPrecision == nil ? "تجهيز النموذج للتشغيل" : "إعادة التجهيز", systemImage: "cpu")
                    }
                    .disabled(model.loading || model.running)
                    if model.loading { ProgressView().padding(.leading, 8) }
                    Spacer()
                    if let p = model.loadedPrecision {
                        Text("جاهز: \(p.rawValue)").foregroundStyle(.green).font(.callout)
                    }
                }
                if model.loadedPrecision == nil && !model.loading {
                    Text("الملفات على الآيباد؛ هذه الخطوة تفتح النموذج في الذاكرة قبل الاختبار.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            if model.loading && !model.status.isEmpty {
                Text(model.status).font(.caption).foregroundStyle(.secondary)
            }
            if let info = model.loadInfo {
                VStack(alignment: .leading, spacing: 2) {
                    Text("وقت التجهيز \(Fmt.s(info.seconds))\(info.convertedFromCheckpoint ? " (مع التحويل أول مرة، اضغطي «إعادة التجهيز» لقياس الوقت العادي)" : "")")
                    Text("ذاكرة التطبيق \(Fmt.mb(info.footprintMB)) · GPU \(Fmt.mb(info.gpuActiveMB)) · المتاح \(Fmt.mb(info.availableMB))")
                }
                .font(.caption.monospacedDigit())
            }
            if let e = model.loadError {
                Text(e).foregroundStyle(.red).font(.caption).textSelection(.enabled)
            }
            if model.filesReady {
                Button("حذف ملفات النموذج", role: .destructive) { confirmDelete = true }
                    .disabled(model.loading || model.running)
                    .confirmationDialog("حذف ملفات النموذج من الآيباد؟", isPresented: $confirmDelete) {
                        Button("حذف", role: .destructive) { Task { await model.deleteFiles() } }
                    }
            }
        } header: {
            Text("النموذج")
        }
    }

    // Reference

    private var referenceSection: some View {
        Section {
            Picker("المقطع", selection: Binding(
                get: { model.referenceSource },
                set: { newValue in
                    if case .preset(let id) = newValue { Task { await model.selectPreset(id) } }
                }
            )) {
                ForEach(ModelStore.presetClips) { p in
                    Text(p.name).tag(AppModel.ReferenceSource.preset(p.id))
                }
                if model.referenceSource == .custom {
                    Text("مقطعي").tag(AppModel.ReferenceSource.custom)
                }
            }
            .pickerStyle(.segmented)
            .disabled(!model.filesReady || model.running)

            Button {
                showImporter = true
            } label: {
                Label("من الملفات (صوت أو فيديو)", systemImage: "folder")
            }
            .disabled(model.running)

            if !model.sourceSamples.isEmpty {
                LabeledContent("المصدر", value: "\(model.sourceName) · \(Fmt.s(model.sourceSeconds))")
                if model.referenceSource == .custom {
                    VStack(alignment: .leading) {
                        Text("يُستخدم من \(Fmt.clock(model.clipStart)) إلى \(Fmt.clock(model.clipEnd)) (\(Fmt.s(model.clipLength))) من أصل \(Fmt.clock(model.sourceSeconds))")
                            .font(.callout.monospacedDigit())
                        Text("البداية").font(.caption)
                        Slider(value: $model.clipStart, in: 0 ... max(0.1, model.sourceSeconds - 1), step: 0.1, onEditingChanged: { editing in
                            if !editing { model.prepareReference() }
                        })
                        Text("النهاية").font(.caption)
                        Slider(value: Binding(get: { model.clipEnd }, set: { model.setClipEnd($0) }),
                               in: 0 ... max(0.1, model.sourceSeconds), step: 0.1, onEditingChanged: { editing in
                            if !editing { model.prepareReference() }
                        })
                        Text("النموذج يأخذ \(Int(AppModel.maxClipSeconds)) ثانية كحد أقصى (الأفضل ٥–١٠، فيها صوتك فقط)..")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                HStack {
                    Button {
                        model.player.toggle(id: "ref", samples: model.referenceSamples)
                    } label: {
                        Label(playerState.playingID == "ref" ? "إيقاف" : "استماع للمقطع", systemImage: playerState.playingID == "ref" ? "stop.fill" : "play.fill")
                    }
                    .disabled(model.referenceSamples.isEmpty)
                    Spacer()
                    Text("بعد القص \(Fmt.s(model.referenceSeconds))").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("النص المقول في المقطع، كلمة بكلمة").font(.caption).foregroundStyle(.secondary)
                TextField("اكتب ما قيل في المقطع بالضبط", text: $model.transcript, axis: .vertical)
                    .lineLimit(2 ... 4)
                    .disabled(model.running)
            }
            if let e = model.referenceError {
                Text(e).foregroundStyle(.red).font(.caption)
            }
        } header: {
            Text("الصوت المرجعي")
        } footer: {
            Text("المقاطع الجاهزة من أمثلة Habibi السعودية. لصوتك: مقطع واضح بلا موسيقى ولا أصوات أخرى، والنص يطابق الكلام تماماً.")
        }
    }

    // Sentences + settings

    private var sentencesSection: some View {
        Section("الجمل الخمس") {
            ForEach(model.sentences.indices, id: \.self) { i in
                HStack(alignment: .firstTextBaseline) {
                    Text("\(i + 1)").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    TextField("جملة \(i + 1)", text: $model.sentences[i], axis: .vertical)
                        .disabled(model.running)
                }
            }
            Button("الجمل الافتراضية") { model.sentences = AppModel.defaultSentences }
                .disabled(model.running)
        }
    }

    private var settingsSection: some View {
        Section {
            Picker("عدد الخطوات", selection: $model.steps) {
                ForEach([32, 16, 12, 8], id: \.self) { Text("\($0)").tag($0) }
            }
            .pickerStyle(.segmented)
            HStack {
                Text("البذرة (seed)")
                Spacer()
                TextField("42", text: $model.seedText)
                    .keyboardType(.numberPad)
                    .multilineTextAlignment(.trailing)
                    .frame(maxWidth: 120)
            }
            Toggle("تسخين قبل القياس (لا يحسب)", isOn: $model.warmUp)
        } header: {
            Text("الإعدادات")
        } footer: {
            Text("٣٢ خطوة = إعداد Habibi الأصلي (أفضل جودة، أبطأ). ١٦ و١٢ أسرع بجودة قريبة غالباً. البذرة الثابتة تجعل النتائج قابلة للمقارنة.")
        }
        .disabled(model.running)
    }

    // Run + results

    private var runSection: some View {
        Section {
            Button {
                Task { await model.runAll() }
            } label: {
                Label("ابدأ الاختبار", systemImage: "play.circle.fill").font(.headline)
            }
            .disabled(!model.canRun)
            if model.loadedPrecision == nil {
                Text("جهز النموذج أولاً (قسم «النموذج»).").font(.caption).foregroundStyle(.secondary)
            }
            if model.running {
                VStack(alignment: .leading) {
                    Text(model.status).font(.callout)
                    ProgressView(value: model.stepFraction)
                }
            }
            if let e = model.runError {
                Text(e).foregroundStyle(.red).font(.caption).textSelection(.enabled)
            }
        }
    }

    private var resultsSection: some View {
        Section {
            ForEach(model.results) { r in
                HStack(alignment: .top, spacing: 12) {
                    Button {
                        model.playResult(r.id)
                    } label: {
                        Image(systemName: playerState.playingID == "s\(r.id)" ? "stop.circle.fill" : "play.circle.fill")
                            .font(.title)
                    }
                    .buttonStyle(.borderless)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("\(r.id). \(r.text)")
                        Text("الوقت \(Fmt.s(r.totalSeconds)) لصوت \(Fmt.s(r.audioSeconds)) · \(Fmt.x(r.realTimeFactor)) من الزمن الحقيقي")
                            .font(.callout.monospacedDigit().bold())
                        Text("تحضير \(Fmt.s(r.prepareSeconds)) · النموذج \(Fmt.s(r.flowSeconds)) · المُرمِّز \(Fmt.s(r.vocoderSeconds)) · \(r.totalFrames) إطار")
                            .font(.caption.monospacedDigit())
                        Text("ذروة ذاكرة التطبيق \(Fmt.mb(r.peakFootprintMB)) · ذروة GPU \(Fmt.mb(r.gpuPeakMB)) · أقل متاح \(Fmt.mb(r.minAvailableMB)) · الحرارة \(r.thermal)")
                            .font(.caption.monospacedDigit())
                        if !r.unknownCharacters.isEmpty {
                            Text("حروف غير معروفة: \(r.unknownCharacters.joined(separator: " "))")
                                .font(.caption).foregroundStyle(.orange)
                        }
                    }
                    .environment(\.layoutDirection, .rightToLeft)
                }
            }
            if let s = model.summary {
                VStack(alignment: .leading, spacing: 3) {
                    Text("المتوسط \(Fmt.s(s.meanTotalSeconds)) للجملة · \(Fmt.x(s.meanRealTimeFactor)) من الزمن الحقيقي")
                        .font(.headline.monospacedDigit())
                    Text("\(s.device) · iPadOS \(s.iOS) · \(String(format: "%.0f", s.ramGB)) GB · \(s.precision) · \(s.steps) خطوة · الحرارة \(s.thermalStart) → \(s.thermalEnd)")
                        .font(.caption.monospacedDigit())
                }
            }
            if !model.exportFiles.isEmpty {
                ShareLink(items: model.exportFiles) {
                    Label("مشاركة النتائج والتسجيلات", systemImage: "square.and.arrow.up")
                }
            }
        } header: {
            Text("النتائج")
        } footer: {
            Text("تُحفظ أيضاً في تطبيق الملفات ▸ على الآيباد ▸ HabibiTest ▸ Runs.")
        }
    }

    private var logSection: some View {
        Section {
            DisclosureGroup("السجل (\(model.log.count))") {
                ForEach(Array(model.log.enumerated()), id: \.offset) { _, line in
                    Text(line).font(.caption.monospaced()).textSelection(.enabled)
                }
            }
        }
    }

    private var creditsSection: some View {
        Section("الترخيص") {
            Text("""
            Habibi-TTS (SJTU X-LANCE): النموذج السعودي SAU مرخّص CC BY-NC-SA 4.0: للاستخدام غير التجاري فقط، ولا يُضمَّن في التطبيق بل يُنزَّل عند الطلب.
            F5-TTS (SWivid) · f5-tts-swift و vocos-swift (Lucas Newman، MIT) · Vocos (charactr، MIT) · MLX (Apple، MIT).
            """)
            .font(.caption)
            .environment(\.layoutDirection, .leftToRight)
        }
    }
}
