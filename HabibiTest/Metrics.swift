import Foundation
import os
import UIKit

enum Metrics {
    // phys_footprint, what iOS counts against the app's memory limit
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }

    static func availableMB() -> Double {
        Double(os_proc_available_memory()) / 1_048_576
    }

    static var thermal: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    static var deviceModel: String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { raw in
            String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
        }
    }

    static var physicalMemoryGB: Double {
        Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
    }

    @MainActor static var systemVersion: String { UIDevice.current.systemVersion }
}

// peak memory while generating, sampled every 100 ms
@MainActor
final class PeakSampler {
    private var timer: Timer?
    private(set) var peakFootprintMB: Double = 0
    private(set) var minAvailableMB: Double = .infinity

    func start() {
        peakFootprintMB = Metrics.footprintMB()
        minAvailableMB = Metrics.availableMB()
        timer?.invalidate()
        let t = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.sample() }
        }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    func sample() {
        peakFootprintMB = max(peakFootprintMB, Metrics.footprintMB())
        minAvailableMB = min(minAvailableMB, Metrics.availableMB())
    }

    func stop() {
        sample()
        timer?.invalidate()
        timer = nil
    }
}
