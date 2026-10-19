import SwiftUI

@main
struct HabibiTestApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView(player: model.player)
                .environmentObject(model)
        }
    }
}
