import SwiftUI

@main
@MainActor
struct UniForgeStudioApp: App {
    @State private var session = StudioSession()

    var body: some Scene {
        WindowGroup {
            StudioRootView(session: session)
        }
    }
}
