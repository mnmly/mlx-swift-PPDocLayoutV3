import SwiftUI

@main
struct PPDocLayoutV3DemoApp: App {
    var body: some Scene {
        WindowGroup("PP-DocLayoutV3") {
            ContentView()
                .frame(minWidth: 940, minHeight: 660)
        }
        .windowResizability(.contentMinSize)
    }
}
