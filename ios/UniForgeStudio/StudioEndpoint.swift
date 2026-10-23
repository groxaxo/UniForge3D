import Foundation

/// The two private generation queues exposed by the Mac's Studio service.
enum StudioEngine: String, CaseIterable, Identifiable, Sendable {
    case mlx
    case cuda

    static let host = "100.85.200.56"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .mlx: "Mac · Local MLX"
        case .cuda: "RTX 3060 · .54"
        }
    }

    var port: Int {
        switch self {
        case .mlx: 7861
        case .cuda: 7863
        }
    }

    /// Returns nil if URL construction ever becomes invalid after configuration changes.
    var url: URL? {
        var components = URLComponents()
        components.scheme = "http"
        components.host = Self.host
        components.port = port
        components.path = "/"
        return components.url
    }
}

/// Only the exact private Studio origins may navigate inside the app's web view.
enum StudioURLPolicy {
    static func allows(_ url: URL) -> Bool {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
              components.scheme?.lowercased() == "http",
              components.host == StudioEngine.host,
              components.user == nil,
              components.password == nil,
              let port = components.port else {
            return false
        }
        return StudioEngine.allCases.contains { $0.port == port }
    }
}
