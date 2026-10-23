import Foundation

/// Private generation queues exposed by the Mac's Studio service.
enum StudioEngine: String, CaseIterable, Identifiable, Sendable {
    case mlx
    case cuda
    case cuda3090

    static let host = "100.85.200.56"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .mlx: "Mac · Local MLX"
        case .cuda: "RTX 3060 · .54"
        case .cuda3090: "RTX 3090 · .51 · GPUQ"
        }
    }

    var shortName: String {
        switch self {
        case .mlx: "Mac MLX"
        case .cuda: "RTX 3060"
        case .cuda3090: "RTX 3090 · GPUQ"
        }
    }

    var port: Int {
        switch self {
        case .mlx: 7861
        case .cuda: 7863
        case .cuda3090: 7864
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
