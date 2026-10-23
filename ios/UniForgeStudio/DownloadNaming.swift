import Foundation

/// Produces a unique leaf name suitable for the app's Documents/Downloads folder.
enum DownloadNaming {
    static func uniqueName(for suggested: String, identifier: UUID = UUID()) -> String {
        let prefix = identifier.uuidString.lowercased() + "-"
        let separators = CharacterSet(charactersIn: "/\\")
        let invalid = CharacterSet.controlCharacters.union(CharacterSet(charactersIn: "<>:\"/\\|?*"))
        let trimmed = suggested.trimmingCharacters(in: .whitespacesAndNewlines)
        let hasTrailingSeparator = trimmed.unicodeScalars.last.map(separators.contains) ?? false
        let leaf = hasTrailingSeparator ? "" : trimmed.components(separatedBy: separators).last ?? ""
        let sanitized = String(leaf.unicodeScalars.filter { !invalid.contains($0) })
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let suffix: String
        if sanitized.lowercased().hasSuffix(".glb") {
            suffix = ".glb"
        } else if sanitized.lowercased().hasSuffix(".json") {
            suffix = ".json"
        } else {
            suffix = ""
        }
        let stem = suffix.isEmpty ? sanitized : String(sanitized.dropLast(suffix.count))
        let cleanStem = stem.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: ".")))
        let candidate = cleanStem.isEmpty ? "asset" : cleanStem
        let maximumStemBytes = 240 - prefix.utf8.count - suffix.utf8.count
        var shortened = ""
        var usedBytes = 0
        for character in candidate {
            let count = character.utf8.count
            guard usedBytes + count <= maximumStemBytes else { break }
            shortened.append(character)
            usedBytes += count
        }
        if shortened.isEmpty { shortened = "asset" }
        return prefix + shortened + suffix
    }
}
