import Foundation
import XCTest
@testable import UniForgeStudio

final class DownloadNamingTests: XCTestCase {
    private func fixedIdentifier() throws -> UUID {
        try XCTUnwrap(UUID(uuidString: "12345678-1234-1234-1234-123456789ABC"))
    }

    func testTraversalAndPathOnlySuggestionsCannotEscapeDownloads() throws {
        let identifier = try fixedIdentifier()
        let prefix = identifier.uuidString.lowercased() + "-"
        XCTAssertEqual(DownloadNaming.uniqueName(for: "../../face.glb", identifier: identifier), prefix + "face.glb")
        for value in ["", "/", "../../", "/tmp/", ".", "..", "...", "  .  "] {
            let name = DownloadNaming.uniqueName(for: value, identifier: identifier)
            XCTAssertEqual(name, prefix + "asset", value)
            XCTAssertFalse(name.contains("/"))
            XCTAssertFalse(name.contains("\\"))
        }
        XCTAssertEqual(DownloadNaming.uniqueName(for: ".glb", identifier: identifier), prefix + "asset.glb")
        XCTAssertEqual(DownloadNaming.uniqueName(for: ".json", identifier: identifier), prefix + "asset.json")
    }

    func testWindowsPathsUseOnlyTheSanitizedBasename() throws {
        let identifier = try fixedIdentifier()
        XCTAssertEqual(
            DownloadNaming.uniqueName(for: "C:\\Users\\Vlad\\Downloads\\face.GLB", identifier: identifier),
            identifier.uuidString.lowercased() + "-face.glb"
        )
        XCTAssertTrue(DownloadNaming.uniqueName(for: "C:\\", identifier: identifier).hasSuffix("-asset"))
    }

    func testControlAndProblematicCharactersAreRemoved() throws {
        let identifier = try fixedIdentifier()
        let name = DownloadNaming.uniqueName(for: "\u{0}\nface<bad>:*?\"|.json\r", identifier: identifier)
        XCTAssertEqual(name, identifier.uuidString.lowercased() + "-facebad.json")
        XCTAssertFalse(name.unicodeScalars.contains { CharacterSet.controlCharacters.contains($0) })
    }

    func testNewDownloadsAvoidCollisionsAndExplicitIdentifiersAreDeterministic() throws {
        let identifier = try fixedIdentifier()
        let first = DownloadNaming.uniqueName(for: "asset.glb")
        let second = DownloadNaming.uniqueName(for: "asset.glb")
        XCTAssertNotEqual(first, second)
        XCTAssertTrue(first.hasSuffix("-asset.glb"))
        XCTAssertEqual(
            DownloadNaming.uniqueName(for: "asset.glb", identifier: identifier),
            DownloadNaming.uniqueName(for: "asset.glb", identifier: identifier)
        )
        XCTAssertTrue(DownloadNaming.uniqueName(for: "asset.glb", identifier: identifier).hasPrefix("12345678-1234-1234-1234-123456789abc-"))
    }

    func testLongUnicodeNamesKeepSupportedExtensionsWithinTheByteLimit() throws {
        for suffix in [".glb", ".json"] {
            let name = DownloadNaming.uniqueName(for: String(repeating: "🐵", count: 300) + suffix, identifier: try fixedIdentifier())
            XCTAssertLessThanOrEqual(name.utf8.count, 240)
            XCTAssertTrue(name.hasSuffix(suffix))
            XCTAssertTrue(name.contains("🐵"))
        }
    }
}
