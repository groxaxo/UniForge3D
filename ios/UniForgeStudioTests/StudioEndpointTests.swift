import Foundation
import XCTest
@testable import UniForgeStudio

final class StudioEndpointTests: XCTestCase {
    func testBothEnginesConstructTheExpectedPrivateURL() throws {
        let mlx = try XCTUnwrap(StudioEngine.mlx.url)
        let cuda = try XCTUnwrap(StudioEngine.cuda.url)
        XCTAssertEqual(mlx.absoluteString, "http://100.85.200.56:7861/")
        XCTAssertEqual(cuda.absoluteString, "http://100.85.200.56:7863/")
        XCTAssertEqual(Set(StudioEngine.allCases.map(\.id)).count, 2)
        XCTAssertTrue(StudioURLPolicy.allows(mlx))
        XCTAssertTrue(StudioURLPolicy.allows(cuda))
    }

    func testAssetAndQueryPathsRemainOnTheirAllowedOrigin() throws {
        for value in [
            "http://100.85.200.56:7861/api/jobs/abc123/download",
            "http://100.85.200.56:7863/api/jobs/abc123/receipt?download=true",
            "http://100.85.200.56:7861/?asset=abc123#preview"
        ] {
            let url = try XCTUnwrap(URL(string: value))
            XCTAssertTrue(StudioURLPolicy.allows(url), value)
        }
    }

    func testRejectsUnexpectedHostsAndPorts() throws {
        for value in [
            "http://100.85.200.56.evil.example:7861/",
            "http://100.85.200.56.:7861/",
            "http://100.85.200.54:7861/",
            "http://localhost:7861/",
            "http://100.85.200.56:80/",
            "http://100.85.200.56:7862/",
            "http://100.85.200.56/",
            "http://[::ffff:100.85.200.56]:7861/"
        ] {
            let url = try XCTUnwrap(URL(string: value))
            XCTAssertFalse(StudioURLPolicy.allows(url), value)
        }
    }

    func testRejectsUserInformationEvenWhenTheHostIsAllowed() throws {
        for value in [
            "http://user@100.85.200.56:7861/",
            "http://user:password@100.85.200.56:7863/",
            "http://@100.85.200.56:7861/",
            "http://100.85.200.56:7861@evil.example/"
        ] {
            let url = try XCTUnwrap(URL(string: value))
            XCTAssertFalse(StudioURLPolicy.allows(url), value)
        }
    }

    func testRejectsOtherSchemesAndRelativeURLs() throws {
        for value in [
            "https://100.85.200.56:7861/",
            "file:///private/reference.png",
            "javascript:alert(1)",
            "data:text/html,hello",
            "about:blank",
            "/api/jobs"
        ] {
            let url = try XCTUnwrap(URL(string: value))
            XCTAssertFalse(StudioURLPolicy.allows(url), value)
        }
    }
}
