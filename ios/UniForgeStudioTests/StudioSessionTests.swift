import Foundation
import WebKit
import XCTest
@testable import UniForgeStudio

@MainActor
final class StudioSessionTests: XCTestCase {
    func testDownloadPolicyHandoffDoesNotCoverAHealthyPage() {
        let session = StudioSession()
        session.webView(session.webView, didStartProvisionalNavigation: nil)
        session.webView(session.webView, didFailProvisionalNavigation: nil,
                        withError: NSError(domain: "WebKitErrorDomain", code: 102))
        XCTAssertNil(session.connectionError)
        XCTAssertFalse(session.isLoading)
    }

    func testCancelledNavigationEndsProgressWithoutAnOfflineMessage() {
        let session = StudioSession()
        session.webView(session.webView, didStartProvisionalNavigation: nil)
        session.webView(session.webView, didFailProvisionalNavigation: nil,
                        withError: NSError(domain: NSURLErrorDomain, code: NSURLErrorCancelled))
        XCTAssertNil(session.connectionError)
        XCTAssertFalse(session.isLoading)
    }

    func testActualOfflineErrorIsVisible() {
        let session = StudioSession()
        session.webView(session.webView, didFailProvisionalNavigation: nil,
                        withError: NSError(domain: NSURLErrorDomain, code: NSURLErrorNotConnectedToInternet))
        XCTAssertNotNil(session.connectionError)
    }

    func testMatchingCodeInAnUnrelatedDomainIsNotSilenced() {
        let session = StudioSession()
        session.webView(session.webView, didFailProvisionalNavigation: nil,
                        withError: NSError(domain: "Unrelated", code: NSURLErrorCancelled))
        XCTAssertNotNil(session.connectionError)
    }

    func testSuccessfulNavigationClearsAStaleOfflineState() {
        let session = StudioSession()
        session.webView(session.webView, didFailProvisionalNavigation: nil,
                        withError: NSError(domain: NSURLErrorDomain, code: NSURLErrorNotConnectedToInternet))
        session.webView(session.webView, didFinish: nil)
        XCTAssertNil(session.connectionError)
    }
}
