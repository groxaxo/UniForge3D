import Foundation
import Observation
import WebKit

struct StudioDownload: Identifiable {
    let id: UUID
    let url: URL
    let bytes: Int64

    var displayName: String {
        let name = url.lastPathComponent
        guard UUID(uuidString: String(name.prefix(36))) != nil,
              name.dropFirst(36).first == "-" else { return name }
        return String(name.dropFirst(37))
    }
}

/// WebKit delegates and UI state are owned entirely by the main actor.
@MainActor
@Observable
final class StudioSession: NSObject, WKNavigationDelegate, WKUIDelegate, WKDownloadDelegate {
    let webView: WKWebView
    private(set) var engine: StudioEngine
    private(set) var isLoading = false
    private(set) var activeDownloads = 0
    private(set) var files: [StudioDownload] = []
    var connectionError: String?
    var downloadError: String?
    private var destinations: [ObjectIdentifier: URL] = [:]
    private var retainedDownloads: [ObjectIdentifier: WKDownload] = [:]

    override init() {
        engine = StudioEngine(rawValue: UserDefaults.standard.string(forKey: "studioEngine") ?? "") ?? .mlx
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        configuration.allowsInlineMediaPlayback = true
        webView = WKWebView(frame: .zero, configuration: configuration)
        super.init()
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsBackForwardNavigationGestures = true
        webView.accessibilityIdentifier = "studio.webview"
        restoreDownloads()
    }

    func startIfNeeded() {
        if webView.url == nil { loadEngine() }
    }

    func select(_ value: StudioEngine) {
        guard engine != value else { return }
        engine = value
        UserDefaults.standard.set(value.rawValue, forKey: "studioEngine")
        loadEngine()
    }

    func reload() {
        connectionError = nil
        if webView.url == nil { loadEngine() }
        else { webView.reload() }
    }

    private func loadEngine() {
        guard let url = engine.url else {
            connectionError = "The Studio address could not be configured."
            return
        }
        connectionError = nil
        webView.load(URLRequest(url: url))
    }

    private var downloadsDirectory: URL {
        URL.documentsDirectory.appending(path: "Downloads", directoryHint: .isDirectory)
    }

    private var stagingDirectory: URL {
        URL.cachesDirectory.appending(path: "UniForgeDownloads", directoryHint: .isDirectory)
    }

    private func restoreDownloads() {
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: downloadsDirectory, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else { return }
        files = urls.compactMap { url in
            guard let values = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]),
                  values.isRegularFile == true else { return nil }
            return StudioDownload(id: UUID(), url: url, bytes: Int64(values.fileSize ?? 0))
        }.sorted { $0.url.lastPathComponent > $1.url.lastPathComponent }
    }

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation?) {
        isLoading = true
        connectionError = nil
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation?) {
        isLoading = false
        connectionError = nil
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation?, withError error: any Error) {
        recordNavigationFailure(error)
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation?, withError error: any Error) {
        recordNavigationFailure(error)
    }

    private func recordNavigationFailure(_ error: any Error) {
        let failure = error as NSError
        isLoading = false
        let cancelled = failure.domain == NSURLErrorDomain && failure.code == NSURLErrorCancelled
        // WebKit ends the original frame when policy hands navigation to WKDownload.
        let downloadHandoff = failure.domain == "WebKitErrorDomain" && failure.code == 102
        guard !cancelled && !downloadHandoff else { return }
        connectionError = "Connect to Tailscale and keep UniForge Studio running on your Mac, then try again."
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        isLoading = false
        connectionError = "The preview stopped. Reload Studio to reconnect to your saved assets."
    }

    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping @MainActor (WKNavigationActionPolicy) -> Void) {
        guard let url = navigationAction.request.url, StudioURLPolicy.allows(url) else {
            decisionHandler(.cancel)
            return
        }
        decisionHandler(navigationAction.shouldPerformDownload ? .download : .allow)
    }

    func webView(_ webView: WKWebView, decidePolicyFor navigationResponse: WKNavigationResponse,
                 decisionHandler: @escaping @MainActor (WKNavigationResponsePolicy) -> Void) {
        guard let url = navigationResponse.response.url, StudioURLPolicy.allows(url) else {
            decisionHandler(.cancel)
            return
        }
        let attachment = (navigationResponse.response as? HTTPURLResponse)?
            .value(forHTTPHeaderField: "Content-Disposition")?.lowercased().contains("attachment") == true
        decisionHandler(attachment || !navigationResponse.canShowMIMEType ? .download : .allow)
    }

    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) {
        retain(download)
    }

    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload) {
        retain(download)
    }

    private func retain(_ download: WKDownload) {
        connectionError = nil
        isLoading = false
        let key = ObjectIdentifier(download)
        retainedDownloads[key] = download
        download.delegate = self
        activeDownloads = retainedDownloads.count
    }

    func download(_ download: WKDownload, decideDestinationUsing response: URLResponse, suggestedFilename: String,
                  completionHandler: @escaping @MainActor (URL?) -> Void) {
        guard let url = response.url, StudioURLPolicy.allows(url) else {
            completionHandler(nil)
            return
        }
        do {
            try FileManager.default.createDirectory(at: stagingDirectory, withIntermediateDirectories: true)
            let directory = stagingDirectory
            let destination = directory.appending(path: DownloadNaming.uniqueName(for: suggestedFilename))
            destinations[ObjectIdentifier(download)] = destination
            completionHandler(destination)
        } catch {
            downloadError = "The download could not be saved. Please try again."
            completionHandler(nil)
        }
    }

    func download(_ download: WKDownload, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                  decisionHandler: @escaping @MainActor (WKDownload.RedirectPolicy) -> Void) {
        decisionHandler(request.url.map(StudioURLPolicy.allows) == true ? .allow : .cancel)
    }

    func downloadDidFinish(_ download: WKDownload) {
        let key = ObjectIdentifier(download)
        if let temporary = destinations.removeValue(forKey: key) {
            do {
                let values = try temporary.resourceValues(forKeys: [.fileSizeKey])
                guard let size = values.fileSize, size > 0 else {
                    throw CocoaError(.fileReadCorruptFile)
                }
                try FileManager.default.createDirectory(at: downloadsDirectory, withIntermediateDirectories: true)
                var directory = downloadsDirectory
                var backup = URLResourceValues()
                backup.isExcludedFromBackup = true
                try directory.setResourceValues(backup)
                let final = directory.appending(path: temporary.lastPathComponent)
                try FileManager.default.moveItem(at: temporary, to: final)
                files.insert(StudioDownload(id: UUID(), url: final, bytes: Int64(size)), at: 0)
            } catch {
                try? FileManager.default.removeItem(at: temporary)
                downloadError = "The download could not be saved. Your asset remains available in Studio."
            }
        }
        retainedDownloads.removeValue(forKey: key)
        activeDownloads = retainedDownloads.count
    }

    func download(_ download: WKDownload, didFailWithError error: any Error, resumeData: Data?) {
        let key = ObjectIdentifier(download)
        if let partial = destinations.removeValue(forKey: key) { try? FileManager.default.removeItem(at: partial) }
        retainedDownloads.removeValue(forKey: key)
        activeDownloads = retainedDownloads.count
        downloadError = "The download did not finish. Your generated asset is still saved in Studio."
    }
}
