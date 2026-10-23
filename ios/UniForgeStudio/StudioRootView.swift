import SwiftUI
import WebKit

@MainActor
struct StudioRootView: View {
    @Bindable var session: StudioSession
    @State private var showDownloads = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if session.isLoading { ProgressView().progressViewStyle(.linear).accessibilityLabel("Loading Studio") }
                if session.activeDownloads > 0 {
                    HStack {
                        ProgressView().controlSize(.small)
                        Text("Saving download…").font(.footnote)
                        Spacer()
                    }
                    .padding(.horizontal).padding(.vertical, 6)
                }
                StudioWebView(session: session)
                    .overlay {
                        if let error = session.connectionError {
                            ContentUnavailableView {
                                Label("Studio is unavailable", systemImage: "network.slash")
                            } description: {
                                Text(error)
                            } actions: {
                                Button("Try again", action: session.reload)
                                    .buttonStyle(.borderedProminent)
                                    .accessibilityIdentifier("studio.retry")
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .background(.background)
                        }
                    }
            }
            .navigationTitle("UniForge")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Menu {
                        ForEach(StudioEngine.allCases) { engine in
                            Button {
                                session.select(engine)
                            } label: {
                                Label(engine.displayName, systemImage: session.engine == engine ? "checkmark" : "server.rack")
                            }
                        }
                    } label: {
                        Label(session.engine.shortName, systemImage: "server.rack")
                    }
                    .accessibilityLabel("Choose generation engine")
                    .accessibilityIdentifier("studio.engine-menu")
                }
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button(action: session.reload) { Image(systemName: "arrow.clockwise") }
                        .accessibilityLabel("Refresh Studio")
                        .accessibilityIdentifier("studio.reload")
                    Button { showDownloads = true } label: { Image(systemName: "arrow.down.circle") }
                        .accessibilityLabel("Downloads, \(session.files.count) saved")
                        .accessibilityIdentifier("studio.downloads")
                }
            }
            .onAppear(perform: session.startIfNeeded)
            .sheet(isPresented: $showDownloads) { DownloadsView(session: session) }
            .alert("Download", isPresented: Binding(
                get: { session.downloadError != nil },
                set: { if !$0 { session.downloadError = nil } }
            )) {
                Button("OK") { session.downloadError = nil }
            } message: { Text(session.downloadError ?? "") }
        }
    }
}

@MainActor
private struct StudioWebView: UIViewRepresentable {
    let session: StudioSession
    func makeUIView(context: Context) -> WKWebView { session.webView }
    func updateUIView(_ uiView: WKWebView, context: Context) {}
}

@MainActor
private struct DownloadsView: View {
    let session: StudioSession
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if session.files.isEmpty {
                    ContentUnavailableView("No downloads yet", systemImage: "arrow.down.circle", description:
                        Text("Download an asset or receipt in Studio. Your files appear here and in the Files app."))
                } else {
                    List(session.files) { file in
                        ShareLink(item: file.url) {
                            HStack {
                                Image(systemName: "doc.zipper")
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(file.displayName).lineLimit(2)
                                    Text(file.bytes.formatted(.byteCount(style: .file))).font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                Image(systemName: "square.and.arrow.up")
                            }
                        }
                    }
                }
            }
            .navigationTitle("Downloads")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
    }
}
