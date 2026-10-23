package com.groxaxo.uniforge;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.DownloadManager;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Insets;
import android.net.Uri;
import android.net.http.SslError;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.webkit.ConsoleMessage;
import android.webkit.SslErrorHandler;
import android.webkit.URLUtil;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.PopupMenu;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

/** Native Android host for the private Studio service; generation stays on its backend. */
public class MainActivity extends Activity {
    private static final int PICK_IMAGES = 201;
    private WebView web;
    private TextView engineButton;
    private ProgressBar progress;
    private LinearLayout offline;
    private ValueCallback<Uri[]> fileCallback;
    private String endpoint;

    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }

    // The trusted Studio React UI needs JavaScript. Navigation is origin-limited,
    // file access is disabled and no JavaScript-to-native interface is exposed.
    @SuppressLint("SetJavaScriptEnabled")
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        endpoint = getPreferences(MODE_PRIVATE).getBoolean("cuda", false) ? StudioEndpoint.CUDA : StudioEndpoint.MLX;
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.WHITE);
        if (Build.VERSION.SDK_INT >= 30) {
            getWindow().setDecorFitsSystemWindows(false);
            root.setOnApplyWindowInsetsListener((view, insets) -> {
                Insets bars = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                Insets keyboard = insets.getInsets(WindowInsets.Type.ime());
                view.setPadding(bars.left, bars.top, bars.right, Math.max(bars.bottom, keyboard.bottom));
                return insets;
            });
            WindowInsetsController controller = getWindow().getInsetsController();
            if (controller != null) controller.setSystemBarsAppearance(
                WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS | WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS,
                WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS | WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS);
        }
        LinearLayout bar = new LinearLayout(this);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding(dp(12), 0, dp(8), 0);
        engineButton = new TextView(this);
        engineButton.setTextColor(Color.rgb(35, 58, 101));
        engineButton.setTextSize(16);
        engineButton.setPadding(dp(4), dp(12), dp(4), dp(12));
        engineButton.setOnClickListener(this::selectEngine);
        updateEngineTitle();
        bar.addView(engineButton, new LinearLayout.LayoutParams(0, dp(48), 1));
        Button refresh = new Button(this);
        refresh.setText("↻");
        refresh.setTextSize(23);
        refresh.setContentDescription(getString(R.string.refresh));
        refresh.setOnClickListener(view -> loadStudio());
        bar.addView(refresh, new LinearLayout.LayoutParams(dp(52), dp(48)));
        root.addView(bar);
        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        root.addView(progress, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(3)));
        FrameLayout frame = new FrameLayout(this);
        web = new WebView(this);
        web.setBackgroundColor(Color.WHITE);
        web.setImportantForAutofill(View.IMPORTANT_FOR_AUTOFILL_NO_EXCLUDE_DESCENDANTS);
        WebSettings settings = web.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(true); // User-selected document URIs for image uploads.
        settings.setGeolocationEnabled(false);
        settings.setUseWideViewPort(true);
        settings.setLoadWithOverviewMode(true);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG);
        web.setWebViewClient(new WebViewClient() {
            @Override public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                if (!request.isForMainFrame() || StudioEndpoint.isAllowed(request.getUrl().toString())) return false;
                Toast.makeText(MainActivity.this, getString(R.string.private_only), Toast.LENGTH_SHORT).show();
                return true;
            }
            @Override public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                if (request.isForMainFrame()) offline.setVisibility(View.VISIBLE);
            }
            @Override public void onReceivedSslError(WebView view, SslErrorHandler handler, SslError error) {
                handler.cancel(); // Never bypass certificate validation.
            }
        });
        web.setWebChromeClient(new WebChromeClient() {
            @Override public void onProgressChanged(WebView view, int value) {
                progress.setProgress(value);
                progress.setVisibility(value == 100 ? View.GONE : View.VISIBLE);
            }
            @Override public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback, FileChooserParams params) {
                if (fileCallback != null) fileCallback.onReceiveValue(null);
                fileCallback = callback;
                Intent picker = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                picker.addCategory(Intent.CATEGORY_OPENABLE);
                picker.setType("image/*");
                picker.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{"image/png", "image/jpeg", "image/webp"});
                picker.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, params.getMode() == FileChooserParams.MODE_OPEN_MULTIPLE);
                picker.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                try { startActivityForResult(picker, PICK_IMAGES); }
                catch (ActivityNotFoundException error) {
                    fileCallback.onReceiveValue(null); fileCallback = null;
                    Toast.makeText(MainActivity.this, getString(R.string.no_picker), Toast.LENGTH_LONG).show();
                }
                return true;
            }
            @Override public boolean onConsoleMessage(ConsoleMessage message) {
                if (message.messageLevel() == ConsoleMessage.MessageLevel.ERROR) Log.e("UniForgeWeb", message.message());
                return false;
            }
        });
        web.setDownloadListener((url, userAgent, disposition, mime, length) -> download(url, userAgent, disposition, mime));
        frame.addView(web, new FrameLayout.LayoutParams(-1, -1));
        offline = new LinearLayout(this);
        offline.setOrientation(LinearLayout.VERTICAL);
        offline.setGravity(Gravity.CENTER);
        offline.setPadding(dp(28), dp(28), dp(28), dp(28));
        offline.setBackgroundColor(Color.WHITE);
        TextView message = new TextView(this);
        message.setText(getString(R.string.unavailable));
        message.setTextSize(18);
        message.setGravity(Gravity.CENTER);
        offline.addView(message);
        Button retry = new Button(this);
        retry.setText(getString(R.string.retry));
        retry.setOnClickListener(view -> loadStudio());
        offline.addView(retry);
        offline.setVisibility(View.GONE);
        frame.addView(offline, new FrameLayout.LayoutParams(-1, -1));
        root.addView(frame, new LinearLayout.LayoutParams(-1, 0, 1));
        setContentView(root);
        if (Build.VERSION.SDK_INT >= 33) {
            getOnBackInvokedDispatcher().registerOnBackInvokedCallback(0, this::navigateBack);
        }
        if (state == null || web.restoreState(state) == null) loadStudio();
    }

    private void updateEngineTitle() {
        engineButton.setText(endpoint.equals(StudioEndpoint.CUDA) ? getString(R.string.engine_cuda) : getString(R.string.engine_mlx));
        engineButton.setContentDescription(getString(R.string.choose_engine));
    }

    private void selectEngine(View anchor) {
        PopupMenu menu = new PopupMenu(this, anchor);
        menu.getMenu().add(0, 1, 0, getString(R.string.menu_mlx));
        menu.getMenu().add(0, 2, 1, getString(R.string.menu_cuda));
        menu.setOnMenuItemClickListener(item -> {
            endpoint = item.getItemId() == 2 ? StudioEndpoint.CUDA : StudioEndpoint.MLX;
            getPreferences(MODE_PRIVATE).edit().putBoolean("cuda", item.getItemId() == 2).apply();
            updateEngineTitle(); loadStudio(); return true;
        });
        menu.show();
    }

    private void loadStudio() {
        offline.setVisibility(View.GONE);
        web.loadUrl(endpoint);
    }

    private void download(String url, String userAgent, String disposition, String mime) {
        if (!StudioEndpoint.isAllowed(url)) return;
        String name = URLUtil.guessFileName(url, disposition, mime).replaceAll("[^a-zA-Z0-9._-]", "-");
        int dot = name.lastIndexOf('.');
        name = (dot > 0 ? name.substring(0, dot) : name) + "-" + System.currentTimeMillis() + (dot > 0 ? name.substring(dot) : "");
        try {
            DownloadManager.Request request = new DownloadManager.Request(Uri.parse(url));
            request.addRequestHeader("User-Agent", userAgent);
            request.setTitle(name);
            if (mime != null) request.setMimeType(mime);
            request.setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED);
            request.setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, name);
            ((DownloadManager)getSystemService(DOWNLOAD_SERVICE)).enqueue(request);
            Toast.makeText(this, getString(R.string.saving), Toast.LENGTH_LONG).show();
        } catch (RuntimeException error) {
            Toast.makeText(this, getString(R.string.download_failed), Toast.LENGTH_LONG).show();
        }
    }

    private void navigateBack() { if (web.canGoBack()) web.goBack(); else finish(); }
    @Override protected void onSaveInstanceState(Bundle state) {
        web.saveState(state); super.onSaveInstanceState(state);
    }
    @Override protected void onActivityResult(int request, int result, Intent intent) {
        super.onActivityResult(request, result, intent);
        if (request == PICK_IMAGES && fileCallback != null) {
            Uri[] selection = null;
            if (result == RESULT_OK && intent != null) {
                if (intent.getClipData() != null) {
                    int count = intent.getClipData().getItemCount();
                    selection = new Uri[count];
                    for (int i = 0; i < count; i++) selection[i] = intent.getClipData().getItemAt(i).getUri();
                } else if (intent.getData() != null) selection = new Uri[]{intent.getData()};
            }
            fileCallback.onReceiveValue(selection); fileCallback = null;
        }
    }
    @Override protected void onDestroy() {
        if (fileCallback != null) { fileCallback.onReceiveValue(null); fileCallback = null; }
        web.destroy(); super.onDestroy();
    }
}
