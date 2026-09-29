package com.newbmp.mcus;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.ValueCallback;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import java.io.IOException;
import java.io.InputStream;
import java.net.URLConnection;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends Activity {
    private static final String APP_ASSET_HOST = "appassets.androidplatform.net";
    private static final String APP_ASSET_PREFIX = "/assets/";
    private WebView webView;

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Window window = getWindow();
        window.setStatusBarColor(Color.rgb(243, 243, 243));
        window.setNavigationBarColor(Color.rgb(249, 249, 249));
        int systemUiFlags = 0;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) systemUiFlags |= View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) systemUiFlags |= View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR;
        window.getDecorView().setSystemUiVisibility(systemUiFlags);
        webView = new WebView(this);
        webView.setBackgroundColor(Color.rgb(243, 243, 243));
        webView.setOverScrollMode(View.OVER_SCROLL_NEVER);
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setUserAgentString(settings.getUserAgentString() + " MCUSAndroid/1.5");
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        // The catalog is bundled with the APK. Keep WebView's normal cache so
        // returning to the app does not re-parse every stylesheet and script.
        settings.setCacheMode(WebSettings.LOAD_DEFAULT);
        // Present bundled files through an HTTPS origin and avoid broad
        // file:// access permissions.
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);
        settings.setDefaultTextEncodingName("utf-8");
        settings.setBuiltInZoomControls(false);
        settings.setDisplayZoomControls(false);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) settings.setSafeBrowsingEnabled(true);
        webView.setWebViewClient(new WebViewClient() {
            @Override public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                WebResourceResponse local = openBundledAsset(request.getUrl());
                return local != null ? local : super.shouldInterceptRequest(view, request);
            }
            @Override public WebResourceResponse shouldInterceptRequest(WebView view, String url) {
                WebResourceResponse local = openBundledAsset(Uri.parse(url));
                return local != null ? local : super.shouldInterceptRequest(view, url);
            }
            @Override public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                Uri uri = request.getUrl();
                if (isBundledAsset(uri)) return false;
                openExternal(uri); return true;
            }
            @Override public boolean shouldOverrideUrlLoading(WebView view, String url) {
                Uri uri = Uri.parse(url);
                if (isBundledAsset(uri)) return false;
                openExternal(uri); return true;
            }
        });
        webView.setWebChromeClient(new WebChromeClient());
        setContentView(webView);
        if (savedInstanceState == null) {
            webView.loadUrl("https://" + APP_ASSET_HOST + APP_ASSET_PREFIX + "index.html?build=1.5");
        } else webView.restoreState(savedInstanceState);
    }

    private boolean isBundledAsset(Uri uri) {
        return uri != null
                && "https".equals(uri.getScheme())
                && APP_ASSET_HOST.equals(uri.getHost())
                && uri.getPath() != null
                && uri.getPath().startsWith(APP_ASSET_PREFIX);
    }

    private WebResourceResponse openBundledAsset(Uri uri) {
        if (!isBundledAsset(uri)) return null;
        String assetPath = uri.getPath().substring(APP_ASSET_PREFIX.length());
        if (assetPath.isEmpty() || assetPath.contains("..") || assetPath.contains("\\")) return null;
        try {
            InputStream stream = getAssets().open(assetPath);
            String mime = URLConnection.guessContentTypeFromName(assetPath);
            if (mime == null) {
                if (assetPath.endsWith(".js")) mime = "application/javascript";
                else if (assetPath.endsWith(".css")) mime = "text/css";
                else if (assetPath.endsWith(".html")) mime = "text/html";
                else if (assetPath.endsWith(".svg")) mime = "image/svg+xml";
                else if (assetPath.endsWith(".wasm")) mime = "application/wasm";
                else mime = "application/octet-stream";
            }
            Map<String, String> headers = new HashMap<>();
            headers.put("Cache-Control", "no-cache");
            return new WebResourceResponse(mime, "UTF-8", 200, "OK", headers, stream);
        } catch (IOException ignored) {
            return null;
        }
    }

    private void openExternal(Uri uri) {
        try { startActivity(new Intent(Intent.ACTION_VIEW, uri)); } catch (Exception ignored) { }
    }

    @Override protected void onSaveInstanceState(Bundle outState) {
        webView.saveState(outState); super.onSaveInstanceState(outState);
    }

    @Override public void onBackPressed() {
        webView.evaluateJavascript("window.MCUL&&window.MCUL.handleAndroidBack?window.MCUL.handleAndroidBack():false", new ValueCallback<String>() {
            @Override public void onReceiveValue(String value) {
                if (!"true".equals(value)) {
                    if (webView.canGoBack()) webView.goBack(); else MainActivity.super.onBackPressed();
                }
            }
        });
    }

    @Override protected void onDestroy() {
        if (webView != null) { webView.loadUrl("about:blank"); webView.destroy(); }
        super.onDestroy();
    }
}
