package com.groxaxo.uniforge;

import java.net.URI;

/** The app's only in-WebView origins are the user's private Studio queues. */
public final class StudioEndpoint {
    public static final String HOST = "100.85.200.56";
    public static final String MLX = "http://" + HOST + ":7861/";
    public static final String CUDA = "http://" + HOST + ":7863/";
    public static final String CUDA3090 = "http://" + HOST + ":7864/";
    private StudioEndpoint() {}

    public static String restore(String stored, boolean legacyCuda) {
        if (MLX.equals(stored) || CUDA.equals(stored) || CUDA3090.equals(stored)) return stored;
        return legacyCuda ? CUDA : MLX;
    }

    public static boolean isAllowed(String value) {
        try {
            URI uri = URI.create(value);
            return "http".equalsIgnoreCase(uri.getScheme()) && HOST.equals(uri.getHost())
                && uri.getRawUserInfo() == null
                && (uri.getPort() == 7861 || uri.getPort() == 7863 || uri.getPort() == 7864);
        } catch (IllegalArgumentException error) { return false; }
    }
}
