package com.groxaxo.uniforge;

import org.junit.Test;
import static org.junit.Assert.*;

public class StudioEndpointTest {
    @Test public void acceptsBothQueuesAndAssetPaths() {
        assertTrue(StudioEndpoint.isAllowed(StudioEndpoint.MLX));
        assertTrue(StudioEndpoint.isAllowed(StudioEndpoint.CUDA + "api/jobs/123/download"));
        assertTrue(StudioEndpoint.isAllowed(StudioEndpoint.CUDA3090 + "api/jobs/123/download"));
    }
    @Test public void restoresAllThreeSelectionsAndMigratesTheOldCudaPreference() {
        assertEquals(StudioEndpoint.CUDA, StudioEndpoint.restore(null, true));
        assertEquals(StudioEndpoint.MLX, StudioEndpoint.restore(null, false));
        assertEquals(StudioEndpoint.CUDA3090, StudioEndpoint.restore(StudioEndpoint.CUDA3090, false));
        assertEquals(StudioEndpoint.MLX, StudioEndpoint.restore(StudioEndpoint.MLX, true));
        assertEquals(StudioEndpoint.MLX, StudioEndpoint.restore("http://unexpected.example/", false));
    }
    @Test public void refusesDifferentHostsAndPorts() {
        assertFalse(StudioEndpoint.isAllowed("http://100.85.200.56.evil.test:7861/"));
        assertFalse(StudioEndpoint.isAllowed("http://100.85.200.54:7861/"));
        assertFalse(StudioEndpoint.isAllowed("http://100.85.200.56:80/"));
    }
    @Test public void refusesCredentialsAndNonWebSchemes() {
        assertFalse(StudioEndpoint.isAllowed("http://user@100.85.200.56:7861/"));
        assertFalse(StudioEndpoint.isAllowed("javascript:alert(1)"));
        assertFalse(StudioEndpoint.isAllowed("file:///sdcard/Download/private.html"));
        assertFalse(StudioEndpoint.isAllowed("content://downloads/item/1"));
        assertFalse(StudioEndpoint.isAllowed("not a URL"));
    }
}
