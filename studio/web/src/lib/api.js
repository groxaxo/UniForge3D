export const stages = [
  ["preprocess", "Prepare images"],
  ["dino", "Read visual features"],
  ["structure", "Build structure"],
  ["shape512", "Shape the object"],
  ["shape1024", "Refine geometry"],
  ["texture", "Generate texture"],
  ["decode", "Bake & export"],
];
export const terminal = (status) =>
  ["complete", "failed", "cancelled"].includes(status);
export const statusLabel = (status) =>
  ({
    queued: "Queued",
    uploading: "Sending images",
    waiting: "Waiting for GPU",
    running: "Generating",
    downloading: "Downloading",
    complete: "Ready",
    failed: "Failed",
    cancelled: "Cancelled",
  })[status] || status;
export async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...options.headers, "X-UniForge-Studio": "1" },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message =
        typeof body.detail === "string"
          ? body.detail
          : "Please check your images, name and seed.";
    } catch {
      /* preserve HTTP status */
    }
    throw new Error(message);
  }
  return response.json();
}
export const sizeLabel = (bytes) =>
  bytes ? `${(bytes / 1e6).toFixed(1)} MB` : "";
// XMLHttpRequest exposes actual bytes sent; fetch has no upload progress events.
export function uploadJob(form, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/jobs");
    request.setRequestHeader("X-UniForge-Studio", "1");
    request.responseType = "json";
    request.timeout = 10 * 60 * 1000;
    onProgress({ phase: "uploading", percent: 0 });
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress({
          phase: "uploading",
          percent: Math.min(100, Math.round(event.loaded / event.total * 100)),
        });
      }
    };
    request.upload.onload = () => onProgress({ phase: "checking", percent: 100 });
    request.onload = () => {
      if (request.status >= 200 && request.status < 300 && request.response?.id) {
        resolve(request.response);
      } else {
        reject(new Error(
          typeof request.response?.detail === "string"
            ? request.response.detail
            : `Upload failed (${request.status}). Please check your images and try again.`,
        ));
      }
    };
    request.onerror = () => reject(new Error("Upload failed. Check your connection and try again."));
    request.ontimeout = () => reject(new Error("The upload timed out. Check your connection and try again."));
    request.onabort = () => reject(new Error("The upload was cancelled. You can try again."));
    request.send(form);
  });
}
