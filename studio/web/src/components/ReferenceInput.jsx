import { ArrowLeft, ImagePlus, X } from "lucide-react";
import { useRef, useState } from "react";
import UploadProgress from "./UploadProgress.jsx";

let nextReferenceId = 0;

export default function ReferenceInput({
  images,
  onChange,
  onExample,
  onError,
  minViews = 1,
  disabled = false,
  onPreparing,
}) {
  const input = useRef(null);
  const reading = useRef(false);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState(null);
  const [preparing, setPreparing] = useState(false);
  const addFiles = (files) => {
    if (disabled || reading.current) return;
    const incoming = [...files];
    if (!incoming.length) return;
    if (images.length + incoming.length > 4) {
      onError("Use up to four views. Remove a view before replacing it.");
      return;
    }
    if (
      incoming.some(
        (file) =>
          !["image/png", "image/jpeg", "image/webp"].includes(file.type),
      )
    ) {
      onError("Please choose PNG, JPG or WebP images.");
      return;
    }
    if (incoming.some((file) => file.size > 15 * 1024 ** 2)) {
      onError("Each image must be 15 MB or smaller.");
      return;
    }
    reading.current = true;
    setPreparing(true);
    onPreparing?.(true);
    onError("");
    setProgress({ label: "Reading images", percent: 0 });
    const totalBytes = incoming.reduce((total, file) => total + file.size, 0);
    const loadedBytes = incoming.map(() => 0);
    const updateProgress = (index, loaded) => {
      loadedBytes[index] = loaded;
      const percent = totalBytes
        ? Math.min(100, Math.round(loadedBytes.reduce((sum, size) => sum + size, 0) / totalBytes * 100))
        : 0;
      setProgress({ label: "Reading images", percent });
    };
    Promise.all(
      incoming.map(
        (file, index) =>
          new Promise((resolve, reject) => {
            const reader = new FileReader();
            // These IDs are React keys, so they need no secure-context browser API.
            const id = `local-reference-${++nextReferenceId}`;
            reader.onprogress = (event) => updateProgress(index, event.loaded);
            reader.onload = () => {
              updateProgress(index, file.size);
              resolve({
                id,
                file,
                url: reader.result,
                name: file.name,
              });
            };
            reader.onerror = () =>
              reject(new Error("One of these files could not be opened."));
            reader.readAsDataURL(file);
          }),
      ),
    )
      .then((items) => {
        onChange((current) => [...current, ...items]);
        setProgress({ label: "Images ready", percent: 100 });
        onError("");
      })
      .catch((error) => {
        setProgress(null);
        onError(error.message);
      })
      .finally(() => {
        reading.current = false;
        setPreparing(false);
        onPreparing?.(false);
      });
  };
  return (
    <section className="reference-input" aria-labelledby="references-title">
      <div className="section-label">
        <h2 id="references-title">Reference images</h2>
        <span>{images.length} / 4</span>
      </div>
      <div className="reference-grid">
        {Array.from({ length: 4 }, (_, index) => {
          const image = images[index];
          return (
            <div
              className={`reference-tile ${image ? "has-image" : ""}`}
              key={image?.id || index}
            >
              {image ? (
                <>
                  <img
                    src={image.file ? image.url : `${image.url}?thumbnail=true`}
                    alt={`Reference view ${index + 1}: ${image.name || "object"}`}
                  />
                  <span className="view-number">0{index + 1}</span>
                  <button
                    type="button"
                    className="remove-view icon-button"
                    aria-label={`Remove view ${index + 1}`}
                    disabled={disabled || preparing}
                    onClick={() =>
                      onChange(images.filter((_, i) => i !== index))
                    }
                  >
                    <X size={15} />
                  </button>
                  {index > 0 && (
                    <button
                      type="button"
                      className="move-view icon-button"
                      aria-label={`Move view ${index + 1} left`}
                      disabled={disabled || preparing}
                      onClick={() => {
                        const copy = [...images];
                        [copy[index - 1], copy[index]] = [
                          copy[index],
                          copy[index - 1],
                        ];
                        onChange(copy);
                      }}
                    >
                      <ArrowLeft size={14} />
                    </button>
                  )}
                </>
              ) : (
                <button
                  type="button"
                  className="empty-view"
                  aria-label={`Add view ${index + 1}`}
                  disabled={disabled || preparing}
                  onClick={() => input.current.click()}
                >
                  <ImagePlus size={23} />
                  <span>View 0{index + 1}</span>
                </button>
              )}
            </div>
          );
        })}
      </div>
      <div
        className={`drop-zone ${dragging ? "is-dragging" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled && !reading.current) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          addFiles(event.dataTransfer.files);
        }}
      >
        <ImagePlus size={18} />
        <span>
          Drop images or{" "}
          <button
            type="button"
            className="text-link"
            disabled={disabled || preparing}
            onClick={() => input.current.click()}
          >
            browse files
          </button>
        </span>
        <input
          ref={input}
          id="reference-files"
          type="file"
          accept="image/png,image/jpeg,image/webp"
          multiple
          aria-label="Upload reference images"
          disabled={disabled || preparing}
          onChange={(event) => {
            addFiles(event.target.files);
            event.target.value = "";
          }}
        />
      </div>
      {progress && (preparing || images.some((image) => image.file)) && (
        <UploadProgress label={progress.label} percent={progress.percent} />
      )}
      <p className="helper">PNG, JPG or WebP · {minViews === 1 ? "1–4 views" : "4 views"} of the same object · first view is the anchor</p>
      <button
        type="button"
        className="text-link example-link"
        disabled={disabled || preparing}
        onClick={onExample}
      >
        Try the example
      </button>
    </section>
  );
}
