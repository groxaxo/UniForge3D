import { ArrowRight, LoaderCircle, Shuffle } from "lucide-react";
import ReferenceInput from "./ReferenceInput.jsx";
import { useState } from "react";
import UploadProgress from "./UploadProgress.jsx";

export default function GenerationForm({
  images,
  setImages,
  name,
  setName,
  seed,
  setSeed,
  error,
  setError,
  onExample,
  onGenerate,
  submitting,
  health,
  upload,
  onReferenceInput,
}) {
  const [preparing, setPreparing] = useState(false);
  const ready =
    images.length >= (health?.min_views || 4) && images.length <= 4 &&
    name.trim() &&
    Number.isInteger(Number(seed)) &&
    Number(seed) >= 0 &&
    Number(seed) <= 4294967295;
  return (
    <aside className="input-rail">
      <div className="rail-heading">
        <h1>Create an asset</h1>
        <p>Turn your images into a textured 3D object.</p>
      </div>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          onGenerate();
        }}
      >
        <ReferenceInput
          images={images}
          onChange={setImages}
          onExample={onExample}
          onError={setError}
          minViews={health?.min_views || 4}
          disabled={submitting}
          onPreparing={(value) => {
            setPreparing(value);
            if (value) onReferenceInput();
          }}
        />
        <div className="generation-fields">
          <label className="field-row">
            <span>Asset name</span>
            <input
              required
              maxLength={80}
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Name your asset"
            />
          </label>
          <div className="field-row">
            <label htmlFor="seed">Seed</label>
            <div className="seed-field">
              <input
                id="seed"
                required
                type="number"
                min="0"
                max="4294967295"
                step="1"
                value={seed}
                onChange={(event) => setSeed(event.target.value)}
              />
              <button
                type="button"
                className="icon-button"
                aria-label="Randomize seed"
                onClick={() =>
                  setSeed(crypto.getRandomValues(new Uint32Array(1))[0])
                }
              >
                <Shuffle size={17} />
              </button>
            </div>
          </div>
          <dl className="model-settings">
            <div>
              <dt>Geometry</dt>
              <dd>1024</dd>
            </div>
            <div>
              <dt>Texture</dt>
              <dd>2048</dd>
            </div>
            <div>
              <dt>Encoder</dt>
              <dd>{health?.encoder || "Original DINOv3"}</dd>
            </div>
          </dl>
        </div>
        <div className="generation-footer">
          <p className="engine-note">
            {health?.engine_id === "mlx"
              ? "Runs locally on this Mac. Allow about 30 minutes for one image; additional views can take longer."
              : "Runs on your private GPU with CUDA source weights."}
          </p>
          {!health?.connected && (
            <p className="connection-note" role="status">
              {health?.message || "Checking the generation engine…"}
            </p>
          )}
          {health?.connected && !health.ready && (
            <p className="connection-note" role="status">
              {health.message ||
                "The GPU needs its Studio worker, encoder and 4 GB of free space."}
            </p>
          )}
          {error && (
            <div className="form-error" role="alert">
              {error}
            </div>
          )}
          <button
            type="submit"
            className="primary-button"
            disabled={!ready || submitting || preparing || !health?.ready}
          >
            {submitting ? (
              <>
                <LoaderCircle className="spin" size={18} />
                {upload?.phase === "checking" ? "Checking images…" : "Uploading images…"}
              </>
            ) : (
              <>
                Generate asset <ArrowRight size={19} />
              </>
            )}
          </button>
          {upload && (
            <UploadProgress
              label={upload.phase === "preparing" ? "Preparing upload" : upload.phase === "checking" ? "Upload complete" : "Uploading images"}
              percent={upload.phase === "preparing" ? null : upload.percent}
              detail={upload.phase === "checking" ? "100% sent · Checking images and adding your job…" : upload.phase === "preparing" ? "Getting your reference images ready to send…" : "Keep this page open while your images upload."}
            />
          )}
        </div>
      </form>
    </aside>
  );
}
