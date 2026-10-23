import "@google/model-viewer";
import { Eye, Maximize, Rotate3D, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import JobProgress from "./JobProgress.jsx";
import { sizeLabel } from "../lib/api.js";

export default function ModelPreview({ asset, onCancel, onRetry }) {
  const viewer = useRef(null);
  const textures = useRef([]);
  const [mode, setMode] = useState("textured");
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");
  const source = asset?.model_url;
  const applyMode = (value) => {
    const materials = viewer.current?.model?.materials;
    if (!materials) return;
    for (const [i, material] of materials.entries()) {
      const pbr = material.pbrMetallicRoughness;
      const original = textures.current[i];
      if (!original) continue;
      pbr.baseColorTexture.setTexture(
        value === "clay" ? null : original.texture,
      );
      pbr.metallicRoughnessTexture.setTexture(
        value === "clay" ? null : original.metallicRoughnessTexture,
      );
      pbr.setBaseColorFactor(
        value === "clay" ? [0.68, 0.71, 0.76, 1] : original.factor,
      );
      pbr.setRoughnessFactor(value === "clay" ? 0.95 : original.roughness);
      pbr.setMetallicFactor(value === "clay" ? 0 : original.metallic);
    }
  };
  useEffect(() => {
    setLoaded(false);
    setError("");
    setMode("textured");
    textures.current = [];
    const element = viewer.current;
    if (!element || !source) return;
    const onLoad = () => {
      textures.current = element.model.materials.map((material) => ({
        texture: material.pbrMetallicRoughness.baseColorTexture.texture,
        metallicRoughnessTexture:
          material.pbrMetallicRoughness.metallicRoughnessTexture.texture,
        factor: [...material.pbrMetallicRoughness.baseColorFactor],
        roughness: material.pbrMetallicRoughness.roughnessFactor,
        metallic: material.pbrMetallicRoughness.metallicFactor,
      }));
      setLoaded(true);
    };
    const onError = () =>
      setError(
        "The 3D preview could not load. You can still download the GLB from Your assets.",
      );
    element.addEventListener("load", onLoad);
    element.addEventListener("error", onError);
    if (element.loaded) onLoad();
    return () => {
      element.removeEventListener("load", onLoad);
      element.removeEventListener("error", onError);
    };
  }, [source]);
  const reset = () => {
    if (!viewer.current) return;
    viewer.current.cameraOrbit = "25deg 80deg 120%";
    viewer.current.cameraTarget = "auto auto auto";
    viewer.current.fieldOfView = "30deg";
    viewer.current.jumpCameraToGoal();
  };
  return (
    <section className="preview" aria-label="3D asset preview">
      <div className="preview-toolbar">
        <span className="preview-label">
          <Eye size={17} /> Preview
        </span>
        <div className="segmented" aria-label="Preview material">
          {["textured", "clay"].map((value) => (
            <button
              key={value}
              aria-pressed={mode === value}
              disabled={!loaded}
              className={mode === value ? "selected" : ""}
              onClick={() => {
                applyMode(value);
                setMode(value);
              }}
            >
              {value === "textured" ? "Textured" : "Clay"}
            </button>
          ))}
        </div>
      </div>
      {source ? (
        <>
          <div className="floor-grid" aria-hidden="true" />
          <model-viewer
            key={source}
            ref={viewer}
            src={source}
            alt={`Interactive 3D preview of ${asset.name}. Drag to orbit; scroll to zoom.`}
            camera-controls
            camera-orbit="25deg 80deg 120%"
            field-of-view="30deg"
            shadow-intensity="0.7"
            shadow-softness="1"
            environment-image="neutral"
            exposure="1.1"
            interaction-prompt="none"
            loading="eager"
            touch-action="pan-y"
            style={{ width: "100%", height: "100%" }}
          >
            <div
              slot="progress-bar"
              className="viewer-loading"
              hidden={loaded || Boolean(error)}
            >
              Loading 3D asset…
            </div>
          </model-viewer>
          {error && (
            <p className="preview-error" role="alert">
              {error}
            </p>
          )}
          {asset.face_preservation && (
            <p className={`face-result face-result-${asset.face_preservation.status}`} role="status">
              {asset.face_preservation.status === "applied" ? "✓ " : ""}{asset.face_preservation.message}
            </p>
          )}
          <div className="orbit-toolbar">
            <Rotate3D size={19} />
            <span>Drag to orbit</span>
            <i />
            <button
              className="icon-button"
              title="Reset camera"
              aria-label="Reset camera"
              onClick={reset}
            >
              <Maximize size={18} />
            </button>
          </div>
          <div className="preview-caption">
            <span>
              {asset.name}
              {asset.id === "example" ? " · example" : ""}
            </span>
            <span>{sizeLabel(asset.metadata?.bytes)}</span>
          </div>
        </>
      ) : asset ? (
        <JobProgress job={asset} onCancel={onCancel} onRetry={onRetry} />
      ) : (
        <div className="empty-preview">
          <RotateCcw size={30} />
          <h2>Your next asset starts here</h2>
          <p>Add four views, then generate your 3D object.</p>
        </div>
      )}
    </section>
  );
}
