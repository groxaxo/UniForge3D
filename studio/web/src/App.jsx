import { Box, Info } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, uploadJob } from "./lib/api.js";
import GenerationForm from "./components/GenerationForm.jsx";
import ModelPreview from "./components/ModelPreview.jsx";
import AssetShelf from "./components/AssetShelf.jsx";

export default function App() {
  const [example, setExample] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [health, setHealth] = useState(null);
  const [images, setImages] = useState([]);
  const [name, setName] = useState("New asset");
  const [seed, setSeed] = useState(42);
  const [preserveFace, setPreserveFace] = useState(false);
  const [faceReference, setFaceReference] = useState(null);
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [selectedId, setSelectedId] = useState("example");
  const [submitting, setSubmitting] = useState(false);
  const [upload, setUpload] = useState(null);
  const referencesEdited = useRef(false);
  const loadExample = (value) => {
    const sample = value || example;
    if (!sample) return;
    setImages(
      sample.images.map((url, index) => ({
        id: `example-${index}`,
        url,
        name: `Example view ${index + 1}`,
      })),
    );
    setName(sample.name);
    setSeed(42);
    setError("");
  };
  useEffect(() => {
    let alive = true;
    api("/api/example")
      .then((value) => {
        if (alive) {
          setExample(value);
          if (!referencesEdited.current) loadExample(value);
        }
      })
      .catch((error) => {
        if (alive) setConnectionError(error.message);
      });
    const refreshJobs = () =>
      api("/api/jobs")
        .then((value) => {
          if (alive) {
            setJobs(value);
            setConnectionError("");
          }
        })
        .catch(() => {
          if (alive)
            setConnectionError(
              "Studio disconnected. Your jobs are saved; reconnect or refresh to continue.",
            );
        });
    const refreshHealth = () =>
      api("/api/health")
        .then((value) => {
          if (alive) setHealth(value);
        })
        .catch(() => {
          if (alive)
            setHealth({
              ready: false,
              connected: false,
              message: "The Studio API is unavailable.",
            });
        });
    refreshJobs();
    refreshHealth();
    const jobsTimer = setInterval(refreshJobs, 3000);
    const healthTimer = setInterval(refreshHealth, 15000);
    return () => {
      alive = false;
      clearInterval(jobsTimer);
      clearInterval(healthTimer);
    };
  }, []);
  const assets = [...jobs, ...(example?.model_url ? [example] : [])];
  const selected = assets.find((asset) => asset.id === selectedId) || assets[0];
  const generate = async () => {
    if (submitting || images.length < 1 || images.length > 4) return;
    setSubmitting(true);
    setUpload({ phase: "preparing", percent: 0 });
    setError("");
    try {
      const form = new FormData();
      form.set("name", name.trim());
      form.set("seed", seed);
      form.set("preserve_face", preserveFace ? "true" : "false");
      if (preserveFace && faceReference) {
        const face = faceReference.file || await fetch(faceReference.url).then((response) => {
          if (!response.ok) throw new Error("The face reference could not be loaded.");
          return response.blob();
        });
        form.set("face_reference", face, "face-reference.png");
      }
      const files = await Promise.all(
        images.map(async (image) => {
          const file =
            image.file ||
            (await fetch(image.url).then((response) => {
              if (!response.ok)
                throw new Error("A reference image could not be loaded.");
              return response.blob();
            }));
          return file;
        }),
      );
      files.forEach((file, index) =>
        form.append("files", file, `view-${index + 1}.png`),
      );
      const job = await uploadJob(form, setUpload);
      setJobs((current) => [job, ...current]);
      setSelectedId(job.id);
    } catch (error) {
      setError(error.message);
    } finally {
      setSubmitting(false);
      setUpload(null);
    }
  };
  const cancel = async (id) => {
    try {
      const job = await api(`/api/jobs/${id}/cancel`, { method: "POST" });
      setJobs((current) =>
        current.map((item) => (item.id === id ? job : item)),
      );
    } catch (error) {
      setError(error.message);
    }
  };
  const retry = (job) => {
    setImages(
      Array.from({ length: job.view_count || 4 }, (_, i) => i + 1).map((index) => ({
        id: `${job.id}-${index}`,
        url: `/api/jobs/${job.id}/images/${index}`,
        name: `View ${index}`,
      })),
    );
    setName(job.name);
    setSeed(job.seed);
    setPreserveFace(Boolean(job.preserve_face));
    setFaceReference(job.face_reference_url ? {url: job.face_reference_url, name: "Saved face reference"} : null);
    setError("");
    document
      .querySelector(".input-rail")
      ?.scrollIntoView({
        behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "auto"
          : "smooth",
      });
  };
  return (
    <>
      <header className="app-header">
        <div className="brand">
          <Box size={29} strokeWidth={2.1} />
          <span>UniForge</span>
          <i />
          <span className="studio-title">Studio</span>
        </div>
        <div
          className="gpu-status"
          title={
            health?.engine_id === "mlx"
              ? `${health.gpu || "Apple Silicon"} · Local MLX with the verified selective oQ5 bundle. One job runs at a time.`
              : health?.engine_id === "cuda3090"
              ? health.message
              : health?.gpu_error
              ? health.message
              : health?.connected
                ? `${health.gpu} · ${Math.round(health.used_mib / 1024)} / ${Math.round(health.total_mib / 1024)} GB used. CUDA source weights with ${health.encoder || "verified encoder"}. Jobs wait if this GPU is occupied.`
                : health?.message
          }
        >
          <span
            className={`status-dot ${health?.ready ? "online" : health?.connected ? "attention" : ""}`}
          />
          <span>
            {health?.engine_id === "mlx"
              ? health.ready ? "Local MLX ready" : "Local MLX needs attention"
              : health?.engine_id === "cuda3090"
              ? health.ready ? "RTX 3090 · .51 · GPUQ" : "RTX 3090 needs attention"
              : health?.gpu_error
              ? "GPU needs attention"
              : health?.connected
                ? health.engine_id === "cuda3060" ? "RTX 3060 · .54" : "Private GPU"
                : health
                  ? "GPU offline"
                  : "Connecting"}
          </span>
          <Info size={14} />
        </div>
      </header>
      {connectionError && (
        <div className="connection-banner" role="alert">
          {connectionError}
        </div>
      )}
      <main className="studio-layout">
        <GenerationForm
          images={images}
          setImages={(value) => {
            referencesEdited.current = true;
            setImages(value);
          }}
          onReferenceInput={() => { referencesEdited.current = true; }}
          name={name}
          setName={setName}
          seed={seed}
          setSeed={setSeed}
          preserveFace={preserveFace}
          setPreserveFace={setPreserveFace}
          faceReference={faceReference}
          setFaceReference={setFaceReference}
          error={error}
          setError={setError}
          onExample={() => loadExample()}
          onGenerate={generate}
          submitting={submitting}
          upload={upload}
          health={health}
        />
        <div className="workspace">
          <ModelPreview asset={selected} onCancel={cancel} onRetry={retry} />
          <AssetShelf
            assets={assets}
            selectedId={selected?.id}
            onSelect={setSelectedId}
            onRetry={retry}
          />
        </div>
      </main>
    </>
  );
}
