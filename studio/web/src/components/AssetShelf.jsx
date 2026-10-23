import { Download, FileCheck2, LoaderCircle, RotateCcw } from "lucide-react";
import { sizeLabel, statusLabel, terminal } from "../lib/api.js";

export default function AssetShelf({ assets, selectedId, onSelect, onRetry }) {
  return (
    <section className="asset-shelf" aria-labelledby="assets-heading">
      <div className="shelf-heading">
        <h2 id="assets-heading">Your assets</h2>
        <span>
          {assets.length} {assets.length === 1 ? "asset" : "assets"}
        </span>
      </div>
      <div className="asset-list">
        {assets.map((asset) => (
          <article
            className={`asset-row ${selectedId === asset.id ? "is-selected" : ""}`}
            key={asset.id}
          >
            <button
              className="asset-select"
              onClick={() => onSelect(asset.id)}
              aria-label={`Preview ${asset.name}${asset.id === "example" ? " example" : ""}`}
              aria-pressed={selectedId === asset.id}
            >
              <img src={`${asset.cover_url}?thumbnail=true`} alt="" />
              <div>
                <h3>
                  {asset.name}
                  {asset.id === "example" ? " · example" : ""}
                </h3>
                <p>
                  {asset.id === "example"
                    ? "Textured GLB · original encoder"
                    : asset.status === "complete"
                      ? `Textured GLB · ${sizeLabel(asset.metadata?.bytes)} · ${asset.engine === "mlx" ? "Local MLX" : asset.encoder || "CUDA"}`
                      : statusLabel(asset.status)}
                </p>
              </div>
            </button>
            <div className="asset-actions">
              {asset.download_url ? (
                <>
                  {asset.id !== "example" && (
                    <button
                      className="icon-button"
                      title="Reuse reference images"
                      aria-label={`Reuse reference images for ${asset.name}`}
                      onClick={() => onRetry(asset)}
                    >
                      <RotateCcw size={18} />
                    </button>
                  )}
                  {asset.receipt_url && (
                    <a
                      className="icon-button receipt-link"
                      href={asset.receipt_url}
                      title="Download generation receipt"
                      aria-label={`Download receipt for ${asset.name}`}
                    >
                      <FileCheck2 size={18} />
                    </a>
                  )}
                  <a
                    className="secondary-button"
                    href={asset.download_url}
                    download
                    aria-label={`Download GLB for ${asset.name}`}
                  >
                    <Download size={18} />
                    <span>Download GLB</span>
                  </a>
                </>
              ) : terminal(asset.status) ? (
                <button
                  className="secondary-button"
                  onClick={() => onRetry(asset)}
                >
                  Try again
                </button>
              ) : (
                <span className="asset-status">
                  <LoaderCircle size={16} className="spin" />
                  {statusLabel(asset.status)}
                </span>
              )}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
