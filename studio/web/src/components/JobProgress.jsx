import {
  Check,
  Circle,
  LoaderCircle,
  RotateCcw,
  TriangleAlert,
} from "lucide-react";
import { stages, statusLabel, terminal } from "../lib/api.js";

export default function JobProgress({ job, onCancel, onRetry }) {
  const completed = new Set(job.completed_stages?.map((stage) => stage.phase));
  const failed = job.status === "failed" || job.status === "cancelled";
  return (
    <section
      className={`job-progress ${failed ? "has-failed" : ""}`}
      aria-live="polite"
    >
      {failed ? (
        <TriangleAlert size={30} />
      ) : (
        <LoaderCircle size={30} className="spin" />
      )}
      <h2>{failed ? statusLabel(job.status) : "Your asset is taking shape"}</h2>
      <p>{job.message}</p>
      <ol className="stage-list">
        {stages.map(([key, label]) => (
          <li
            key={key}
            className={
              completed.has(key) ? "done" : job.stage === key ? "active" : ""
            }
          >
            {completed.has(key) ? (
              <Check size={15} />
            ) : job.stage === key ? (
              <LoaderCircle className="spin" size={15} />
            ) : (
              <Circle size={12} />
            )}
            <span>{label}</span>
            {completed.has(key) && (
              <small>
                {Math.round(
                  job.completed_stages.find((stage) => stage.phase === key)
                    .seconds,
                )}
                s
              </small>
            )}
          </li>
        ))}
      </ol>
      <p className="stage-count">{completed.size} of 7 stages complete</p>
      {job.error && (
        <details className="job-error">
          <summary>Generation details</summary>
          <pre>{job.error}</pre>
        </details>
      )}
      {terminal(job.status) ? (
        <button className="secondary-button" onClick={() => onRetry(job)}>
          <RotateCcw size={16} /> Use these views again
        </button>
      ) : (
        <button
          className="quiet-button"
          disabled={job.cancel_requested}
          onClick={() => onCancel(job.id)}
        >
          {job.cancel_requested ? "Cancelling…" : "Cancel generation"}
        </button>
      )}
    </section>
  );
}
