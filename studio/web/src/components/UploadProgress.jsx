export default function UploadProgress({ label, percent, detail }) {
  return (
    <div className="upload-progress">
      <div className="upload-progress-label" role="status" aria-live="polite">
        <span>{label}</span>
        {percent !== null && <strong>{percent}%</strong>}
      </div>
      <progress aria-label={label} max="100" value={percent ?? undefined} />
      {detail && <p>{detail}</p>}
    </div>
  );
}
