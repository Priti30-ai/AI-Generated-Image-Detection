import "../styles/ResultCard.css";

function ResultCard({ result, previewUrl, onReset }) {
  if (!result) return null;

  const isReal = result.label === "REAL";
  const confidence = Number(result.confidence) || 0;

  return (
    <div className={`result-card ${isReal ? "is-real" : "is-ai"}`}>
      {previewUrl && (
        <div className="result-preview-wrapper">
          <img
            src={previewUrl}
            alt="Uploaded preview"
            className="result-preview-img"
          />
        </div>
      )}

      <div className={`result-badge ${isReal ? "badge-real" : "badge-ai"}`}>
        <span>{isReal ? "✓ Real Photograph" : "✦ AI Generated"}</span>
      </div>

      <div className="confidence-section">
        <div className="confidence-header">
          <span>Confidence Level</span>
          <span className={`confidence-score ${isReal ? "text-real" : "text-ai"}`}>
            {confidence.toFixed(1)}%
          </span>
        </div>
        <div className="confidence-track">
          <div
            className={`confidence-fill ${isReal ? "fill-real" : "fill-ai"}`}
            style={{ width: `${Math.min(Math.max(confidence, 0), 100)}%` }}
          />
        </div>
      </div>

      <p className="result-summary-text">
        {isReal
          ? "Our CNN classifier identified features consistent with authentic real-world photography."
          : "Our CNN classifier detected synthetic patterns, frequency signatures, or artifacts typical of AI image generation."}
      </p>

      <button className="reset-button" onClick={onReset}>
        Test Another Image
      </button>
    </div>
  );
}

export default ResultCard;
