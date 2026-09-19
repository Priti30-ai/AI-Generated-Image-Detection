import { useState } from "react";
import Navbar from "../components/Navbar";
import ImageUpload from "../components/ImageUpload";
import ResultCard from "../components/ResultCard";
import Footer from "../components/Footer";
import "../styles/Home.css";

function Home() {
  const [predictionResult, setPredictionResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handlePredictionSuccess = (result, preview) => {
    setPredictionResult(result);
    setPreviewUrl(preview);
  };

  const handleReset = () => {
    setPredictionResult(null);
    setPreviewUrl(null);
  };

  return (
    <>
      <Navbar />

      <main className="home-container">
        <section className="home-hero">
          <h1>
            AI-Generated <span>Image Detection</span>
          </h1>
          <p>
            Upload any image to verify authenticity and inspect whether it was
            captured by a camera or synthesized by AI models.
          </p>
        </section>

        {!predictionResult ? (
          <ImageUpload onPredictionSuccess={handlePredictionSuccess} />
        ) : (
          <ResultCard
            result={predictionResult}
            previewUrl={previewUrl}
            onReset={handleReset}
          />
        )}

        <section className="home-features">
          <div className="feature-pill">
            <span>⚡</span>
            <span>Real-time Inference</span>
          </div>
          <div className="feature-pill">
            <span>🛡️</span>
            <span>10MB Max File Size</span>
          </div>
          <div className="feature-pill">
            <span>🧠</span>
            <span>Deep CNN Backbone</span>
          </div>
          <div className="feature-pill">
            <span>🔒</span>
            <span>Private & In-Memory</span>
          </div>
        </section>
      </main>

      <Footer />
    </>
  );
}

export default Home;