import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import "../styles/About.css";

function About() {
  return (
    <>
      <Navbar />

      <main className="about-page">
        <div className="about-header">
          <h1>About the AI Image Detector</h1>
          <p>
            Understanding how Convolutional Neural Networks analyze frequency,
            texture artifacts, and structural cues to detect synthetic media.
          </p>
        </div>

        <section className="about-card">
          <h2>🔬 Model Architecture & Design</h2>
          <p>
            This system utilizes a deep <strong>Convolutional Neural Network (CNN)</strong> built in TensorFlow & Keras 3. 
            The architecture is engineered specifically to extract multi-scale spatial features, pixel correlation patterns, and texture anomalies:
          </p>

          <div className="pipeline-steps">
            <div className="step-box">
              <h3>1. Preprocessing</h3>
              <p>Embedded <code>Rescaling(1/255)</code> and data augmentation layers inside the model graph ensure robust, normalized inference.</p>
            </div>
            <div className="step-box">
              <h3>2. Feature Extraction</h3>
              <p>4 hierarchical <code>Conv2D</code> blocks (32, 64, 128, 256 filters) paired with Batch Normalization and MaxPooling.</p>
            </div>
            <div className="step-box">
              <h3>3. Classification Head</h3>
              <p>Dense layer with 128 ReLU units, 50% Dropout regularization, and a single Sigmoid neuron output.</p>
            </div>
          </div>
        </section>

        <section className="about-card">
          <h2>📊 Classification Logic & Confidence</h2>
          <p>
            The network produces a continuous probability score $P \in [0.0, 1.0]$:
          </p>
          <ul>
            <li><strong>REAL Image ($P \ge 0.5$):</strong> The confidence represents $P \times 100\%$, indicating alignment with authentic camera sensor noise, natural lighting distributions, and organic textures.</li>
            <li><strong>AI GENERATED Image ($P &lt; 0.5$):</strong> The confidence represents $(1 - P) \times 100\%$, indicating detection of synthetic generative artifacts (e.g. diffusion noise residuals, unnatural frequency signatures, smooth over-regularization).</li>
          </ul>
        </section>

        <section className="about-card">
          <h2>⚠️ Known Limitations & Edge Cases</h2>
          <p>
            While deep CNN classifiers achieve strong accuracy on standard benchmarks, machine learning detection methods face important real-world nuances:
          </p>
          <ul>
            <li><strong>Heavy Compression:</strong> Strong JPEG or WebP compression, repeated social media re-encoding, and severe downscaling can wipe out high-frequency sensor noise signatures.</li>
            <li><strong>Post-Processing & Filters:</strong> Heavy photographic post-processing (digital filters, HDR tone-mapping, artistic vignettes) may sometimes trigger synthetic feature flags.</li>
            <li><strong>Evolving Generative Models:</strong> Generative architectures (Diffusion models, FLUX, Midjourney v6) constantly refine output fidelity, necessitating continuous model retraining with diverse datasets.</li>
          </ul>

          <div className="limitations-alert">
            <h3>Best Practice Recommendation</h3>
            <p>
              Use automated detection tools as an assistive triage aid alongside human review, metadata provenance verification (such as C2PA / Content Credentials), and contextual analysis.
            </p>
          </div>
        </section>
      </main>

      <Footer />
    </>
  );
}

export default About;
