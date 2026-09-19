import { useState, useRef, useEffect } from "react";
import { predictImage } from "../services/api";
import "../styles/ImageUpload.css";

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10MB

// Sample test images (Data URIs for instant testing of real & AI concepts)
const SAMPLE_PRESETS = [
  {
    name: "AI Generated Art",
    type: "AI",
    url: "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=600&auto=format&fit=crop&q=80",
  },
  {
    name: "Real Camera Photo",
    type: "Real",
    url: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600&auto=format&fit=crop&q=80",
  },
  {
    name: "Real Landscape",
    type: "Real",
    url: "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=600&auto=format&fit=crop&q=80",
  },
];

function ImageUpload({ onPredictionSuccess }) {
  const [activeTab, setActiveTab] = useState("upload"); // 'upload' | 'camera' | 'url' | 'samples'
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [isDragActive, setIsDragActive] = useState(false);
  const [imageUrlInput, setImageUrlInput] = useState("");
  const [isCameraActive, setIsCameraActive] = useState(false);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // Global Clipboard Paste (Ctrl+V) listener
  useEffect(() => {
    const handlePaste = (event) => {
      const items = event.clipboardData?.items;
      if (!items) return;

      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf("image") !== -1) {
          const blob = items[i].getAsFile();
          if (blob) {
            handleFileSelection(blob, "pasted-image.png");
            break;
          }
        }
      }
    };

    window.addEventListener("paste", handlePaste);
    return () => {
      window.removeEventListener("paste", handlePaste);
    };
  }, []);

  // Cleanup camera stream on unmount or tab switch
  useEffect(() => {
    if (activeTab !== "camera") {
      stopCamera();
    }
  }, [activeTab]);

  const handleFileSelection = (file, customName) => {
    setErrorMessage("");

    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setErrorMessage("Please select a valid image format (PNG, JPG, JPEG, WEBP, GIF, BMP).");
      return;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      setErrorMessage("File size exceeds 10MB. Please choose a smaller image.");
      return;
    }

    const namedFile = customName
      ? new File([file], customName, { type: file.type })
      : file;

    setSelectedFile(namedFile);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelection(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragActive(true);
  };

  const handleDragLeave = () => {
    setIsDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFileSelection(file);
    }
  };

  const handleRemoveFile = () => {
    if (previewUrl && !previewUrl.startsWith("http")) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setErrorMessage("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Live Camera handlers
  const startCamera = async () => {
    setErrorMessage("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsCameraActive(true);
    } catch (err) {
      console.error("Camera access error:", err);
      setErrorMessage("Unable to access camera. Please allow camera permissions in your browser.");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsCameraActive(false);
  };

  const captureFrame = () => {
    if (!videoRef.current) return;

    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (blob) {
        handleFileSelection(blob, "live-capture.jpg");
        stopCamera();
      }
    }, "image/jpeg", 0.95);
  };

  // URL Loader handler
  const handleLoadUrl = async (e) => {
    e.preventDefault();
    if (!imageUrlInput.trim()) return;

    setIsLoading(true);
    setErrorMessage("");

    try {
      const response = await fetch(imageUrlInput);
      if (!response.ok) throw new Error("Could not download image from the provided URL.");

      const blob = await response.blob();
      if (!blob.type.startsWith("image/")) {
        throw new Error("URL does not point to a valid image.");
      }

      handleFileSelection(blob, "web-image.jpg");
      setImageUrlInput("");
    } catch (err) {
      console.error("URL load error:", err);
      setErrorMessage(err.message || "Failed to load image from URL. CORS restrictions may apply to external links.");
    } finally {
      setIsLoading(false);
    }
  };

  // Sample Preset loader
  const handleSelectSample = async (sample) => {
    setIsLoading(true);
    setErrorMessage("");

    try {
      const response = await fetch(sample.url);
      const blob = await response.blob();
      handleFileSelection(blob, `${sample.name.toLowerCase().replace(/\s+/g, "_")}.jpg`);
    } catch (err) {
      console.error("Failed loading sample:", err);
      setErrorMessage("Could not load sample image. Please upload a file instead.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setErrorMessage("Please select, paste, or capture an image first.");
      return;
    }

    setIsLoading(true);
    setErrorMessage("");

    try {
      const result = await predictImage(selectedFile);
      if (onPredictionSuccess) {
        onPredictionSuccess(result, previewUrl);
      }
    } catch (err) {
      console.error("Prediction failed:", err);
      const serverError =
        err.response?.data?.detail ||
        err.message ||
        "Failed to connect to backend server. Ensure FastAPI backend is running.";
      setErrorMessage(serverError);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="upload-container">
      {/* Input Source Tabs */}
      {!previewUrl && (
        <div className="input-source-tabs">
          <button
            type="button"
            className={`tab-btn ${activeTab === "upload" ? "active" : ""}`}
            onClick={() => setActiveTab("upload")}
          >
            <span className="tab-icon">📁</span>
            <span className="tab-label">File / Paste</span>
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === "camera" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("camera");
              startCamera();
            }}
          >
            <span className="tab-icon">📷</span>
            <span className="tab-label">Live Camera</span>
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === "url" ? "active" : ""}`}
            onClick={() => setActiveTab("url")}
          >
            <span className="tab-icon">🌐</span>
            <span className="tab-label">Image URL</span>
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === "samples" ? "active" : ""}`}
            onClick={() => setActiveTab("samples")}
          >
            <span className="tab-icon">✨</span>
            <span className="tab-label">Samples</span>
          </button>
        </div>
      )}

      {/* Mode 1: File Dropzone / Paste */}
      {!previewUrl && activeTab === "upload" && (
        <div
          className={`dropzone ${isDragActive ? "drag-active" : ""}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <span className="dropzone-icon">📁</span>
          <p className="dropzone-prompt">Drag & drop image here, or browse</p>
          <p className="dropzone-subtext">Supports PNG, JPG, WEBP, GIF, BMP (Max 10MB)</p>
          <div className="paste-badge">
            <span>💡 Tip: You can press <strong>Ctrl+V</strong> anywhere to paste an image</span>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="file-input-hidden"
            onChange={handleInputChange}
          />
        </div>
      )}

      {/* Mode 2: Live Camera Capture */}
      {!previewUrl && activeTab === "camera" && (
        <div className="camera-area">
          <video ref={videoRef} autoPlay playsInline muted className="camera-video" />
          <div className="camera-actions">
            {!isCameraActive ? (
              <button type="button" className="action-pill-btn" onClick={startCamera}>
                Start Camera
              </button>
            ) : (
              <button type="button" className="action-pill-btn capture-btn" onClick={captureFrame}>
                📸 Take Snapshot
              </button>
            )}
          </div>
        </div>
      )}

      {/* Mode 3: Image URL Input */}
      {!previewUrl && activeTab === "url" && (
        <form className="url-form" onSubmit={handleLoadUrl}>
          <input
            type="url"
            placeholder="Paste public image URL (https://...)"
            value={imageUrlInput}
            onChange={(e) => setImageUrlInput(e.target.value)}
            className="url-input"
            required
          />
          <button type="submit" className="url-btn" disabled={isLoading || !imageUrlInput.trim()}>
            Load Image
          </button>
        </form>
      )}

      {/* Mode 4: Samples */}
      {!previewUrl && activeTab === "samples" && (
        <div className="samples-grid">
          {SAMPLE_PRESETS.map((sample, idx) => (
            <div
              key={idx}
              className="sample-card"
              onClick={() => handleSelectSample(sample)}
            >
              <img src={sample.url} alt={sample.name} className="sample-thumb" />
              <div className="sample-info">
                <span className="sample-name">{sample.name}</span>
                <span className={`sample-tag ${sample.type === "Real" ? "tag-real" : "tag-ai"}`}>
                  {sample.type}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Selected Image Preview Area */}
      {previewUrl && (
        <div className="selected-preview-area">
          <div className="preview-container">
            <img src={previewUrl} alt="Selected preview" className="preview-image" />
          </div>
          <div className="file-info-bar">
            <span className="file-name">{selectedFile?.name || "Image ready"}</span>
            <button
              type="button"
              className="remove-file-btn"
              onClick={handleRemoveFile}
              disabled={isLoading}
            >
              Remove
            </button>
          </div>
        </div>
      )}

      {errorMessage && (
        <div className="error-banner">
          <span>⚠️</span>
          <span>{errorMessage}</span>
        </div>
      )}

      <div className="upload-actions">
        <button
          type="button"
          className="submit-btn"
          onClick={handleSubmit}
          disabled={!selectedFile || isLoading}
        >
          {isLoading ? (
            <>
              <span className="spinner" />
              <span>Analyzing Image...</span>
            </>
          ) : (
            <span>Detect Image</span>
          )}
        </button>
      </div>
    </div>
  );
}

export default ImageUpload;