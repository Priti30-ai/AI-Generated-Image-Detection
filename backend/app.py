import os
import sys
import io
from contextlib import asynccontextmanager
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

# Ensure project root is available in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.inference_utils import run_prediction

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "ai_generated_image_detector.keras")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading Keras model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"[WARNING] Model file not found at {MODEL_PATH}.")
    yield
    print("Shutting down backend...")


app = FastAPI(
    title="AI Generated Image Detection API",
    description="FastAPI service for classifying images as REAL or AI GENERATED",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local development and cloud production deployment
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "*"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in origins else origins,
    allow_credentials=True if "*" not in origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file (multipart/form-data),
    validates file size (under 10MB) and image format,
    and returns classification label and confidence score.
    """
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded or model file not found."
            )

    # Read uploaded file contents
    contents = await file.read()

    # Validate file size (under 10MB)
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds maximum allowed limit of 10MB ({len(contents)} bytes received)."
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    # Validate image format
    try:
        pil_image = Image.open(io.BytesIO(contents))
        pil_image.verify()  # Validate image integrity
        # Re-open after verify (verify closes file stream in PIL)
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image file: {str(e)}"
        )

    # Execute inference via shared utility
    try:
        prediction_result = run_prediction(model, pil_image)
        return {
            "label": prediction_result["label"],
            "confidence": float(prediction_result["confidence"])
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("backend.app:app", host=host, port=port, reload=True)
