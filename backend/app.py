import os
import sys
import io
from contextlib import asynccontextmanager

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf


# Ensure backend folder is available in sys.path
BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))

if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from scripts.inference_utils import run_prediction


# Model path inside backend/model/
MODEL_PATH = os.path.join(
    BACKEND_ROOT,
    "model",
    "ai_generated_image_detector.keras"
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
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
    lifespan=lifespan,
)


# CORS configuration
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")

if allowed_origins_env:
    origins = [
        origin.strip()
        for origin in allowed_origins_env.split(",")
        if origin.strip()
    ]
else:
    origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials="*" not in origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file,
    validates file size and image format,
    and returns classification label and confidence score.
    """

    global model

    # Load model if it is not already loaded
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded or model file not found.",
            )

    # Read uploaded file
    contents = await file.read()

    # Check empty file
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Check file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File size exceeds the 10MB limit "
                f"({len(contents)} bytes received)."
            ),
        )

    # Validate image
    try:
        pil_image = Image.open(io.BytesIO(contents))
        pil_image.verify()

        # Reopen after verify() and convert to RGB
        pil_image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image file: {str(e)}",
        )

    # Run prediction
    try:
        result = run_prediction(model, pil_image)

        return {
            "label": result["label"],
            "confidence": float(result["confidence"]),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )