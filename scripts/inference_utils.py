import os
import io
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


def get_model_input_size(model, default_size=(128, 128)):
    """Extract (height, width) target size from model input shape."""
    try:
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if input_shape and len(input_shape) == 4 and input_shape[1] is not None and input_shape[2] is not None:
            return (int(input_shape[1]), int(input_shape[2]))
    except Exception:
        pass
    return default_size


def format_pil_image_to_rgb(pil_img):
    """
    Normalizes any PIL image format (RGBA, CMYK, Palette, Grayscale) to clean RGB.
    Handles EXIF orientation for smartphone photos and transparency compositing.
    """
    # 1. Correct orientation based on EXIF tag (critical for real phone camera photos)
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass

    # 2. Handle transparency (RGBA / LA / P with alpha)
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        pil_img = pil_img.convert("RGBA")
        background = Image.new("RGB", pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        return background

    # 3. Standard RGB conversion (handles CMYK, L, P, 1, etc.)
    return pil_img.convert("RGB")


def preprocess_image(image_input, target_size=(128, 128)):
    """
    Preprocesses any real-time image (path, bytes, PIL Image) for model inference.
    Handles any aspect ratio, orientation, and color space.
    Note: NO manual / 255.0 division, as the Rescaling layer is embedded in the model.
    """
    if isinstance(image_input, (str, os.PathLike)):
        pil_img = Image.open(image_input)
    elif isinstance(image_input, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        # Fallback for file-like streams
        pil_img = Image.open(image_input)

    pil_img = format_pil_image_to_rgb(pil_img)
    pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    image_array = np.array(pil_img, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def run_prediction(model, image_input):
    """
    Runs model inference and returns the classification label and confidence.
    Output >= 0.5 means REAL, < 0.5 means AI GENERATED.
    """
    target_size = get_model_input_size(model)
    image_array = preprocess_image(image_input, target_size=target_size)
    prediction = float(model.predict(image_array, verbose=0)[0][0])

    if prediction >= 0.5:
        label = "REAL"
        confidence = float(prediction * 100)
    else:
        label = "AI_GENERATED"
        confidence = float((1.0 - prediction) * 100)

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "raw_score": prediction
    }
