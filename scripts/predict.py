import sys
import os
import tensorflow as tf

# Get project root directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.inference_utils import run_prediction

# Model path
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "ai_generated_image_detector.keras"
)

def main():
    # Check image path
    if len(sys.argv) < 2:
        print("ERROR: Please provide an image path")
        print("Usage: python scripts/predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at '{image_path}'")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'")
        sys.exit(1)

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Predict using shared inference utility (no manual / 255.0 double-normalization)
    result_data = run_prediction(model, image_path)

    print(f"RESULT: {result_data['label']}")
    print(f"CONFIDENCE: {result_data['confidence']:.2f}%")

if __name__ == "__main__":
    main()