import sys
import os
import tensorflow as tf
import numpy as np


# Get project root directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# Model path
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "ai_generated_image_detector.keras"
)


# Load model
model = tf.keras.models.load_model(MODEL_PATH)


# Check image path
if len(sys.argv) < 2:
    print("ERROR: Please provide an image path")
    sys.exit(1)


image_path = sys.argv[1]


# Load image
image = tf.keras.utils.load_img(
    image_path,
    target_size=(32, 32)
)


# Convert image to array
image_array = tf.keras.utils.img_to_array(image)


# Normalize
image_array = image_array / 255.0


# Add batch dimension
image_array = np.expand_dims(
    image_array,
    axis=0
)


# Predict
prediction = model.predict(
    image_array,
    verbose=0
)[0][0]


# Result
if prediction >= 0.5:

    result = "REAL"
    confidence = prediction * 100

else:

    result = "AI GENERATED"
    confidence = (1 - prediction) * 100


print(f"RESULT: {result}")
print(f"CONFIDENCE: {confidence:.2f}%")