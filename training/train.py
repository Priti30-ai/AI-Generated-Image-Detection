import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

IMAGE_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)
CLASS_NAMES = ["ai", "real"]  # ai: 0, real: 1


def build_model(input_shape=INPUT_SHAPE):
    """
    Builds CNN architecture with internal data augmentation and rescaling.
    Binary classification:
      - 0: AI GENERATED
      - 1: REAL (sigmoid output >= 0.5)
    """
    model = tf.keras.Sequential([
        # Input Layer
        tf.keras.layers.Input(shape=input_shape),

        # Augmentation Block (active only during training)
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),

        # Rescaling layer embedded inside model
        tf.keras.layers.Rescaling(1.0 / 255.0),

        # Block 1 - 32 filters
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Block 2 - 64 filters
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Block 3 - 128 filters
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Block 4 - 256 filters (added for 128x128 resolution)
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Classification Head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    return model


def load_datasets(data_dir, image_size=IMAGE_SIZE, batch_size=32):
    """
    Loads train and val datasets from training/data/{train,val}/{real,ai}/
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Dataset directories not found. Expected '{train_dir}' and '{val_dir}'. "
            f"Please populate 'training/data/train/{{ai,real}}' and 'training/data/val/{{ai,real}}'."
        )

    print(f"Loading training data from {train_dir}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=42
    )

    print(f"Loading validation data from {val_dir}...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    # Performance optimization: prefetch & cache
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds


def train(
    data_dir,
    output_model_path,
    metrics_path,
    epochs=25,
    batch_size=32,
    learning_rate=1e-3
):
    """
    Executes model training, callbacks, evaluation, and saves model and metrics.json.
    """
    train_ds, val_ds = load_datasets(data_dir, image_size=IMAGE_SIZE, batch_size=batch_size)

    model = build_model(input_shape=INPUT_SHAPE)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save final model explicitly
    model.save(output_model_path)
    print(f"Model saved successfully to '{output_model_path}'")

    # Evaluate and compute confusion matrix
    print("\nEvaluating model on validation dataset...")
    y_true = []
    y_pred_probs = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "confusion_matrix": {
            "matrix": cm,
            "labels": {
                "0": "ai (AI GENERATED)",
                "1": "real (REAL)"
            },
            "description": "Rows: True label [ai, real], Columns: Predicted label [ai, real]"
        },
        "epochs_trained": len(history.history["loss"]),
        "final_train_loss": round(float(history.history["loss"][-1]), 4),
        "final_val_loss": round(float(history.history["val_loss"][-1]), 4)
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nTraining Metrics saved to '{metrics_path}':")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train AI vs Real Image Detection CNN Model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "training", "data"),
        help="Path to dataset containing train/ and val/ folders"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "model", "ai_generated_image_detector.keras"),
        help="Path to save trained .keras model"
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "training", "metrics.json"),
        help="Path to save metrics JSON"
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    try:
        train(
            data_dir=args.data_dir,
            output_model_path=args.model_path,
            metrics_path=args.metrics_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease follow the instructions in 'training/README.md' to set up your dataset.")
        sys.exit(1)


if __name__ == "__main__":
    main()
