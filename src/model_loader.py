import os
import json
import joblib
import tensorflow as tf
import sys

from src.attention_layer import AttentionLayer

def load_artifacts(base_path):

    # Load scaler
    scaler = joblib.load(os.path.join(base_path, "scaler.save"))

    # Load label encoder
    label_encoder = joblib.load(os.path.join(base_path, "label_encoder.save"))

    # Load metadata
    with open(os.path.join(base_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # Load model
    model = tf.keras.models.load_model(
        os.path.join(base_path, "best_model.keras"),
        custom_objects={"AttentionLayer": AttentionLayer}
    )

    return model, scaler, label_encoder, metadata