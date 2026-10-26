from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from cnnClassifier.pipeline.prediction import InvalidImageError, PredictionPipeline


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MiB per uploaded image
CORS(app)

classifier = PredictionPipeline(
    model_path=os.getenv("MODEL_PATH", "artifacts/training/model.h5")
)
training_lock = threading.Lock()


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/train")
def train_route():
    """Run the project's full training and evaluation pipeline."""
    if not training_lock.acquire(blocking=False):
        return jsonify({"error": "Training is already in progress."}), 409

    try:
        project_root = Path(__file__).resolve().parent
        process = subprocess.run(
            [sys.executable, "main.py"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            app.logger.error(
                "Training failed (exit %s):\n%s",
                process.returncode,
                (process.stderr or process.stdout)[-4000:],
            )
            return jsonify({"error": "Training failed. Check the server logs for details."}), 500

        return jsonify({"message": "Training and evaluation completed successfully."})
    except OSError:
        app.logger.exception("Could not start the training process")
        return jsonify({"error": "Could not start the training process."}), 500
    finally:
        training_lock.release()


@app.post("/predict")
def predict_route():
    try:
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            image_input = payload.get("image")
        else:
            uploaded_file = request.files.get("image")
            image_input = uploaded_file.read() if uploaded_file else None

        if not image_input:
            return jsonify({"error": "Choose an image before requesting a prediction."}), 400

        result = classifier.predict(image_input)
        return jsonify(result)
    except InvalidImageError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError:
        app.logger.exception("Prediction model file is missing")
        return jsonify({"error": "The prediction model is not available on the server."}), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Please try another image."}), 500


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    return jsonify({"error": "Image is too large. Upload an image smaller than 12 MiB."}), 413


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
