from __future__ import annotations

import logging
import os

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


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


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
