from __future__ import annotations

import base64
import binascii
import io
import threading
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow.keras.models import load_model


# ImageDataGenerator.flow_from_directory() assigns class indices in alphabetical order.
CLASS_NAMES = ("Cyst", "Normal", "Stone", "Tumor")


class InvalidImageError(ValueError):
    """Raised when an uploaded payload cannot be decoded as an image."""


class PredictionPipeline:
    def __init__(
        self,
        model_path: str | Path = "artifacts/training/model.h5",
        image_size: tuple[int, int] = (224, 224),
        class_names: tuple[str, ...] = CLASS_NAMES,
    ):
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.class_names = class_names
        self._model = None
        self._model_lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    if not self.model_path.is_file():
                        raise FileNotFoundError(f"Model not found: {self.model_path}")
                    self._model = load_model(self.model_path, compile=False)
                    output_count = int(self._model.output_shape[-1])
                    if output_count != len(self.class_names):
                        raise ValueError(
                            f"Model returns {output_count} classes, but {len(self.class_names)} labels are configured."
                        )
        return self._model

    @staticmethod
    def _read_payload(image_input: bytes | bytearray | str | Path | BinaryIO) -> bytes:
        if isinstance(image_input, (bytes, bytearray)):
            return bytes(image_input)
        if hasattr(image_input, "read"):
            return image_input.read()
        if isinstance(image_input, Path):
            return image_input.read_bytes()
        if isinstance(image_input, str):
            encoded = image_input.split(",", 1)[1] if image_input.startswith("data:") and "," in image_input else image_input
            try:
                return base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise InvalidImageError("The image data is not valid base64.") from exc
        raise InvalidImageError("Provide an uploaded image file.")

    def _prepare_image(self, image_input) -> np.ndarray:
        payload = self._read_payload(image_input)
        if not payload:
            raise InvalidImageError("The uploaded image is empty.")
        try:
            with Image.open(io.BytesIO(payload)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                resampling = getattr(Image, "Resampling", Image).BILINEAR
                image = image.resize((self.image_size[1], self.image_size[0]), resampling)
                array = np.asarray(image, dtype=np.float32)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
            raise InvalidImageError("That file is not a supported, readable image.") from exc

        # Match the rescale=1./255 preprocessing used by the training generator.
        return np.expand_dims(array / 255.0, axis=0)

    def predict(self, image_input) -> dict:
        batch = self._prepare_image(image_input)
        model = self._get_model()
        scores = np.asarray(model.predict(batch, verbose=0), dtype=np.float64)
        if scores.ndim != 2 or scores.shape[0] != 1 or scores.shape[1] != len(self.class_names):
            raise ValueError(f"Unexpected model output shape: {scores.shape}")
        scores = scores[0]
        if not np.all(np.isfinite(scores)):
            raise ValueError("The model returned invalid prediction scores.")

        best_index = int(np.argmax(scores))
        ranked_scores = sorted(
            (
                {"class_name": name, "score": round(float(score), 6)}
                for name, score in zip(self.class_names, scores)
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        return {
            "prediction": self.class_names[best_index],
            "confidence": round(float(scores[best_index]), 6),
            "scores": ranked_scores,
        }
