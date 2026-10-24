"""Model loading, preprocessing, prediction, vegetation verification and Grad-CAM.

The saved model already contains a Rescaling layer (x/127.5 - 1), so inputs are RAW 0-255
float32 pixels. Do not apply MobileNetV2 preprocess_input again.
"""
import io
import json
import logging
import os
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import settings

log = logging.getLogger("corn_leaf.ml")

IMG_SIZE = 224
DISPLAY_SIZE = 448  # overlay is rendered larger than the model input for a sharper UI

NON_MAIZE_CLASSES = {"Not_Leaf", "Not_Corn", "Non_Corn", "Non_Maize", "Other", "Background"}


class InvalidImageError(ValueError):
    pass


class OodGate:
    """Mahalanobis distance of the pooled 1280-d feature to the nearest corn-class mean.

    Fitted during training on corn leaf images only. Large distance = unfamiliar image.
    """

    def __init__(self, path) -> None:
        d = np.load(path)
        self.means = d["means"].astype(np.float64)  # (K, D)
        self.precision = d["precision"].astype(np.float64)  # (D, D)
        self.threshold = float(d["threshold"])

    def distance(self, feat: np.ndarray) -> float:
        diff = self.means - feat.astype(np.float64)[None]
        return float(np.min(np.einsum("kd,de,ke->k", diff, self.precision, diff)))


@dataclass
class Result:
    class_name: str
    confidence: float
    probabilities: dict[str, float]
    heatmap_png: bytes
    image_png: bytes
    ood_distance: float | None = None
    vegetation_score: float = 1.0
    is_plant_like: bool = True


def analyze_foliage_characteristics(rgb: np.ndarray) -> tuple[bool, float]:
    """Inspect color spectra and texture to verify whether the image contains plant foliage.

    Returns (is_plant_like, vegetation_score).
    Detects solid blank colors, non-plant artificial patterns, or total absence of leaf pigments.
    """
    # 1. Check for blank / solid color / extreme low-variance image
    std_per_channel = rgb.std(axis=(0, 1))
    overall_std = float(std_per_channel.mean())
    if overall_std < 7.5:
        return False, 0.0

    # 2. Check vegetation color distribution in HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Green foliage mask (OpenCV H is 0-180; 22-88 corresponds to yellow-green, green, dark green)
    green_mask = (h >= 22) & (h <= 90) & (s >= 25) & (v >= 20)

    # Lesion / necrosis foliage mask (yellowish-tan, rust-brown typical of blights & rusts)
    lesion_mask = (h >= 8) & (h < 22) & (s >= 35) & (v >= 25)

    plant_pixels = int(np.count_nonzero(green_mask | lesion_mask))
    total_pixels = rgb.shape[0] * rgb.shape[1]
    veg_ratio = float(plant_pixels / total_pixels)

    # 3. Excess Green index (ExG = 2*G - R - B)
    r_ch, g_ch, b_ch = rgb[:, :, 0].astype(float), rgb[:, :, 1].astype(float), rgb[:, :, 2].astype(float)
    exg = 2.0 * g_ch - r_ch - b_ch
    exg_ratio = float(np.count_nonzero(exg > 5.0) / total_pixels)

    combined_score = max(veg_ratio, exg_ratio * 0.8)

    # If the image has virtually zero leaf-like foliage pixels (< 3.5%) and low variation, reject
    if combined_score < 0.035 and overall_std < 18.0:
        return False, combined_score
    if combined_score < 0.015:
        return False, combined_score

    return True, combined_score


class Classifier:
    def __init__(self) -> None:
        self.class_names: list[str] = json.loads(settings.class_names_path.read_text())
        self.model = keras.models.load_model(settings.model_path, compile=False)
        self.rescale = self.model.get_layer("rescaling")
        self.backbone = self.model.get_layer("mobilenetv2_1.00_224")
        self.gap = self.model.get_layer("global_average_pooling2d")
        self.dropout = self.model.get_layer("dropout")
        self.dense = self.model.get_layer("dense")
        if self.dense.units != len(self.class_names):
            raise RuntimeError(
                f"class_names.json length ({len(self.class_names)}) does not match "
                f"model output units ({self.dense.units})"
            )
        self.ood: OodGate | None = None
        if settings.ood_stats_path.exists():
            try:
                self.ood = OodGate(settings.ood_stats_path)
                if self.ood.means.shape[1] != self.gap.output.shape[-1]:
                    log.warning("ood_stats.npz feature size mismatch; ignoring OOD gate")
                    self.ood = None
                else:
                    log.info("OOD Gate initialized successfully from %s", settings.ood_stats_path)
            except Exception as e:
                log.warning("Failed to load ood_stats.npz: %s", e)
                self.ood = None

        self.predict_rgb(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))  # warm-up

    def reject_reason(self, r: Result) -> str | None:
        """Determine if and why this image is not recognized as a maize leaf."""
        # 1. Model explicitly classified into negative class
        if r.class_name in NON_MAIZE_CLASSES:
            return "model_not_leaf"

        # 2. Plant foliage spectrum verification failed (non-plant, blank, or synthetic image)
        if not r.is_plant_like:
            return "non_plant_image"

        # 3. Feature-space Mahalanobis OOD distance gate
        if self.ood is not None and r.ood_distance is not None and r.ood_distance > self.ood.threshold * settings.ood_scale:
            return "unfamiliar_image"

        # 4. Extreme low confidence threshold
        if r.confidence < settings.reject_below_confidence:
            return "low_confidence"

        return None

    @staticmethod
    def load_image(data: bytes) -> tuple[np.ndarray, np.ndarray]:
        """Returns (model_input_224, display_448), both RGB uint8 resized from the same source."""
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise InvalidImageError("File is not a valid image")
        img = ImageOps.exif_transpose(img).convert("RGB")
        if min(img.size) < 32:
            raise InvalidImageError("Image is too small (minimum 32 px per side)")
        small = np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR), dtype=np.uint8)
        big = np.asarray(img.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.BICUBIC), dtype=np.uint8)
        return small, big

    def predict_rgb(self, rgb: np.ndarray, display: np.ndarray | None = None) -> Result:
        display = rgb if display is None else display
        is_plant, veg_score = analyze_foliage_characteristics(rgb)

        x = tf.constant(rgb[None].astype("float32"))
        with tf.GradientTape() as tape:
            conv = self.backbone(self.rescale(x), training=False)
            tape.watch(conv)
            pooled = self.gap(conv)
            probs = self.dense(self.dropout(pooled, training=False))
            idx = int(tf.argmax(probs[0]))
            score = probs[:, idx]
        grads = tape.gradient(score, conv)
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))[0].numpy()
        cam = cam / cam.max() if cam.max() > 0 else cam
        cam = cv2.resize(cam, display.shape[1::-1], interpolation=cv2.INTER_CUBIC).clip(0, 1)
        heat = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)  # BGR
        overlay = cv2.addWeighted(cv2.cvtColor(display, cv2.COLOR_RGB2BGR), 0.55, heat, 0.45, 0)

        p = probs[0].numpy()
        dist = self.ood.distance(pooled[0].numpy()) if self.ood is not None else None
        return Result(
            ood_distance=dist,
            vegetation_score=veg_score,
            is_plant_like=is_plant,
            class_name=self.class_names[idx],
            confidence=float(p[idx]),
            probabilities={n: float(v) for n, v in zip(self.class_names, p)},
            heatmap_png=cv2.imencode(".png", overlay)[1].tobytes(),
            image_png=cv2.imencode(".jpg", cv2.cvtColor(display, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(),
        )

    def predict_bytes(self, data: bytes) -> Result:
        small, big = self.load_image(data)
        return self.predict_rgb(small, big)
