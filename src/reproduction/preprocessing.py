"""Paper-style foreground crop; this is not validated tumor segmentation."""

from pathlib import Path

import cv2
import numpy as np


def preprocess_image(path: Path | str, config: dict) -> tuple[np.ndarray, dict]:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Unreadable image: {path}")
    kernel = config["blur_kernel"]
    blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    _, mask = cv2.threshold(blurred, config["threshold"], 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fallback = not contours
    box = (0, 0, gray.shape[1], gray.shape[0])
    if contours:
        box = cv2.boundingRect(max(contours, key=cv2.contourArea))
    x, y, width, height = box
    crop = blurred[y : y + height, x : x + width]
    size = config["image_size"]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    image = np.repeat(resized[..., None], 3, axis=-1).astype(np.float32) / 255
    return image, {"crop_fallback": fallback, "bbox": list(map(int, box))}


def audit_preprocessing(records, root: Path, config: dict) -> dict:
    fallbacks = []
    for path in records.path:
        _, info = preprocess_image(Path(root) / path, config)
        if info["crop_fallback"]:
            fallbacks.append(path)
    return {
        "crop_fallback_count": len(fallbacks),
        "crop_fallback_paths": fallbacks,
        "blur_kernel": config["blur_kernel"],
        "threshold": config["threshold"],
    }


def augment_image(image, seed):
    import tensorflow as tf

    factors = tf.random.stateless_uniform((2,), seed, minval=0.8, maxval=1.2)
    image = image * factors[0]
    mean = tf.reduce_mean(image, axis=(0, 1), keepdims=True)
    return tf.clip_by_value((image - mean) * factors[1] + mean, 0.0, 1.0)


def make_dataset(manifest, config: dict, split: str, evaluation_view: str = "clean"):
    import tensorflow as tf

    if evaluation_view not in ("clean", "transformed"):
        raise ValueError("Unknown evaluation view")
    root = Path(config["data_dir"])
    paths = [str(root / path) for path in manifest.path]
    dataset = tf.data.Dataset.from_tensor_slices((paths, manifest.label.to_numpy(np.int32)))
    training = split == "train"
    if training:
        dataset = dataset.shuffle(len(manifest), seed=config["seed"], reshuffle_each_iteration=True)

    def decode(index, record):
        path, label = record

        def load(value):
            return preprocess_image(value.decode(), config)[0]

        image = tf.numpy_function(load, [path], tf.float32)
        image.set_shape((config["image_size"], config["image_size"], 3))
        if training or evaluation_view == "transformed":
            image = augment_image(image, (config["seed"], tf.cast(index, tf.int32)))
        return image, label

    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.deterministic = True
    return (
        dataset.enumerate()
        .map(decode, num_parallel_calls=2)
        .batch(config["batch_size"])
        .with_options(options)
        .prefetch(1)
    )
