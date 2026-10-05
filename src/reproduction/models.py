"""The six ImageNet backbones and the paper's shared classifier head."""

import tensorflow as tf

from .config import MODELS


@tf.keras.utils.register_keras_serializable(package="brain_mri")
class ImageNetScaling(tf.keras.layers.Layer):
    """Optional corrected scaling, explicitly different from paper [0,1]."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

    def call(self, inputs):
        modules = {
            "MobileNetV2": "mobilenet_v2",
            "Xception": "xception",
            "InceptionV3": "inception_v3",
            "ResNet50": "resnet50",
            "VGG16": "vgg16",
            "DenseNet121": "densenet",
        }
        preprocessing = getattr(tf.keras.applications, modules[self.model_name]).preprocess_input
        return preprocessing(inputs * 255.0)

    def get_config(self):
        return {**super().get_config(), "model_name": self.model_name}


def build_model(name: str, config: dict, weights: str | None = "imagenet") -> tf.keras.Model:
    if name not in MODELS:
        raise ValueError(f"Unsupported backbone: {name}")
    shape = (config["image_size"], config["image_size"], 3)
    inputs = tf.keras.Input(shape, name="mri_slice")
    x = (
        ImageNetScaling(name, name="imagenet_scaling")(inputs)
        if config["input_scaling"] == "imagenet"
        else inputs
    )
    backbone = getattr(tf.keras.applications, name)(
        include_top=False, weights=weights, input_shape=shape
    )
    backbone.trainable = config["trainable_backbone"]
    # Frozen backbones use inference-mode BatchNorm; fine-tuning follows fit's training flag.
    x = backbone(x) if backbone.trainable else backbone(x, training=False)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_30")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_20")(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax", name="classification")(x)
    return tf.keras.Model(inputs, outputs, name=name)
