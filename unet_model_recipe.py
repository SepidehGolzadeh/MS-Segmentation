"""
U-Net model architecture for 2D medical image segmentation.

This module defines:
- double_conv: two consecutive Conv-BN-ReLU blocks
- encoder_block: feature extraction + downsampling
- decoder_block: upsampling + skip-connection fusion
- unet_model: full U-Net (encoder + bridge + decoder)

Default output is binary segmentation with a sigmoid activation.
"""

from typing import Tuple
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, concatenate, Input,
    MaxPooling2D, BatchNormalization, ReLU
)


def double_conv(inputs, filters: int = 32):
    """
    Two consecutive convolution blocks used throughout U-Net.

    Structure:
        Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU

    Notes:
        - "same" padding preserves spatial size.
        - He initialization is suitable for ReLU activations.
        - use_bias=False is common when BatchNorm is used (BN has its own bias/shift).

    Args:
        inputs: Input feature map tensor.
        filters (int): Number of convolution filters.

    Returns:
        Tensor: Output feature map after two Conv-BN-ReLU blocks.
    """
    conv_settings = {
        "filters": filters,
        "kernel_size": 3,
        "padding": "same",
        "kernel_initializer": "he_normal",
        "use_bias": False,
    }

    x = Conv2D(**conv_settings)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(**conv_settings)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x


def encoder_block(inputs, filters: int = 32):
    """
    Encoder block: feature extraction + downsampling.

    Steps:
        1) Apply double_conv to extract features.
        2) Apply max pooling (2x2) to downsample spatial resolution.

    Args:
        inputs: Input tensor.
        filters (int): Number of convolution filters for this level.

    Returns:
        output: Downsampled tensor passed to the next encoder stage.
        skip: Feature tensor saved for skip connections in the decoder.
    """
    skip = double_conv(inputs, filters)
    output = MaxPooling2D(pool_size=(2, 2))(skip)
    return output, skip


def decoder_block(inputs, skip_layer_input, filters: int = 32):
    """
    Decoder block: upsampling + skip connection + feature refinement.

    Steps:
        1) Upsample using transposed convolution (2x2, stride=2).
        2) Concatenate with corresponding encoder feature map (skip connection).
        3) Refine merged features using double_conv.

    Args:
        inputs: Input tensor from previous decoder stage.
        skip_layer_input: Skip connection tensor from encoder.
        filters (int): Number of convolution filters for this level.

    Returns:
        Tensor: Output feature map after upsampling and refinement.
    """
    x = Conv2DTranspose(
        filters=filters,
        kernel_size=(2, 2),
        strides=2,
        padding="same",
    )(inputs)

    # Skip connection: fuse encoder features with decoder features
    x = concatenate([x, skip_layer_input], axis=3)

    return double_conv(x, filters)


def unet_model(input_size: Tuple[int, int, int] = (224, 224, 3), starting_filters: int = 32):
    """
    Build the full U-Net model.

    Architecture:
        Encoder (downsampling path) -> Bridge -> Decoder (upsampling path)
        Skip connections are used between corresponding encoder/decoder levels.

    Args:
        input_size (Tuple[int, int, int]): Input shape (H, W, C).
        starting_filters (int): Number of filters in the first encoder level.

    Returns:
        Model: Keras model instance.
    """
    inputs = Input(input_size)

    # -------------------------
    # Encoder (downsampling path)
    # -------------------------
    x1, s1 = encoder_block(inputs, starting_filters)
    x2, s2 = encoder_block(x1, starting_filters * 2)
    x3, s3 = encoder_block(x2, starting_filters * 4)
    x4, s4 = encoder_block(x3, starting_filters * 8)

    # -------------------------
    # Bridge (bottleneck)
    # -------------------------
    b = double_conv(x4, starting_filters * 16)

    # -------------------------
    # Decoder (upsampling path)
    # -------------------------
    d1 = decoder_block(b, s4, starting_filters * 8)
    d2 = decoder_block(d1, s3, starting_filters * 4)
    d3 = decoder_block(d2, s2, starting_filters * 2)
    d4 = decoder_block(d3, s1, starting_filters)

    # Output layer (binary segmentation)
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(d4)

    # For multi-class segmentation:
    # outputs = Conv2D(num_classes, (1, 1), activation="softmax")(d4)

    model = Model(inputs=inputs, outputs=outputs, name="U-Net")
    return model