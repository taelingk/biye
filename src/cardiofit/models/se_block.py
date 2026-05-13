"""SE (Squeeze-and-Excitation) Block and ResidualSEBlock — ported from Dong Xue's paper.

Reference: ~/code/codongxue/src/training/train_svco_model.py (lines 354-434)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    GlobalAveragePooling1D,
    Layer,
    Multiply,
    Reshape,
)


@tf.keras.utils.register_keras_serializable(package="cardiofit")
class SEBlock(Layer):
    """Squeeze-and-Excitation block for channel-wise feature recalibration.

    GlobalAveragePooling → Dense(C//r, relu) → Dense(C, sigmoid).
    """

    def __init__(self, reduction_ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.squeeze = GlobalAveragePooling1D()
        self.excitation1 = Dense(channels // self.reduction_ratio, activation="relu")
        self.excitation2 = Dense(channels, activation="sigmoid")
        super().build(input_shape)

    def call(self, inputs):
        x = self.squeeze(inputs)
        x = self.excitation1(x)
        x = self.excitation2(x)
        x = Reshape((1, -1))(x)
        return Multiply()([inputs, x])

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="cardiofit")
class ResidualSEBlock(Layer):
    """ResNet-style residual block with SE attention.

    Conv1D(3,s) → BN → ReLU → Conv1D(3,1) → BN → SE → Add → ReLU.
    Optionally uses 1×1 conv shortcut for dimension matching.
    """

    def __init__(
        self,
        filters: int,
        stride: int = 1,
        use_1x1_conv: bool = False,
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self._use_1x1_conv = use_1x1_conv
        self.kernel_regularizer = kernel_regularizer

        self.conv1 = Conv1D(
            filters,
            3,
            padding="same",
            strides=stride,
            kernel_regularizer=kernel_regularizer,
        )
        self.bn1 = BatchNormalization()
        self.act1 = Activation("relu")
        self.conv2 = Conv1D(
            filters, 3, padding="same", kernel_regularizer=kernel_regularizer
        )
        self.bn2 = BatchNormalization()
        self.se = SEBlock()

        self.shortcut_conv = None
        self.shortcut_bn = None
        if use_1x1_conv:
            self.shortcut_conv = Conv1D(
                filters, 1, strides=stride, kernel_regularizer=kernel_regularizer
            )
            self.shortcut_bn = BatchNormalization()
        self.act2 = Activation("relu")

    def call(self, inputs):
        x = self.act1(self.bn1(self.conv1(inputs)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_bn(self.shortcut_conv(inputs))
        else:
            shortcut = inputs
        return self.act2(x + shortcut)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "stride": self.stride,
                "use_1x1_conv": self._use_1x1_conv,
                "kernel_regularizer": (
                    tf.keras.regularizers.serialize(self.kernel_regularizer)
                    if self.kernel_regularizer
                    else None
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        kernel_regularizer_config = config.pop("kernel_regularizer", None)
        if kernel_regularizer_config:
            config["kernel_regularizer"] = tf.keras.regularizers.deserialize(
                kernel_regularizer_config
            )
        return cls(**config)
