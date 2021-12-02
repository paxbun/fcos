from typing import List, Optional
from dataclasses import dataclass, field

import tensorflow as tf


class ResNetTop(tf.keras.layers.Layer):
    def __init__(self, *args, **kwrags):
        super(ResNetTop, self).__init__(*args, **kwrags)
        self.initial_conv = tf.keras.layers.Conv2D(
            64, (7, 7), strides=(2, 2), padding="same", use_bias=False)
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.initial_conv(inputs)
        x = self.batch_norm(x)
        return x

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape


class ResNetBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        out_channels: int,
        downscale: bool,
        num_groups: Optional[int],
        group_width: Optional[int],
        *args,
        **kwargs
    ):
        super(ResNetBlock, self).__init__(*args, **kwargs)

        self.out_channels = out_channels
        self.downscale = downscale

        assert (num_groups == None) == (group_width == None)
        if num_groups == None:
            self.bottleneck_channels = self.out_channels // 4
        else:
            self.bottleneck_channels = num_groups * group_width

        self.main_path = [
            tf.keras.layers.Conv2D(
                self.bottleneck_channels,
                (1, 1),
                strides=((2, 2) if self.downscale else (1, 1)),
                padding="same",
                use_bias=False
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(
                self.bottleneck_channels,
                (3, 3),
                padding="same",
                use_bias=False,
                groups=num_groups,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(
                self.out_channels,
                (1, 1),
                padding="same",
                use_bias=False
            ),
            tf.keras.layers.BatchNormalization(),
        ]

        self.residual_path = []
        self.sum = tf.keras.layers.Add()

    def build(self, input_shape):
        if input_shape[-1] != self.out_channels:
            self.residual_path = [
                tf.keras.layers.Conv2D(
                    self.out_channels,
                    (1, 1),
                    strides=((2, 2) if self.downscale else (1, 1)),
                    padding="same",
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization()
            ]

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        if self.downscale:
            return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, self.out_channels)
        else:
            return (input_shape[:3], self.out_channels)

    def call(self, inputs):
        main = inputs
        for layer in self.main_path:
            main = layer(main)

        residual = inputs
        for layer in self.residual_path:
            residual = layer(residual)

        return self.sum([main, residual])


@dataclass
class ResNetStageConfig:
    out_channels: int = 0
    num_repeats: int = 0
    group_width: Optional[int] = None


@dataclass
class ResNetConfig:
    stages: List[ResNetStageConfig] = field(default_factory=list)
    num_groups: Optional[int] = None


class ResNetBase(tf.keras.layers.Layer):
    def __init__(self, config: ResNetConfig, *args, **kwargs):
        super(ResNetBase, self).__init__(*args, **kwargs)
        self.top = ResNetTop()
        self.stages = []
        for stage in config.stages:
            blocks = []
            for idx in range(stage.num_repeats):
                blocks.append(
                    ResNetBlock(
                        stage.out_channels,
                        idx == 0,
                        config.num_groups,
                        stage.group_width,
                    )
                )
            self.stages.append(blocks)

    def call(self, inputs):
        rtn = []
        out = self.top(inputs)
        for stage in self.stages:
            for block in stage:
                out = block(out)
            rtn.append(out)
        return rtn

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        rtn = []
        out = self.top.compute_output_shape(input_shape)
        for stage in self.stages:
            for block in stage:
                out = block.compute_output_shape(out)
            rtn.append(out)
        return rtn

class ResNetTest(ResNetBase):
    def __init__(self, *args, **kwargs):
        super(ResNetTest, self).__init__(
            ResNetConfig(stages=[
                ResNetStageConfig(256, 3),
                ResNetStageConfig(512, 3),
            ]),
            *args, **kwargs
        )



class ResNet50(ResNetBase):
    def __init__(self, *args, **kwargs):
        super(ResNet50, self).__init__(
            ResNetConfig(stages=[
                ResNetStageConfig(256, 3),
                ResNetStageConfig(512, 4),
                ResNetStageConfig(1024, 6),
                ResNetStageConfig(2048, 3)
            ]),
            *args, **kwargs
        )


class ResNet101(ResNetBase):
    def __init__(self, *args, **kwargs):
        super(ResNet101, self).__init__(
            ResNetConfig(stages=[
                ResNetStageConfig(256, 3),
                ResNetStageConfig(512, 4),
                ResNetStageConfig(1024, 6),
                ResNetStageConfig(2048, 3)
            ]),
            *args, **kwargs
        )


class ResNeXt32x8d(ResNetBase):
    def __init__(self, *args, **kwargs):
        super(ResNeXt32x8d, self).__init__(
            ResNetConfig(stages=[
                ResNetStageConfig(256, 3, 8),
                ResNetStageConfig(512, 4, 16),
                ResNetStageConfig(1024, 6, 32),
                ResNetStageConfig(2048, 3, 64)
            ], num_groups=32),
            *args, **kwargs
        )


class ResNeXt64x4d(ResNetBase):
    def __init__(self, *args, **kwargs):
        super(ResNeXt32x8d, self).__init__(
            ResNetConfig(stages=[
                ResNetStageConfig(256, 3, 4),
                ResNetStageConfig(512, 4, 8),
                ResNetStageConfig(1024, 6, 16),
                ResNetStageConfig(2048, 3, 32)
            ], num_groups=64),
            *args, **kwargs
        )
