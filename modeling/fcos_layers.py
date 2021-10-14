from constants import NUM_CLASSES, OUTPUT_SHAPES

import tensorflow as tf
import tensorflow_addons as tfa


class TrainableBaseExponential(tf.keras.layers.Layer):
    def __init__(self, *args, **kwrags):
        super(TrainableBaseExponential, self).__init__(*args, **kwrags)

    def build(self, input_shape):
        assert type(input_shape) == list
        self.num_stages = len(input_shape)
        self.scalar_list = [self.add_weight(
            f"scalar{i}") for i in range(1, self.num_stages + 1)]

    def call(self, inputs):
        return [tf.exp(scalar * input) for scalar, input in zip(self.scalar_list, inputs)]


class FCOSHead(tf.keras.layers.Layer):
    def __init__(
        self,
        centerness_at_reg_branch: bool = False,
        *args,
        **kwargs
    ):
        super(FCOSHead, self).__init__(*args, **kwargs)
        self.centerness_at_reg_branch = centerness_at_reg_branch

        self.conv_list = [
            tf.keras.layers.Conv2D(
                512, (3, 3), padding="same", activation="relu", use_bias=True, groups=2)
            for _ in range(4)
        ]

        # number of classes + negative sample class (1)
        self.class_conv = tf.keras.layers.Conv2D(
            NUM_CLASSES + 1, (3, 3), padding="same", activation="sigmoid")
        self.reg_conv = tf.keras.layers.Conv2D(
            4, (3, 3), padding="same", activation=None)
        self.centerness_conv = tf.keras.layers.Conv2D(
            1, (3, 3), padding="same", activation="sigmoid")
        self.reg_conv_trainable_exp = TrainableBaseExponential()

        self.reshapers = [
            tf.keras.layers.Reshape((height * width, NUM_CLASSES + 6))
            for height, width in OUTPUT_SHAPES
        ]

    def call(self, inputs):
        class_out_list = []
        centerness_out_list = []
        reg_out_list = []

        for x in inputs:
            for conv in self.conv_list:
                x = conv(x)
            class_x, reg_x = tf.split(x, num_or_size_splits=2, axis=-1)

            class_out_list.append(self.class_conv(class_x))
            reg_out_list.append(self.reg_conv(reg_x))

            if self.centerness_at_reg_branch:
                centerness_out_list.append(self.centerness_conv(reg_x))
            else:
                centerness_out_list.append(self.centerness_conv(class_x))

        reg_out_list = self.reg_conv_trainable_exp(reg_out_list)

        rtn = []
        for class_out, centerness_out, reg_out, reshaper in zip(class_out_list, centerness_out_list, reg_out_list, self.reshapers):
            output = tf.keras.layers.concatenate(
                [class_out, centerness_out, reg_out],
                axis=-1,
            )
            output = reshaper(output)
            rtn.append(output)

        rtn = tf.keras.layers.concatenate(rtn, axis=1)
        return rtn


class FeaturePyramid(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(FeaturePyramid, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        assert type(input_shape) == list
        assert len(input_shape) > 0

        self.num_input_stages = len(input_shape)

        self.lateral_conv_list = [
            tf.keras.layers.Conv2D(
                256, (1, 1), padding="same", use_bias=False, activation=tfa.layers.GroupNormalization(32)
            )
            for _ in range(self.num_input_stages)
        ]
        self.interpolators = [
            tf.keras.layers.experimental.preprocessing.Resizing(
                shape[1], shape[2], interpolation="nearest")
            for shape in input_shape[:-1]
        ]
        self.mergers = [
            tf.keras.layers.Add()
            for _ in range(self.num_input_stages - 1)
        ]
        self.final_conv_list = [
            tf.keras.layers.Conv2D(
                256, (3, 3), padding="same", use_bias=False, activation=tfa.layers.GroupNormalization(32)
            )
            for _ in range(self.num_input_stages)
        ]

        self.p6 = tf.keras.layers.Conv2D(
            256, (3, 3), strides=(2, 2), padding="same")
        self.p7_relu = tf.keras.layers.ReLU()
        self.p7 = tf.keras.layers.Conv2D(
            256, (3, 3), strides=(2, 2), padding="same")

    def call(self, inputs):
        inputs = [conv(input)
                  for conv, input in zip(self.lateral_conv_list, inputs)]
        for i in range(self.num_input_stages - 2, -1, -1):
            previous = inputs[i + 1]
            previous = self.interpolators[i](previous)
            inputs[i] = self.mergers[i]([previous, inputs[i]])

        inputs = [conv(input)
                  for conv, input in zip(self.final_conv_list, inputs)]
        p6 = self.p6(inputs[-1])
        p7 = self.p7_relu(p6)
        p7 = self.p7(p7)
        inputs.extend((p6, p7))

        return inputs
