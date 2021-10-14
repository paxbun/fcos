from constants import *
from typing import Tuple

import tensorflow as tf


class FCOSPreprocessor:
    def __init__(self, centerness_threshold: float = 0.0):
        threshold = centerness_threshold
        self.threshold_sq = threshold * threshold

    def __call__(self, raw_feature: dict) -> Tuple[tf.Tensor, tf.Tensor]:
        input = tf.image.resize(raw_feature["image"], IMAGE_SHAPE)

        objects = raw_feature["objects"]
        input_height, input_width = IMAGE_SHAPE
        input_area = input_height * input_width
        epsilon = 1e-8

        # Append a negative sample for fallback logic
        bbox = objects["bbox"]
        bbox = tf.concat([
            bbox,
            [[0, 0, 1, 1]]
        ], axis=0)
        bbox = tf.reshape(bbox, (1, 1, -1, 4))

        label = objects["label"] + 1
        label = tf.concat([label, [0]], axis=0)
        label = tf.reshape(label, (1, 1, -1))

        area = objects["area"]
        area = tf.concat([area, [input_area]], axis=0)
        area = tf.reshape(area, (1, 1, -1))
        area = tf.cast(area / input_area, tf.float32)

        outputs = []
        for (y_coords, x_coords), (height, width), (minvalid_dist, maxvalid_dist) in zip(GRIDS, OUTPUT_SHAPES, MAX_DISTS):
            y_coords = tf.reshape(y_coords, (height, width, 1))
            x_coords = tf.reshape(x_coords, (height, width, 1))

            # (height, width, num_objects + 1)
            l, r = x_coords - bbox[:, :, :,
                                   LEFT], bbox[:, :, :, RIGHT] - x_coords
            t, b = y_coords - bbox[:, :, :,
                                   TOP], bbox[:, :, :, BOTTOM] - y_coords
            min_dist = tf.minimum(tf.minimum(
                l, r) * input_width, tf.minimum(t, b) * input_height)
            max_dist = tf.maximum(tf.maximum(
                l, r) * input_width, tf.maximum(t, b) * input_height)
            centerness_sq = (tf.minimum(l, r) / (tf.maximum(l, r) + epsilon)
                             * tf.minimum(t, b) / (tf.maximum(t, b) + epsilon))

            # (height, width, num_objects + 1)
            valid = minvalid_dist < max_dist
            valid = tf.logical_and(valid, max_dist <= maxvalid_dist)
            # true if the bounding box contains the pixel
            valid = tf.logical_and(valid, 0 < min_dist)
            # true if the centerness is greater than the threshold
            valid = tf.logical_and(valid, self.threshold_sq <= centerness_sq)

            # (height, width, num_objects + 1)
            # Make the areas of the invalid points incredibly large
            _area = area * (tf.cast(tf.logical_not(valid),
                                    tf.float32) * 1000000 + 1)

            idx = tf.argmin(_area, axis=-1)

            # (height, width, num_objects + 1)
            _class = label * tf.cast(valid, tf.int64)

            # (height, width, NUM_CLASSES + 1, num_objects + 1)
            _class = tf.one_hot(_class, depth=(NUM_CLASSES + 1), axis=2)

            # Multiply centerness_sq by valid to prevent centerness from being NaN
            valid = tf.cast(valid, tf.float32)
            centerness_sq = centerness_sq * valid
            centerness = tf.sqrt(centerness_sq)

            # (height, width, 5, num_objects + 1)
            valid = tf.reshape(valid, (height, width, 1, -1))
            centerness_and_reg = tf.stack([centerness, t, l, b, r], axis=-2)
            centerness_and_reg = centerness_and_reg * valid

            # (height, width, NUM_CLASSES + 6, num_objects + 1)
            output = tf.concat([_class, centerness_and_reg], axis=-2)

            # (height, width, NUM_CLASSES + 6)
            output = tf.gather(output, idx, axis=3, batch_dims=2)

            # (height * width, NUM_CLASSES + 6)
            output = tf.reshape(output, (height * width, NUM_CLASSES + 6))

            outputs.append(output)

        # (sum of height * width, NUM_CLASSES + 6)
        outputs = tf.concat(outputs, axis=0)
        return (input / 255.0, outputs)
