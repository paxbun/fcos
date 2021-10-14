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
        image_area = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
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
        area = tf.concat([area, [image_area]], axis=0)
        area = tf.reshape(area, (1, 1, -1))
        area = tf.cast(area / image_area, tf.float32)

        outputs = []
        for (x_coords, y_coords), (width, height), (minvalid_dist, maxvalid_dist) in zip(GRIDS, OUTPUT_SHAPES, MAX_DISTS):
            x_coords = tf.reshape(x_coords, (width, height, 1))
            y_coords = tf.reshape(y_coords, (width, height, 1))

            # (width, height, num_objects + 1)
            l, r = x_coords - bbox[:, :, :, LEFT], bbox[:, :, :, RIGHT] - x_coords
            t, b = y_coords - bbox[:, :, :, TOP], bbox[:, :, :, BOTTOM] - y_coords
            min_dist = tf.minimum(tf.minimum(l, r), tf.minimum(t, b))
            max_dist = tf.maximum(tf.maximum(l, r), tf.maximum(t, b))
            centerness_sq = (tf.minimum(l, r) / (tf.maximum(l, r) + epsilon)
                             * tf.minimum(t, b) / (tf.maximum(t, b) + epsilon))

            # (width, height, num_objects + 1)
            valid = minvalid_dist < max_dist
            valid = tf.logical_and(valid, max_dist <= maxvalid_dist)
            # true if the bounding box contains the pixel
            valid = tf.logical_and(valid, 0 < min_dist)
            # true if the centerness is greater than the threshold
            valid = tf.logical_and(valid, self.threshold_sq <= centerness_sq)

            # (width, height, num_objects + 1)
            # Make the areas of the invalid points incredibly large
            _area = area * (tf.cast(tf.logical_not(valid),
                                    tf.float32) * 1000000 + 1)

            idx = tf.argmin(_area, axis=-1)

            # (width, height, num_objects + 1)
            _class = label * tf.cast(valid, tf.int64)

            # (width, height, NUM_CLASSES + 1, num_objects + 1)
            _class = tf.one_hot(_class, depth=(NUM_CLASSES + 1), axis=2)

            # Multiply centerness_sq by valid to prevent centerness from being NaN
            valid = tf.cast(valid, tf.float32)
            centerness_sq = centerness_sq * valid
            centerness = tf.sqrt(centerness_sq)

            # (width, height, 5, num_objects + 1)
            valid = tf.reshape(valid, (width, height, 1, -1))
            centerness_and_reg = tf.stack([centerness, l, r, t, b], axis=-2)
            centerness_and_reg = centerness_and_reg * valid

            # (width, height, NUM_CLASSES + 6, num_objects + 1)
            output = tf.concat([_class, centerness_and_reg], axis=-2)

            # (width, height, NUM_CLASSES + 6)
            output = tf.gather(output, idx, axis=3, batch_dims=2)

            # (width * height, NUM_CLASSES + 6)
            output = tf.reshape(output, (width * height, NUM_CLASSES + 6))

            outputs.append(output)

        # (sum of width * height, NUM_CLASSES + 6)
        outputs = tf.concat(outputs, axis=0)
        return (input / 255.0, outputs)
