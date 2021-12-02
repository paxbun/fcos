from constants import *
from modeling.fcos import FCOS

import tensorflow as tf
import tensorflow_datasets as tfds

def resize_image(raw_feature: dict) -> tf.Tensor:
    input = tf.image.resize(raw_feature["image"], IMAGE_SHAPE)
    return input

def get_boxes_from_prediction_raw(y_pred):
    # y_pred: (None, sum of height * width, NUM_CLASSES + 6)
    # (1, sum of height * width)
    y_coords = [
        tf.reshape(y, (1, height * width))
        for (y, _), (height, width) in zip(GRIDS, OUTPUT_SHAPES)
    ]
    y_coords = tf.concat(y_coords, axis=1)
    x_coords = [
        tf.reshape(x, (1, height * width))
        for (_, x), (height, width) in zip(GRIDS, OUTPUT_SHAPES)
    ]
    x_coords = tf.concat(x_coords, axis=1)

    # (None, sum of height * width)
    left = y_pred[:, :, NUM_CLASSES + 2 + LEFT]
    top = y_pred[:, :, NUM_CLASSES + 2 + TOP]
    right = y_pred[:, :, NUM_CLASSES + 2 + RIGHT]
    bottom = y_pred[:, :, NUM_CLASSES + 2 + BOTTOM]
    cls = tf.argmax(y_pred[:, :, :NUM_CLASSES + 1], axis=-1)
    centerness = y_pred[:, :, NUM_CLASSES + 1]
    score = tf.gather(y_pred, cls, axis=2, batch_dims=2) * centerness

    return tf.stack([
        y_coords - top,
        x_coords - left,
        y_coords + bottom,
        x_coords + right,
    ], axis=2), cls, score

def get_boxes_from_raw(boxes, cls, score):
    # boxes: (sum of height * width, 4)
    # cls  : (sum of height * width, )
    # score: (sum of height * width, )

    indices = tf.where(cls > 0)
    boxes = tf.gather(boxes, indices, axis=0)
    cls = tf.gather(cls, indices)
    score = tf.gather(score, indices)

    print(boxes, cls, score)

ds = tfds.load("coco", split="test2015", shuffle_files=True)
raw_features = next(iter(ds.take(1)))
input = resize_image(raw_features)

m = FCOS.make()
m.load_weights("out7/saved_weights")

result = m.predict(tf.expand_dims(input, axis=0))
boxes, cls, score = get_boxes_from_prediction_raw(result)
boxes = get_boxes_from_raw(boxes[0], cls[0], score[0])