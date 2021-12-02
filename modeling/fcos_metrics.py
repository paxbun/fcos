from inspect import isroutine
from constants import *
from typing import List, Optional
from typing_extensions import Literal
from dataclasses import dataclass

import tensorflow as tf
import numpy as np


@dataclass
class CocoAveragePrecisionConfiguration:
    nms_threshold: float = 0.6
    iou_threshold: float = 0.5
    scale: Optional[Literal["small", "medium", "large"]] = None


class CocoAveragePrecision(tf.keras.metrics.Metric):
    def __init__(self, config: CocoAveragePrecisionConfiguration, **kwargs):
        super(CocoAveragePrecision, self).__init__(**kwargs)
        self.config = config
        self.tp = self.add_weight("tp", dtype=tf.int64, initializer="zeros")
        self.fp = self.add_weight("fp", dtype=tf.int64, initializer="zeros")

    @staticmethod
    def _get_boxes_from_prediction(y_pred):
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
        score = tf.gather(y_pred, cls, axis=2, batch_dims=1) * centerness

        return tf.stack([
            y_coords - top,
            x_coords - left,
            y_coords + bottom,
            x_coords + right,
        ], axis=2), cls, score

    @staticmethod
    def _iou(boxes_1, boxes_2, epsilon: float):
        intersection_height = tf.math.minimum(boxes_1[:, :, :, BOTTOM], boxes_2[:, :, :, BOTTOM]) - \
            tf.math.maximum(boxes_1[:, :, :, TOP], boxes_2[:, :, :, TOP])
        intersection_width = tf.math.minimum(boxes_1[:, :, :, RIGHT], boxes_2[:, :, :, RIGHT]) - \
            tf.math.maximum(boxes_1[:, :, :, LEFT], boxes_2[:, :, :, LEFT])
        intersection_area = intersection_height * intersection_width
        area_1 = (boxes_1[:, :, :, BOTTOM] - boxes_1[:, :, :, TOP]) * \
            (boxes_1[:, :, :, RIGHT] - boxes_1[:, :, :, LEFT])
        area_2 = (boxes_2[:, :, :, BOTTOM] - boxes_2[:, :, :, TOP]) * \
            (boxes_2[:, :, :, RIGHT] - boxes_2[:, :, :, LEFT])
        union_area = area_1 + area_2 - intersection_area
        iou = intersection_area / (union_area + epsilon)
        return iou

    @staticmethod
    def _check_box_scale(boxes, scale: Optional[Literal["small", "medium", "large"]]) -> bool:
        # boxes: (None, sum of height * width, 1, 4)
        image_height, image_width = ORIGINAL_IMAGE_SHAPE

        # (None, sum of height * width, 1)
        area = (boxes[:, :, :, BOTTOM] - boxes[:, :, :, TOP]) * \
            (boxes[:, :, :, RIGHT] - boxes[:, :, :, LEFT])
        area = area * (image_height * image_width)

        # (None, sum of height * width, 1)
        if scale == "small":
            return tf.less_equal(area, 32 * 32)
        elif scale == "medium":
            return tf.logical_and(tf.greater(area, 32 * 32), tf.less_equal(area, 96 * 96))
        else:
            return tf.greater(area, 96 * 96)

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred_boxes, pred_cls, pred_score = CocoAveragePrecision._get_boxes_from_prediction(
            y_pred)
        
        pred_nms = CocoAveragePrecision._perform_non_max_suppression(
            pred_boxes, pred_cls, pred_score, self.config.nms_threshold)

        tf.image.non

        pred_boxes = tf.expand_dims(pred_boxes, 1)
        pred_cls = tf.expand_dims(pred_cls, 1)
        pred_nms = tf.expand_dims(pred_nms, 1)

        true_boxes, true_cls, true_score = CocoAveragePrecision._get_boxes_from_prediction(
            y_true)
        true_nms = CocoAveragePrecision._perform_non_max_suppression(
            true_boxes, true_cls, true_score, 0.99)

        true_boxes = tf.expand_dims(true_boxes, 2)
        true_cls = tf.expand_dims(true_cls, 2)
        true_nms = tf.expand_dims(true_nms, 2)

        cls_cond = tf.equal(pred_cls, true_cls)

        iou = CocoAveragePrecision._iou(pred_boxes, true_boxes, 1e-8)
        iou_cond = tf.greater(iou, self.config.iou_threshold)

        scale_cond = CocoAveragePrecision._check_box_scale(
            true_boxes, self.config.scale)

        # (None, sum of height * width, sum of height * width)
        cond = tf.logical_and(cls_cond, iou_cond)
        cond = tf.logical_and(cond, scale_cond)

        # (None, sum of height * width)
        cond = tf.reduce_any(cond, 2)

        new_tp = tf.math.count_nonzero(tf.logical_and(pred_nms, cond))
        new_fp = tf.math.count_nonzero(
            tf.logical_and(pred_nms, tf.logical_not(cond)))

        self.tp.assign_add(new_tp)
        self.fp.assign_add(new_fp)

    def result(self):
        return self.tp / (self.tp + self.fp)


@dataclass
class CocoAveragedAveragePrecisionConfiguration:
    nms_threshold: float = 0.6
    scale: Optional[Literal["small", "medium", "large"]] = None


class CocoAveragedAveragePrecision(tf.keras.metrics.Metric):
    def __init__(self, config: CocoAveragedAveragePrecisionConfiguration, **kwargs):
        super(CocoAveragedAveragePrecision, self).__init__(**kwargs)
        self.config = config
        self.precisions: List[CocoAveragePrecision] = [
            CocoAveragePrecision(
                config=CocoAveragePrecisionConfiguration(
                    iou_threshold=iou_threshold, scale=self.config.scale
                )
            )
            for iou_threshold in np.arange(0.5, 0.95, 0.05)
        ]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for precision in self.precisions:
            precision.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return sum(precision.result() for precision in self.precisions) / len(self.precisions)


@dataclass
class CocoAverageRecallConfiguration:
    max: int = 1
    scale: Optional[Literal["small", "medium", "large"]] = None


class CocoAverageRecallConfiguration(tf.keras.metrics.Metric):
    pass
