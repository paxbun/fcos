from constants import *

import tensorflow as tf


class FCOSLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        focal_loss_gamma: int,
        focal_loss_alpha: float,
        use_giou: bool,
        epsilon: float = 1e-8,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha
        self.use_giou = use_giou
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        y_true_class, y_true_centerness, y_true_reg = \
            y_true[:, :, :(NUM_CLASSES + 1)], \
            y_true[:, :, NUM_CLASSES + 1], \
            y_true[:, :, -4:]
        y_pred_class, y_pred_centerness, y_pred_reg = \
            y_pred[:, :, :(NUM_CLASSES + 1)], \
            y_pred[:, :, NUM_CLASSES + 1], \
            y_pred[:, :, -4:]

        is_positive = 1 - y_true_class[:, :, 0]
        num_positives = tf.reduce_sum(is_positive, axis=1)
        num_pixels = y_true.shape[1]

        focal_loss = FCOSLoss._focal_loss(
            y_true_class, y_pred_class, self.focal_loss_gamma, self.focal_loss_alpha, self.epsilon)
        centerness_loss = FCOSLoss._centerness_loss(
            y_true_centerness, y_pred_centerness, self.epsilon)
        iou_loss = (FCOSLoss._giou if self.use_giou else FCOSLoss._iou)(
            y_true_reg, y_pred_reg, self.epsilon)
        iou_loss = iou_loss * is_positive

        rtn = focal_loss + centerness_loss + iou_loss
        rtn = rtn * num_pixels / tf.reshape(num_positives, (-1, 1))

        return iou_loss

    @staticmethod
    def _focal_loss(y_true, y_pred, gamma, alpha, epsilon):
        loss_true = -y_true * \
            tf.math.pow(1 - y_pred, gamma) * \
            tf.math.log(y_pred + epsilon) * \
            alpha
        loss_false = (y_true - 1) * \
            tf.math.pow(y_pred, gamma) * \
            tf.math.log(1 - y_pred + epsilon) * \
            (1 - alpha)
        return tf.reduce_sum(loss_true + loss_false, axis=-1)

    @staticmethod
    def _centerness_loss(y_true, y_pred, epsilon):
        loss_true = -y_true * tf.math.log(y_pred + epsilon)
        loss_false = (y_true - 1) * tf.math.log(1 - y_pred + epsilon)
        return loss_true + loss_false

    @staticmethod
    def _giou(y_true, y_pred, epsilon):
        y_true_l, y_true_r, y_true_t, y_true_b = \
            y_true[:, :, LEFT], \
            y_true[:, :, RIGHT], \
            y_true[:, :, TOP], \
            y_true[:, :, BOTTOM]
        y_pred_l, y_pred_r, y_pred_t, y_pred_b = \
            y_pred[:, :, LEFT], \
            y_pred[:, :, RIGHT], \
            y_pred[:, :, TOP], \
            y_pred[:, :, BOTTOM]
        area_true = (y_true_l + y_true_r) * (y_true_t + y_true_b)
        area_pred = (y_pred_l + y_pred_r) * (y_pred_t + y_pred_b)
        intersection_height = tf.math.minimum(
            y_true_l, y_pred_l
        ) + tf.math.minimum(
            y_true_r, y_pred_r
        )
        intersection_width = tf.math.minimum(
            y_true_t, y_pred_t
        ) + tf.math.minimum(
            y_true_b, y_pred_b
        )
        convex_height = tf.math.maximum(
            y_true_l, y_pred_l
        ) + tf.math.maximum(
            y_true_r, y_pred_r
        )
        convex_width = tf.math.maximum(
            y_true_t, y_pred_t
        ) + tf.math.maximum(
            y_true_b, y_pred_b
        )
        intersection_area = intersection_height * intersection_width
        convex_area = convex_height * convex_width
        union_area = area_true + area_pred - intersection_area
        iou = intersection_area / (union_area + epsilon)
        # GIoU = IoU - |C \ (A U B)|/|C|
        #      = IoU - (|C| - |A U B|)/|C|
        #      = IoU - 1 + |A U B|/|C|
        giou = iou - 1 + union_area / (convex_area + epsilon)
        return 1 - giou

    @staticmethod
    def _iou(y_true, y_pred, epsilon):
        y_true_l, y_true_r, y_true_t, y_true_b = \
            y_true[:, :, 0], \
            y_true[:, :, 1], \
            y_true[:, :, 2], \
            y_true[:, :, 3]
        y_pred_l, y_pred_r, y_pred_t, y_pred_b = \
            y_pred[:, :, 0], \
            y_pred[:, :, 1], \
            y_pred[:, :, 2], \
            y_pred[:, :, 3]
        area_true = (y_true_l + y_true_r) * (y_true_t + y_true_b)
        area_pred = (y_pred_l + y_pred_r) * (y_pred_t + y_pred_b)
        intersection_height = tf.math.minimum(
            y_true_l, y_pred_l
        ) + tf.math.minimum(
            y_true_r, y_pred_r
        )
        intersection_width = tf.math.minimum(
            y_true_t, y_pred_t
        ) + tf.math.minimum(
            y_true_b, y_pred_b
        )
        intersection_area = intersection_height * intersection_width
        union_area = area_true + area_pred - intersection_area
        iou = intersection_area / (union_area + epsilon)
        return -tf.math.log(iou + epsilon)
