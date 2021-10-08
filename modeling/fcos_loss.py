from constants import NUM_CLASSES

import tensorflow as tf


class FCOSLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        focal_loss_gamma: int,
        focal_loss_alpha: float,
        use_giou: bool,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha
        self.use_giou = use_giou

    def call(self, y_true, y_pred):
        y_true_class, y_true_centerness, y_true_reg = \
            y_true[:, :, :, :(NUM_CLASSES + 1)], \
            y_true[:, :, :, NUM_CLASSES + 1], \
            y_true[:, :, :, -4:]
        y_pred_class, y_pred_centerness, y_pred_reg = \
            y_pred[:, :, :, :(NUM_CLASSES + 1)], \
            y_pred[:, :, :, NUM_CLASSES + 1], \
            y_pred[:, :, :, -4:]

        is_positive = 1 - y_true_class[0]
        num_positives = tf.reduce_sum(is_positive, [-2, -1])

        height, width = y_true.shape[1:3]
        num_pixels = height * width

        focal_loss = FCOSLoss._focal_loss(
            y_true_class, y_pred_class, self.focal_loss_gamma, self.focal_loss_alpha)
        centerness_loss = FCOSLoss._centerness_loss(
            y_true_centerness, y_pred_centerness)
        iou_loss = (FCOSLoss._giou if self.use_giou else FCOSLoss._iou)(
            y_true_reg, y_pred_reg)
        iou_loss = iou_loss * is_positive
        iou_loss = -tf.math.log(iou_loss)

        rtn = focal_loss + centerness_loss + iou_loss
        rtn = rtn * num_pixels / num_positives

        return rtn

    @staticmethod
    def _focal_loss(y_true, y_pred, gamma, alpha):
        loss_true = -y_true * \
            tf.math.pow(1 - y_pred, gamma) * \
            tf.math.log(y_pred) * \
            alpha
        loss_false = (y_true - 1) * \
            tf.math.pow(y_pred, gamma) * \
            tf.math.log(1 - y_pred) * \
            (1 - alpha)
        return tf.reduce_sum(loss_true + loss_false, axis=-1)

    @staticmethod
    def _centerness_loss(y_true, y_pred):
        loss_true = -y_true * tf.math.log(y_pred)
        loss_false = (y_true - 1) * tf.math.log(1 - y_pred)
        return loss_true + loss_false

    @staticmethod
    def _giou(y_true, y_pred):
        y_true0, y_true1, y_true2, y_true3 = \
            y_true[:, :, :, 0], \
            y_true[:, :, :, 1], \
            y_true[:, :, :, 2], \
            y_true[:, :, :, 3]
        y_pred0, y_pred1, y_pred2, y_pred3 = \
            y_pred[:, :, :, 0], \
            y_pred[:, :, :, 1], \
            y_pred[:, :, :, 2], \
            y_pred[:, :, :, 3]
        area_true = (y_true0 + y_true1) * (y_true2 + y_true3)
        area_pred = (y_pred0 + y_pred1) * (y_pred2 + y_pred3)
        intersection_height = tf.math.minimum(
            y_true0, y_pred0
        ) + tf.math.minimum(
            y_true1, y_pred1
        )
        intersection_width = tf.math.minimum(
            y_true2, y_pred2
        ) + tf.math.minimum(
            y_true3, y_pred3
        )
        convex_height = tf.math.maximum(
            y_true0, y_pred0
        ) + tf.math.maximum(
            y_true1, y_pred1
        )
        convex_width = tf.math.maximum(
            y_true2, y_pred2
        ) + tf.math.maximum(
            y_true3, y_pred3
        )
        intersection_area = intersection_height * intersection_width
        convex_area = convex_height * convex_width
        union_area = area_true + area_pred - intersection_area
        iou = intersection_area / union_area
        # GIoU = IoU - |C \ (A U B)|/|C|
        #      = IoU - (|C| - |A U B|)/|C|
        #      = IoU - 1 + |A U B|/|C|
        return iou - 1 + union_area / convex_area

    @staticmethod
    def _iou(y_true, y_pred):
        y_true0, y_true1, y_true2, y_true3 = \
            y_true[:, :, :, 0], \
            y_true[:, :, :, 1], \
            y_true[:, :, :, 2], \
            y_true[:, :, :, 3]
        y_pred0, y_pred1, y_pred2, y_pred3 = \
            y_pred[:, :, :, 0], \
            y_pred[:, :, :, 1], \
            y_pred[:, :, :, 2], \
            y_pred[:, :, :, 3]
        area_true = (y_true0 + y_true1) * (y_true2 + y_true3)
        area_pred = (y_pred0 + y_pred1) * (y_pred2 + y_pred3)
        intersection_height = tf.math.minimum(
            y_true0, y_pred0
        ) + tf.math.minimum(
            y_true1, y_pred1
        )
        intersection_width = tf.math.minimum(
            y_true2, y_pred2
        ) + tf.math.minimum(
            y_true3, y_pred3
        )
        intersection_area = intersection_height * intersection_width
        union_area = area_true + area_pred - intersection_area
        return intersection_area / union_area
