import tensorflow as tf


class FCOSLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        focal_loss_gamma: int,
        use_giou: bool,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.focal_loss_gamma = focal_loss_gamma
        self.use_giou = use_giou

    def call(self, y_true, y_pred):
        # TODO
        pass

    @staticmethod
    def _giou(y_true, y_pred):
        area_true = (y_true[0] + y_true[1]) * (y_true[2] + y_true[3])
        area_pred = (y_pred[0] + y_pred[1]) * (y_pred[2] + y_pred[3])
        intersection_height = tf.math.minimum(
            y_true[0], y_pred[0]
        ) + tf.math.minimum(
            y_true[1], y_pred[1]
        )
        intersection_width = tf.math.minimum(
            y_true[2], y_pred[2]
        ) + tf.math.minimum(
            y_true[3], y_pred[3]
        )
        convex_height = tf.math.maximum(
            y_true[0], y_pred[0]
        ) + tf.math.maximum(
            y_true[1], y_pred[1]
        )
        convex_width = tf.math.maximum(
            y_true[2], y_pred[2]
        ) + tf.math.maximum(
            y_true[3], y_pred[3]
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
        area_true = (y_true[0] + y_true[1]) * (y_true[2] + y_true[3])
        area_pred = (y_pred[0] + y_pred[1]) * (y_pred[2] + y_pred[3])
        intersection_height = tf.math.minimum(
            y_true[0], y_pred[0]
        ) + tf.math.minimum(
            y_true[1], y_pred[1]
        )
        intersection_width = tf.math.minimum(
            y_true[2], y_pred[2]
        ) + tf.math.minimum(
            y_true[3], y_pred[3]
        )
        intersection_area = intersection_height * intersection_width
        union_area = area_true + area_pred - intersection_area
        return intersection_area / union_area

    @staticmethod
    def _focal_loss(y_true, y_pred, gamma):
        loss_true = -y_true * \
            tf.math.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
        loss_false = (y_true - 1) * tf.math.pow(y_pred, gamma) * \
            tf.math.log(1 - y_pred)
        return loss_true + loss_false
