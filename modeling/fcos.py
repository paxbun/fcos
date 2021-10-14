from constants import IMAGE_SHAPE
from .fcos_loss import FCOSLoss
from .fcos_layers import FeaturePyramid, FCOSHead
from .resnet import ResNet50, ResNet101, ResNeXt32x8d, ResNeXt64x4d

import tensorflow as tf
import tensorflow_addons as tfa


class FCOS(tf.keras.Model):
    _resnets = {
        "ResNet50": ResNet50,
        "ResNet101": ResNet101,
        "ResNeXt32x8d": ResNeXt32x8d,
        "ResNeXt64x4d": ResNeXt64x4d,
    }

    @staticmethod
    def make(resnet: str = "ResNet50"):
        fcos = FCOS(resnet)
        fcos.compile(
            optimizer=tfa.optimizers.SGDW(0.0001, 0.01, 0.9),
            loss=FCOSLoss(2, 0.25, True),
        )
        fcos.build((None, *IMAGE_SHAPE, 3))
        return fcos

    # @staticmethod
    # def make_functional(resnet: str = "ResNet50"):
    #     x = input = tf.keras.layers.Input((*IMAGE_SHAPE, 3))
    #     x = FCOS._resnets[resnet]()(x)
    #     x = FeaturePyramid()(x[1:])
    #     x = FCOSHead()(x)
    #     model = tf.keras.Model(input, x)
    #     model.compile(
    #         optimizer=tf.keras.optimizers.SGD(),
    #         loss=FCOSLoss(2, 0.25, True),
    #     )
    #     model.build((None, *IMAGE_SHAPE, 3))
    #     return model

    @staticmethod
    def make_lr_scheduler(num_batches_per_epoch):
        def lr_scheduler(epoch, lr):
            if epoch < 60000 / num_batches_per_epoch:
                return lr
            elif epoch < 80000 / num_batches_per_epoch:
                return lr / 10
            else:
                return lr / 100
        return lr_scheduler

    def __init__(
        self,
        resnet: str,
        *args,
        **kwargs
    ):
        super(FCOS, self).__init__(*args, **kwargs)
        self.resnet = FCOS._resnets[resnet]()
        self.fpn = FeaturePyramid()
        self.head = FCOSHead()

    def call(self, inputs):
        output = self.resnet(inputs)
        pyramid = self.fpn(output[1:])
        rtn = self.head(pyramid)
        return rtn
