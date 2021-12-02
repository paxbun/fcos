from tensorflow.keras import callbacks
from modeling.resnet import ResNet50, ResNetTest, ResNetTop, ResNetBlock
from absl import app, flags

import tensorflow as tf
import tensorflow_datasets as tfds
import datetime


FLAGS = flags.FLAGS
flags.DEFINE_string("out", "out", help="foo")

def preprocessing_fn(raw_feature: dict):
    image = raw_feature["image"]
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    label = raw_feature["label"]

    return image, label


def prepare_dataset(split: str):
    ds = tfds.load("cifar10", split=split, shuffle_files=True)
    ds = ds.map(
        preprocessing_fn,
        num_parallel_calls=-1,
        deterministic=False,
    )
    ds = ds.batch(64)
    return ds


def main(_):
    ds = prepare_dataset("train")
    ds_valid = prepare_dataset("test")

    out_dir = FLAGS.out

    log_dir = out_dir + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    input = tf.keras.layers.Input((32, 32, 3))
    # output = ResNetTop()(input)
    # output = ResNetTest()(input)
    # output = ResNet50()(input)
    output = input
    for stage in range(2):
        for idx in range(4):
            output = ResNetBlock(
                out_channels=128 * (stage + 1),
                downscale=(idx == 0),
                num_groups=None,
                group_width=None
            )(output)
    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(10, activation="softmax")(output)
    model = tf.keras.models.Model(inputs=input, outputs=output)
    model.compile(
        # optimizer=tf.keras.optimizers.Adam(0.01),
        # optimizer=tf.keras.optimizers.SGD(0.1, 0.9),
        optimizer=tf.keras.optimizers.SGD(0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.fit(
        ds,
        epochs=100,
        validation_data=ds_valid,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
            tf.keras.callbacks.CSVLogger(out_dir + "/log.csv", append=True),
            tf.keras.callbacks.experimental.BackupAndRestore(out_dir + "/bck"),
        ]
    )
    model.save_weights(out_dir + "/saved_weights")

if __name__ == "__main__":
    app.run(main)