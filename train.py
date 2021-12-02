from fcos_data import FCOSPreprocessor
from modeling.fcos import FCOS

import math
import tensorflow as tf
import tensorflow_datasets as tfds
import datetime

def prepare_dataset(split: str):
    ds = tfds.load("coco", split=split, shuffle_files=True)
    ds = ds.map(
        FCOSPreprocessor(),
        num_parallel_calls=-1,
        deterministic=False,
    )
    ds = ds.batch(16)
    return ds

if __name__ == "__main__":
    log_dir = "out11/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    train_ds = prepare_dataset("train")
    validation_ds = prepare_dataset("validation")
    m = FCOS.make()
    m.fit(
        train_ds,
        validation_data=validation_ds,
        # epochs=int(math.ceil(90000 / len(ds))),
        epochs=1,
        callbacks=[
            tf.keras.callbacks.LearningRateScheduler(
                FCOS.make_lr_scheduler(len(train_ds))),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
            tf.keras.callbacks.CSVLogger("out11/log.csv", append=True),
            tf.keras.callbacks.experimental.BackupAndRestore("out11/bck"),
        ]
    )
    m.save_weights("out11/saved_weights")
