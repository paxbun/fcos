from fcos_data import FCOSPreprocessor
from modeling.fcos import FCOS

import math
import tensorflow as tf
import tensorflow_datasets as tfds
import datetime

if __name__ == "__main__":
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ds = tfds.load("coco", split="train", shuffle_files=True)
    ds = ds.map(
        FCOSPreprocessor(),
        num_parallel_calls=-1,
        deterministic=False,
    )
    ds = ds.batch(16)
    m = FCOS.make()
    m.fit(
        ds,
        epochs=int(math.ceil(90000 / len(ds))),
        callbacks=[
            tf.keras.callbacks.LearningRateScheduler(
                FCOS.make_lr_scheduler(len(ds))),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
            tf.keras.callbacks.CSVLogger("log.csv", append=True),
            tf.keras.callbacks.experimental.BackupAndRestore("bck"),
        ]
    )
    m.save_weights("saved_weights")
