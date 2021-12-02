from modeling.fcos import FCOS

import tensorflow as tf

m = FCOS.make_functional()
m.summary()

tf.keras.utils.plot_model(m, show_shapes=True)
