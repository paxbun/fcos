import tensorflow as tf

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 1024
IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT)
NUM_CLASSES = 80

OUTPUT_SHAPE_P3 = (100, 128)
OUTPUT_SHAPE_P4 = (50, 64)
OUTPUT_SHAPE_P5 = (25, 32)
OUTPUT_SHAPE_P6 = (13, 16)
OUTPUT_SHAPE_P7 = (7, 8)
OUTPUT_SHAPES = [
    OUTPUT_SHAPE_P3,
    OUTPUT_SHAPE_P4,
    OUTPUT_SHAPE_P5,
    OUTPUT_SHAPE_P6,
    OUTPUT_SHAPE_P7
]

MAX_DISTS = [
    (-1, 64),
    (64, 128),
    (128, 256),
    (256, 512),
    (512, 100000)
]
MAX_DISTS = [
    (x / IMAGE_WIDTH, y / IMAGE_HEIGHT)
    for x, y in MAX_DISTS
]


def make_grid_from_output_shape(output_shape):
    x_unit, y_unit = 0.5 / output_shape[0], 0.5 / output_shape[1]
    x_points = tf.linspace(x_unit, 1 - x_unit, output_shape[0])
    y_points = tf.linspace(y_unit, 1 - y_unit, output_shape[1])
    return tf.meshgrid(y_points, x_points)


GRIDS = [
    make_grid_from_output_shape(output_shape)
    for output_shape in OUTPUT_SHAPES
]

LEFT = 0
RIGHT = 1
TOP = 2
BOTTOM = 3