import tensorflow as tf

# Size of an image in the original dataset
ORIGINAL_IMAGE_SHAPE = (334, 500)

# Size of the input (excluding number of channels, which is 3)
IMAGE_SHAPE = (800, 1024)

# Number of classes in the COCO dataset
NUM_CLASSES = 80

# Size of the output of P3
OUTPUT_SHAPE_P3 = (100, 128)

# Size of the output of P4
OUTPUT_SHAPE_P4 = (50, 64)

# Size of the output of P5
OUTPUT_SHAPE_P5 = (25, 32)

# Size of the output of P6
OUTPUT_SHAPE_P6 = (13, 16)

# Size of the output of P7
OUTPUT_SHAPE_P7 = (7, 8)

# Size of the output of [P3, P4, ..., P7]
OUTPUT_SHAPES = [
    OUTPUT_SHAPE_P3,
    OUTPUT_SHAPE_P4,
    OUTPUT_SHAPE_P5,
    OUTPUT_SHAPE_P6,
    OUTPUT_SHAPE_P7
]

# (minval, maxval) of the distances for each stage (not normalized)
MAX_DISTS = [
    (-1, 64),
    (64, 128),
    (128, 256),
    (256, 512),
    (512, 100000)
]


def make_grid_from_output_shape(height, width):
    y_half_unit, x_half_unit = 0.5 / height, 0.5 / width
    y_points = tf.linspace(y_half_unit, 1 - y_half_unit, height)
    x_points = tf.linspace(x_half_unit, 1 - x_half_unit, width)
    x, y = tf.meshgrid(x_points, y_points)
    return y, x


# Collection of (y, x) points for [P3, P4, ..., P7]
GRIDS = [
    make_grid_from_output_shape(*output_shape)
    for output_shape in OUTPUT_SHAPES
]

# Column index of distance to the upper side of the box
TOP = 0

# Column index of distance to the left side of the box
LEFT = 1

# Column index of distance to the lower side of the box
BOTTOM = 2

# Column index of distance to the right side of the box
RIGHT = 3

# Number of groups in GN
NUM_GROUPS_GN = 32

# gamma in focal loss
FOCAL_LOSS_GAMMA = 2

# alpha in balanced focal loss
FOCAL_LOSS_ALPHA = 0.25
