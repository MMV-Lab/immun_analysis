from aicsimageio import AICSImage
from skimage.filters import threshold_otsu
import numpy as np
from skimage import img_as_bool
from skimage.morphology import footprints, remove_small_objects, white_tophat


def contrast_stretching(image, percentile):
    upper_v = np.percentile(image, percentile[1])
    lower_v = np.percentile(image, percentile[0])

    image[image > upper_v] = upper_v
    image[image < lower_v] = lower_v
    image = (image - lower_v) / (upper_v - lower_v)
    return image


def get_binary_img(image):
    return img_as_bool(image / 255)


def get_image(dir):
    return AICSImage(dir).get_image_data("YX")


def otsu_threshold(image, factor):
    Threshold = threshold_otsu(image)
    return image > Threshold * factor


def remove_so(image, min_size, connectivity):
    return remove_small_objects(image, min_size=min_size, connectivity=connectivity)


def tophat_filter(image, size):
    return white_tophat(image, footprints.ellipse(size[0], size[1]))


