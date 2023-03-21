import numpy as np
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from skimage.filters import threshold_otsu
from yaml import safe_load
from skimage.morphology import remove_small_objects, white_tophat
from skimage.morphology.footprints import ellipse


def binarize(image, asbool=False):
    image[image > 0] = 1
    if asbool:
        image = image.astype(np.bool_)
    return image


def contrast_stretching(image, percentile):
    upper_v = np.percentile(image, percentile[1])
    lower_v = np.percentile(image, percentile[0])

    image[image > upper_v] = upper_v
    image[image < lower_v] = lower_v
    image = (image - lower_v) / (upper_v - lower_v)
    return image


def get_config(path):
    with open(path, "r") as file:
        config = safe_load(file)
    return config


def get_image_data(reader, channel = False):
    if channel:
        image = reader.get_image_data("YX", C=reader.channel_names.index(channel), T=0, Z=0)
    else:
        image = reader.get_image_data("YX", C=0, T=0, Z=0)
    return image


def get_reader(path):
    return AICSImage(path)


def otsu_threshold(image, factor):
    Threshold = threshold_otsu(image)
    return image > Threshold * factor


def remove_so(image, min_size, connectivity):
    return remove_small_objects(image, min_size=min_size, connectivity=connectivity)


def save_image(image, path, asuint=False):
    if asuint:
        image = image.astype(np.uint8)
        image[image > 0] = 255
    OmeTiffWriter.save(image, path, dim_order="YX")


def tophat_filter(image, size):
    return white_tophat(image, ellipse(size[0], size[1]))
