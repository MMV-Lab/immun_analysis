import argparse
import yaml
import numpy as np
from skimage import filters, io, morphology, img_as_float32
from aicsimageio import AICSImage
from cv2 import getStructuringElement, MORPH_ELLIPSE, morphologyEx, MORPH_TOPHAT

############   Functions   ############
def get_image(dir):
    return AICSImage(dir).get_image_data("YX")


def contrast_stretching(image, percentile):
    upper_v = np.percentile(image, percentile[1])
    lower_v = np.percentile(image, percentile[0])

    image[image > upper_v] = upper_v
    image[image < lower_v] = lower_v
    return (image - lower_v) / (upper_v - lower_v)


def tophat_filter(image, size):
    kernel50 = getStructuringElement(MORPH_ELLIPSE, (size[0], size[1]))
    return morphologyEx(image, MORPH_TOPHAT, kernel50)


def otsu_threshold(image, factor):
    Threshold = filters.threshold_otsu(image)
    return image > Threshold * factor


def remove_small_objects(image, min_size, connectivity):
    return morphology.remove_small_objects(
        image, min_size=min_size, connectivity=connectivity
    )


############   Main   ############
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loads .tiff images and segments them with OTSU-Threshold"
    )
    # Positional Arguments
    parser.add_argument("input", type=str, help="Path to the image")
    parser.add_argument("output", type=str, help="Path to output & name")
    # Required Arguments
    parser.add_argument("-c", type=str, help="Path to config.yaml")
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, "|>", getattr(args, arg))

    with open(args.c, "r") as file:
        config = yaml.safe_load(file)

    image = get_image(args.input)

    image_contrast = contrast_stretching(
        image, config["otsu"]["contrast"]["percentile"]
    )

    image_tophat = tophat_filter(image_contrast, config["otsu"]["tophat"]["size"])

    image_otsu = otsu_threshold(image_tophat, config["otsu"]["threshold"]["factor"])

    image_otsu_smallobject = remove_small_objects(
        image_otsu,
        config["otsu"]["small_objects"]["min_size"],
        config["otsu"]["small_objects"]["connectivity"],
    )

    io.imsave(
        args.output,
        img_as_float32(image_otsu_smallobject),
        plugin="pil",
        optimize=True,
        bits=1,
    )
