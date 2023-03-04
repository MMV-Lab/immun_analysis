import argparse
import numpy as np
from skimage import filters, io, morphology, img_as_float32
from aicsimageio import AICSImage
import cv2

############   Functions   ############
def get_image(dir):
    return AICSImage(dir).get_image_data("YX")


def contrast_stretching(image):
    upper_v = np.percentile(image, 99.9)
    lower_v = np.percentile(image, 0.1)

    image[image > upper_v] = upper_v
    image[image < lower_v] = lower_v
    return (image - lower_v) / (upper_v - lower_v)


def tophat_filter(image):
    kernel50 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel50)


def otsu_threshold(image, factor):
    Threshold = filters.threshold_otsu(image)
    return image > Threshold * factor


def remove_small_objects(image):
    return morphology.remove_small_objects(image, min_size=30, connectivity=2)


############   Main   ############
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loads .tiff images and segments them with OTSU-Threshold"
    )
    # Positional Arguments
    parser.add_argument("--input", type=str, help="Path to the images")
    parser.add_argument("--output", type=str, help="Path to store the segmented images")
    # Required Arguments
    parser.add_argument("--otsu", type=float, help="Otsu threshold factor", default=2.1)
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, "|>", getattr(args, arg))

    image = get_image(args.input)

    image_contrast = contrast_stretching(image)

    image_tophat = tophat_filter(image_contrast)

    image_otsu = otsu_threshold(image_tophat, args.otsu)

    image_otsu_smallobject = remove_small_objects(image_otsu)

    io.imsave(
        args.output,
        img_as_float32(image_otsu_smallobject),
        plugin="pil",
        optimize=True,
        bits=1,
    )
