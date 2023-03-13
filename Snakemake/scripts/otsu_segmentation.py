import argparse
import yaml
import numpy as np
from utils import contrast_stretching, get_image, remove_so, otsu_threshold, tophat_filter
from aicsimageio.writers import OmeTiffWriter


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
        image, config["otsu_segmentation"]["contrast"]["percentile"]
    )

    image_tophat = tophat_filter(
        image_contrast, config["otsu_segmentation"]["tophat"]["element_size"]
    )

    image_otsu = otsu_threshold(
        image_tophat, config["otsu_segmentation"]["otsu_threshold"]["factor"]
    )

    image_otsu_smallobject = remove_so(
        image_otsu,
        config["otsu_segmentation"]["small_objects"]["min_size"],
        config["otsu_segmentation"]["small_objects"]["connectivity"],
    )
    
    image_processed = image_otsu_smallobject.astype(np.uint8)

    OmeTiffWriter.save(image_processed, args.output, dim_order="YX")
