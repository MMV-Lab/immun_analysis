from skimage import measure
import numpy as np
from utils import get_binary_img, get_image
import argparse
from aicsimageio.writers import OmeTiffWriter


############   Main   ############
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loads .tiff images and segments them with OTSU-Threshold"
    )
    # Positional Arguments
    parser.add_argument(
        "--input", type=str, nargs="+", help="Paths to the otsu and dapi image"
    )
    parser.add_argument(
        "--output", type=str, help="Path to store the registered images"
    )
    # Will be swapped out for new script
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, "|>", getattr(args, arg))

    otsu = get_binary_img(get_image(args.input[0]))
    dapi = get_binary_img(get_image(args.input[1]))

    x1_labels = measure.label(otsu, connectivity=2)
    x1_regionprops = measure.regionprops(x1_labels)

    object_coords = [obj["coords"] for obj in x1_regionprops]

    for obj in object_coords:
        count = 0
        for coords in obj:
            if dapi[coords[0], coords[1]] == 1:
                count = count + 1
        percent = count / len(obj)
        if percent == 0:
            for coords in obj:
                otsu[coords[0], coords[1]] = 0

    image_processed = otsu.astype(np.uint8)

    OmeTiffWriter.save(image_processed, args.output, dim_order="YX")
