import argparse
import numpy as np
from utils import binarize, get_reader, get_image_data, get_config, save_image
from skimage.morphology import dilation, disk, label


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
        "--output", type=str, nargs="+", help="Path to store the registered images"
    )
    parser.add_argument("-c", type=str, help="Path to config.yaml")
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, "|>", getattr(args, arg))

    config = get_config(args.c)
    print("start")
    otsu = binarize(get_image_data(get_reader(args.input[0])))
    reader = get_reader(args.input[1])
    dapi = binarize(get_image_data(reader))

    lbdapi, numdapi = label(dapi, return_num=True)
    lbotsu, numotsu = label(otsu, return_num=True)

    distance_filter_threshold = round(
        config["colocalization"]["distance_filter_threshold"]
        / (reader.physical_pixel_sizes.X)
    )

    disk_dilation = disk(distance_filter_threshold)

    dapi = dilation(dapi, disk_dilation)

    save_image(
        dapi,
        args.output[0],
        reader.physical_pixel_sizes.Y,
        reader.physical_pixel_sizes.X,
        asuint=True,
    )

    lb, num = label(otsu, return_num=True)
    print("Before: ", num)

    new_mask = np.zeros(otsu.shape)
    for j in range(1, num + 1):
        if np.count_nonzero(dapi[lb == j]) > 0:
            new_mask[lb == j] = 1.0

    image_processed = new_mask.astype(np.uint8)

    lb, num = label(image_processed, return_num=True)
    print("After: ", num)

    save_image(
        image_processed,
        args.output[1],
        reader.physical_pixel_sizes.Y,
        reader.physical_pixel_sizes.X,
        asuint=True,
    )
