import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
from cellpose import models
from os.path import join, isfile
from utils import (
    convert_physical_to_pixel_size,
    get_image_data,
    get_reader,
    save_image,
)

from skimage.morphology import disk, binary_dilation
import numpy as np

import logging

logging.getLogger("porespy").setLevel(logging.WARNING)

MODELPATH = "workflow/model/dapi"
DEFAULTMODEL = "cellpose_residual_on_style_on_concatenation_off_DAPI_CELLPOSE_TRAIN_2023_01_10_11_34_21.773728"
CHAN = 0
CHAN2 = 0
############   Main   ############
if __name__ == "__main__":
    if snakemake.config["dapi_segmentation"]["model_name"] != "default":
        model_path = join(
            MODELPATH, snakemake.config["dapi_segmentation"]["model_name"]
        )
        if not isfile(model_path):
            raise NotImplementedError(
                "pass in either default or the name of a pretrained model"
            )
    else:
        model_path = join(MODELPATH, DEFAULTMODEL)

    diameter = (
        snakemake.config["dapi_segmentation"]["cellpose"]["diameter"]
        if snakemake.config["dapi_segmentation"]["model_name"] != "default"
        else 21.194
    )

    model = models.CellposeModel(
        gpu=snakemake.config["dapi_segmentation"]["gpu"], pretrained_model=model_path
    )

    diameter = model.diam_labels if diameter == 0 else diameter

    reader = get_reader(snakemake.input[0])
    img = get_image_data(reader, snakemake.config["registration"])

    masks, _, _ = model.eval([img], channels=[CHAN, CHAN2], diameter=diameter)
    img = masks[0]
    img[img != 0] = 255
    save_image(
        img,
        snakemake.output[0],
        snakemake.config["registration"],
        reader.physical_pixel_sizes.Y,
        reader.physical_pixel_sizes.X,
        asuint=True,
    )

    distance_filter_threshold = convert_physical_to_pixel_size(
        snakemake.config["colocalization"]["distance_filter_threshold"],
        reader.physical_pixel_sizes,
        dim="X",
    )

    disk_dilation = disk(distance_filter_threshold)

    img[img > 0] = 1
    mask_dilated = binary_dilation(img, disk_dilation)

    if np.count_nonzero(mask_dilated) == 0:
        raise ValueError(
            f"No signals detected. Please check the channel \"{snakemake.config['registration']}\" of the input image for signals or revise the used cellpose model."
        )

    save_image(
        mask_dilated,
        snakemake.output[1],
        snakemake.config["registration"],
        reader.physical_pixel_sizes.Y,
        reader.physical_pixel_sizes.X,
        asuint=True,
    )
