import warnings
from skimage.filters import threshold_otsu
import numpy as np
from utils import (
    convert_physical_to_pixel_size,
    get_image_data,
    get_reader,
    remove_so,
    save_image,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


############   Main   ############
if __name__ == "__main__":
    reader = get_reader(snakemake.input[0])
    img = get_image_data(reader, snakemake.wildcards.TARGETS)

    threshold = threshold_otsu(img)
    img_otsu = (
        img
        > threshold * snakemake.config["otsu_segmentation"]["otsu_threshold"]["factor"]
    )

    min_size = convert_physical_to_pixel_size(
        snakemake.config["otsu_segmentation"]["small_objects"]["min_size"],
        reader.physical_pixel_sizes,
    )

    img_otsu_so = remove_so(
        img_otsu,
        min_size,
        snakemake.config["otsu_segmentation"]["small_objects"]["connectivity"],
    )

    if np.count_nonzero(img_otsu_so) == 0:
        raise ValueError(
            f'No signals detected. Please check the channel "{snakemake.wildcards.TARGETS}" of the input image for signals.'
        )

    save_image(
        img_otsu_so,
        snakemake.output[0],
        snakemake.wildcards.TARGETS,
        reader.physical_pixel_sizes.Y,
        reader.physical_pixel_sizes.X,
        asuint=True,
    )
