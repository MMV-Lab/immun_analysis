import warnings
import numpy as np
from skimage.morphology.footprints import ellipse
from skimage.morphology import white_tophat
from utils import (
    convert_physical_to_pixel_size,
    get_image_data,
    get_reader,
    save_image,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


def contrast_stretching(img: np.ndarray, percentile: tuple) -> np.ndarray:
    """Perform contrast stretching on an input image by rescaling the pixel values to a specified range based on percentiles.

    Parameters
    ----------
    img : np.ndarray
        The input image to be contrast stretched.
    percentile : tuple
        A tuple specifying the lower and upper percentiles to use for rescaling the pixel values.

    Returns
    -------
    np.ndarray
        The contrast-stretched image with pixel values rescaled to the specified range.
    """
    lower_v = np.percentile(img, percentile[0])
    upper_v = np.percentile(img, percentile[1])

    img = np.clip(img, lower_v, upper_v)
    img = (img - lower_v) / (upper_v - lower_v)

    return img


############   Main   ############
if __name__ == "__main__":
    reader = get_reader(snakemake.input[0])
    img = get_image_data(reader, snakemake.wildcards.TARGETS)

    img_contrast = contrast_stretching(
        img, snakemake.config["top_hat"]["contrast"]["percentile"]
    )

    ellipse_size = snakemake.config["top_hat"]["tophat"]["element_size"]

    ellipse_size[0] = convert_physical_to_pixel_size(
        ellipse_size[0], reader.physical_pixel_sizes, dim="Y"
    )
    ellipse_size[1] = convert_physical_to_pixel_size(
        ellipse_size[1], reader.physical_pixel_sizes, dim="X"
    )
    img_tophat = white_tophat(
        img_contrast,
        ellipse(
            ellipse_size[0],
            ellipse_size[1],
        ),
    )

    save_image(
        img_tophat,
        snakemake.output[0],
        snakemake.wildcards.TARGETS,
        reader.physical_pixel_sizes.Y,
        reader.physical_pixel_sizes.X,
    )
