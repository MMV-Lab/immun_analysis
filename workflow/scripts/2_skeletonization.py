import warnings

warnings.simplefilter(action="ignore", category=RuntimeWarning)
from skimage.morphology import skeletonize, label
from utils import (
    binarize,
    get_reader,
    get_image_data,
    prune_short_branches,
    save_image,
)


############   Main   ############
if __name__ == "__main__":
    reader = get_reader(snakemake.input[0])
    img = get_image_data(reader, snakemake.wildcards.TARGETS)

    img_labeled = label(img, connectivity=2)

    img_binary = binarize(img_labeled, asbool=True)

    img_skeleton = skeletonize(img_binary, method="lee")

    img_skeleton_binary = binarize(img_skeleton, asbool=False)

    img_skeleton_pruned = prune_short_branches(img_skeleton_binary)

    save_image(
        img_skeleton_pruned,
        snakemake.output[0],
        snakemake.wildcards.TARGETS,
        reader.physical_pixel_sizes.Y,
        reader.physical_pixel_sizes.X,
        asuint=True,
    )
