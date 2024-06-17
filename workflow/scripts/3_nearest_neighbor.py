import warnings
import numpy as np
import pandas as pd
from utils import (
    binarize,
    convert_physical_to_pixel_size,
    get_reader,
    get_image_data,
)

from skimage import measure
from sklearn.metrics.pairwise import euclidean_distances

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_mindist(
    distances: np.ndarray, unique: int = 0, verbose: bool = False
) -> np.ndarray:
    """
    Calculate the minimum distances for each row in the input array.

    Args:
        distances (numpy.ndarray): The 2D input array containing distances between objects.
        unique (int): Determines the behavior for handling unique distances.
            - 0: No uniqueness constraints.
            - 1: Remove object as a possible partner for its closest neighboring object.
            - 2: Remove object as a possible partner for all other objects.
        verbose (bool): If True, print additional information during the calculation.

    Returns:
        numpy.ndarray: An array containing the minimum distances for each row in the input array.
    """
    min_dists = np.zeros((distances.shape[0], 2))
    for idx in range(distances.shape[0]):
        # Mask zeros in the array to handle zero distances
        distances_maskedzero = np.ma.masked_equal(distances[idx], 0.0, copy=False)
        idy = np.argmin(distances_maskedzero)
        min_dists[idx] = (idy, distances[idx, idy])

        # Remove object as a possible partner for its closest neighboring object
        if unique == 1:
            distances[idy, idx] = 0.0
        # Remove object as a possible partner for all other objects
        elif unique == 2:
            distances[idx:, idx] = 0.0

        if verbose:
            print(distances[idy, idx])
    return min_dists


def get_neighbors(array: np.ndarray, limit: float) -> np.ndarray:
    """
    Returns a boolean array indicating which elements of the input array are bigger than 0 and less than or equal to the limit.

    Args:
        array: The input array, which can contain NaN values.
        limit: The limit value to compare the elements of the array with.

    Returns:
        A boolean array indicating which elements of the input array are not NaN and are less than the limit.
    """
    return np.logical_and(array > 0, array <= limit)


############   Main   ############
if __name__ == "__main__":
    distance_limits = snakemake.config["nearest_neighbor"]["distances"]
    distance_limit_names = ["distance_" + str(i) for i in distance_limits]
    distance_limits_stored = distance_limits
    # neighbors_stored = np.zeros((len(distance_limits), 0))
    sampleid = np.zeros(0)
    result_dfs = []
    for dir_img in snakemake.input:
        reader = get_reader(dir_img)
        img = binarize(get_image_data(reader))
        img_labeled = measure.label(img, connectivity=1)
        img_props = measure.regionprops(img_labeled)
        centroid_coords = np.array(
            [
                [
                    prop["centroid"][0] * reader.physical_pixel_sizes.Y,
                    prop["centroid"][1] * reader.physical_pixel_sizes.X,
                ]
                for prop in img_props
            ]
        )

        distances = euclidean_distances(centroid_coords)

        min_distances = get_mindist(distances, unique=1)
        sampleid = np.hstack(
            [
                sampleid,
                np.repeat(
                    np.array(dir_img.split(".")[-2].split("/")[-1], dtype=str),
                    min_distances.shape[0],
                ),
            ]
        )

        neighbors = np.vstack(
            [get_neighbors(distances, limit).sum(axis=1) for limit in distance_limits]
        )

        result_dfs.append(
            pd.DataFrame(
                np.vstack([centroid_coords.T, min_distances.T, neighbors]).T,
                columns=["y", "x", "target_index", "min_dist"] + distance_limit_names,
            ),
        )

    df_result = pd.concat(
        result_dfs,
        ignore_index=True,
    )
    df_result.insert(0, "sample", sampleid)
    df_result.to_excel(snakemake.output[0], index=False)

    df_sum = df_result.drop("x", axis=1).drop("y", axis=1).drop("target_index", axis=1)
    for col in df_sum.columns[1:]:
        df_sum[col] = pd.to_numeric(df_sum[col])
    get_metrics = {"min_dist": ["count", "mean", "std", "median"]}
    for dist in distance_limit_names:
        get_metrics[dist] = ["sum", "mean", "std", "median", "max"]
    df_sum = df_sum.groupby("sample", as_index=True).agg(get_metrics)
    df_sum.columns = ["_".join(a) for a in df_sum.columns.to_flat_index()]
    df_sum.rename({"min_dist_count": "cell_count"}, axis=1).to_excel(
        snakemake.output[1]
    )
