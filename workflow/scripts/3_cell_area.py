import networkx as nx
import get_networkx_graph_from_array
import numpy as np
import pandas as pd
from skimage import measure
from skimage.morphology import label
from utils import (
    binarize,
    get_reader,
    get_image_data,
)

if __name__ == "__main__":
    df_result = pd.DataFrame([], columns=["y", "x", "area", "sample"])
    for dir_img in snakemake.input:
        reader = get_reader(dir_img)
        img = binarize(get_image_data(reader, snakemake.wildcards.TARGETS), asbool=True)
        labeled_img = label(img)
        img_props = measure.regionprops(labeled_img)
        arr_area = np.array(
            [
                [prop["centroid"][0], prop["centroid"][1], prop["area"]]
                for prop in img_props
            ]
        )
        arr_area[:, 0] *= reader.physical_pixel_sizes.Y
        arr_area[:, 1] *= reader.physical_pixel_sizes.X
        arr_area[:, 2] *= reader.physical_pixel_sizes.Y
        arr_area[:, 2] *= reader.physical_pixel_sizes.X
        arr_sample = np.repeat(
            np.array(dir_img.split(".")[-2].split("/")[-1], dtype=str),
            arr_area.shape[0],
        )
        df_result = pd.concat(
            [
                df_result,
                pd.DataFrame(
                    np.vstack([arr_area.T, arr_sample]).T,
                    columns=["y", "x", "area", "sample"],
                ),
            ],
            axis=0,
        )
    df_result.to_excel(snakemake.output[0])
    df_sum = df_result.drop("x", axis=1).drop("y", axis=1)
    df_sum["area"] = pd.to_numeric(df_sum["area"])
    df_sum = (
        df_sum.groupby("sample", as_index=False)
        .agg(["count", "sum", "mean", "std", "sem", "median", "max"])
        .rename({"count": "cell_count"}, axis=1)
    )
    df_sum.columns = [a[1] for a in df_sum.columns.to_flat_index()]
    df_sum.to_excel(snakemake.output[1])
