import networkx as nx
import get_networkx_graph_from_array
import numpy as np
import pandas as pd
import itertools as it
from utils import (
    binarize,
    convert_pixel_to_physical_size,
    get_reader,
    get_image_data,
)


def longest_path_from_subplot(subgraph):
    diameter_nodes = []
    longest_path = 0
    for nodes_combinations in it.combinations(nx.periphery(subgraph), 2):
        temp_diameter_nodes = nx.shortest_path(
            subgraph, source=nodes_combinations[0], target=nodes_combinations[1]
        )
        if len(temp_diameter_nodes) > len(diameter_nodes):
            path_length = np.sum(
                np.linalg.norm(
                    np.array(temp_diameter_nodes[:-1])
                    - np.array(temp_diameter_nodes[1:]),
                    ord=2,  # l2 norm (euclidean distance)
                    axis=1,  # along neighboring nodes only
                )
            )
            if path_length > longest_path:
                diameter_nodes = temp_diameter_nodes
                longest_path = path_length
    return longest_path


############   Main   ############
if __name__ == "__main__":
    df_result = pd.DataFrame([], columns=["y", "x", "longest_path", "sample"])
    for dir_img in snakemake.input:
        reader = get_reader(dir_img)
        img = binarize(get_image_data(reader, snakemake.wildcards.TARGETS))
        img_graph = get_networkx_graph_from_array.get_networkx_graph_from_array(img)
        arr_longest_path = np.zeros(shape=(0, 3))
        # Iterate over all connected components
        for component in nx.connected_components(img_graph):
            component_list = list(component)
            for node in component_list:
                if len(list(img_graph.neighbors(node))) in {1, 0}:
                    start_node = node
                    break
            if len(component_list) == 1:
                longest_path = 0
            else:
                subgraph = img_graph.subgraph(component)
                longest_path = longest_path_from_subplot(subgraph)
                longest_path = convert_pixel_to_physical_size(
                    longest_path, reader.physical_pixel_sizes, dim="X", rounded=False
                )

                # longest_path = nx.diameter(subgraph)

            start_node_pixel_size = [
                convert_pixel_to_physical_size(
                    i, reader.physical_pixel_sizes, dim=j, rounded=False
                )
                for i, j in zip(start_node, ("Y", "X"))
            ]
            arr_longest_path = np.vstack(
                [
                    arr_longest_path,
                    [start_node_pixel_size[0], start_node_pixel_size[1], longest_path],
                ]
            )
        arr_sample = np.repeat(
            np.array(dir_img.split(".")[-2].split("/")[-1], dtype=str),
            arr_longest_path.shape[0],
        )
        df_result = pd.concat(
            [
                df_result,
                pd.DataFrame(
                    np.vstack([arr_longest_path.T, arr_sample]).T,
                    columns=["y", "x", "longest_path", "sample"],
                ),
            ],
            axis=0,
        )
    df_result.to_excel(snakemake.output[0])
    df_sum = df_result.drop("x", axis=1).drop("y", axis=1)
    df_sum["longest_path"] = pd.to_numeric(df_sum["longest_path"])
    df_sum = (
        df_sum.groupby("sample", as_index=False)
        .agg(["count", "sum", "mean", "std", "sem", "median", "max"])
        .rename({"count": "cell_count"}, axis=1)
    )
    df_sum.columns = [a[1] for a in df_sum.columns.to_flat_index()]
    df_sum.to_excel(snakemake.output[1])
