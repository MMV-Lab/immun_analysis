from tracemalloc import start
from aicsimageio import AICSImage
import networkx as nx
import os
import numpy as np
import pandas as pd
from utils import (
    binarize,
    convert_pixel_to_physical_size,
    get_reader,
    get_image_data,
)
import get_networkx_graph_from_array


def count_endpoints_branchpoints_branches_in_component(graph, start_node):
    """Starts at a random endpoint start_node and calculates endpoints, branchpoints and branches
    for the graph representation of the skeletonized object in graph.

    Parameters
    ----------
    graph : nx.Graph
        connected component representation of a skeletonized object
    start_node : _type_
        random endpoint to start from

    Returns
    -------
    int
        number of endpoints, branchpoints and branches
    """
    endpoints = 0
    branchpoints = 0
    branches = 1
    visited = set()
    queue = [start_node]
    while queue:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            neighbors = list(graph.neighbors(current))
            num_of_neighbors = len(neighbors)
            if num_of_neighbors == 1:
                endpoints += 1
            elif num_of_neighbors >= 3:
                branchpoints += 1
                branches += num_of_neighbors - 1
            elif num_of_neighbors == 2:
                # detect circles
                count_visited_neighbors = 0
                for neighbor in neighbors:
                    if neighbor in visited:  # Does this work?
                        count_visited_neighbors += 1
                if count_visited_neighbors == num_of_neighbors:
                    branches -= 1

            queue.extend(neighbors)
    return endpoints, branchpoints, branches


############   Main   ############
if __name__ == "__main__":
    df_result = pd.DataFrame(
        [], columns=["y", "x", "endpoints", "branchpoints", "branches", "sample"]
    )
    for dir_img in snakemake.input:
        reader = get_reader(dir_img)
        img = binarize(get_image_data(reader, snakemake.wildcards.TARGETS))
        img_graph = get_networkx_graph_from_array.get_networkx_graph_from_array(img)
        arr_branches = np.zeros(shape=(0, 5))
        # Iterate over all connected components
        for component in nx.connected_components(img_graph):
            component_list = list(component)
            for node in component_list:
                if len(list(img_graph.neighbors(node))) in {1, 0}:
                    start_node = node
                    break
            if len(component_list) == 1:
                endpoints, branchpoints, branches = 0, 0, 0
            else:
                (
                    endpoints,
                    branchpoints,
                    branches,
                ) = count_endpoints_branchpoints_branches_in_component(
                    img_graph, start_node
                )

            start_node_pixel_size = [
                convert_pixel_to_physical_size(
                    i, reader.physical_pixel_sizes, dim=j, rounded=False
                )
                for i, j in zip(start_node, ("Y", "X"))
            ]
            arr_branches = np.vstack(
                [
                    arr_branches,
                    [
                        start_node_pixel_size[0],
                        start_node_pixel_size[1],
                        endpoints,
                        branchpoints,
                        branches,
                    ],
                ]
            )
        arr_sample = np.repeat(
            np.array(dir_img.split(".")[-2].split("/")[-1], dtype=str),
            arr_branches.shape[0],
        )
        df_result = pd.concat(
            [
                df_result,
                pd.DataFrame(
                    np.vstack([arr_branches.T, arr_sample]).T,
                    columns=[
                        "y",
                        "x",
                        "endpoints",
                        "branchpoints",
                        "branches",
                        "sample",
                    ],
                ),
            ],
            axis=0,
        )
    df_result.to_excel(snakemake.output[0])
    df_sum = df_result.drop("x", axis=1).drop("y", axis=1)
    for col in df_sum.columns[:-1]:
        df_sum[col] = pd.to_numeric(df_sum[col])
    get_metrics = {
        "endpoints": ["count", "sum", "mean", "std", "sem", "median", "max"],
        "branchpoints": ["sum", "mean", "std", "sem", "median", "max"],
        "branches": ["sum", "mean", "std", "sem", "median", "max"],
    }
    df_sum = df_sum.groupby("sample", as_index=False).agg(get_metrics)
    df_sum.columns = [
        "_".join(a) if a[0] != "sample" else a[0]
        for a in df_sum.columns.to_flat_index()
    ]
    df_sum.rename({"endpoints_count": "cell_count"}, axis=1).to_excel(
        snakemake.output[1]
    )
