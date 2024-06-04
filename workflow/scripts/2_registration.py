import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

from utils import convert_physical_to_pixel_size, get_reader, get_image_data, save_image
import numpy as np
import pandas as pd
from skimage.morphology import label, disk, binary_dilation

from scipy.sparse import csr_matrix


def remove_irrelevant_obj(marker, reference):
    """
    Filters objects from a marker image based on their overlap with objects in a reference image.

    Args:
        marker (np.ndarray): A 2D array representing the marker image.
        reference (np.ndarray): A 2D array representing the reference image.

    Returns:
        np.ndarray: A 2D array of the same shape as `marker`, where each element is True
            if the corresponding pixel in `marker` overlaps with an object in `reference`,
            and False otherwise.
    """
    labeled_marker, num = label(marker, return_num=True)
    coords = get_coords(labeled_marker, False)
    registered_image = np.zeros(marker.shape)
    for lb in range(num):
        if np.any(reference[coords[lb][0], coords[lb][1]]):
            registered_image[coords[lb][0], coords[lb][1]] = True
    return registered_image


def get_coords(img: np.ndarray, asarray):
    """
    Returns the coordinates of the non-zero elements in the input img.

    Args:
        img: a numpy array representing the input data.

    Returns:
        A list of coordinates representing the non-zero elements in the input array.
    """

    # Compute the sparse matrix representation of the input array
    cols = np.arange(img.size)
    row_indices = img.ravel()
    shape = (img.max() + 1, img.size)
    img_matrix = csr_matrix((cols, (row_indices, cols)), shape=shape)

    # Iterate over each row in the sparse matrix and
    # Convert the column indices of the elements back into their original coordinates
    if asarray:  # required for intersection
        coords = [
            np.asarray(np.unravel_index(row.data, img.shape)).T for row in img_matrix
        ]
    else:  # required for dilation and simple registration
        coords = [np.unravel_index(row.data, img.shape) for row in img_matrix]

    # Exclude the first element which corresponds to the zero elements in the array
    return coords[1:]


def convert_array_to_sets(list_of_arrays):
    """
    Converts a list of NumPy arrays to a list of sets of tuples.

    Args:
        list_of_arrays: A list of NumPy arrays.

    Returns:
        A list (img) of sets (obj) of tuples (coords).
    """
    return [set(map(tuple, arr.tolist())) for arr in list_of_arrays]


def get_individual_dilated_dapi(dapi_coords, disk_dilation):
    """
    Dilates each object in sparse coordinate matrix separately to avoid overlaps and returns a DataFrame.

    Args:
        dapi_coords (list): A list of 2D NumPy arrays, where each array contains
            the coordinates (y, x) of an object in the DAPI image.
        disk_dilation (int): The size of the structuring element for dilation
            (radius of the disk).

    Returns:
        pd.DataFrame: A DataFrame containing the dilated coordinates for each
            object, with columns "label", "y", and "x".
    """
    distance_filter_threshold = 7
    stack = list()
    for idx, obj in enumerate(dapi_coords):
        y, x = obj[0], obj[1]
        # Shift coordinates and create mask
        x_min, y_min = np.min(x), np.min(y)
        x_shifted = x - x_min + distance_filter_threshold
        y_shifted = y - y_min + distance_filter_threshold
        mask = np.full(
            (
                np.max(y_shifted) + distance_filter_threshold,
                np.max(x_shifted) + distance_filter_threshold,
            ),
            False,
        )
        mask[y_shifted, x_shifted] = True
        # Dilate mask and extract dilated coordinates
        dilated_mask = binary_dilation(mask, disk_dilation)
        dilated_coords = np.argwhere(dilated_mask)
        # Shift coordinates back
        dilated_coords[:, 0] += y_min - distance_filter_threshold
        dilated_coords[:, 1] += x_min - distance_filter_threshold
        valid_coords = dilated_coords[np.all(dilated_coords >= 0, axis=1)]
        valid_coords = valid_coords[np.all(valid_coords < marker_img.shape[0], axis=1)]
        # Add label and store coordinates
        stack.append(np.vstack([np.repeat(idx, valid_coords.shape[0]), valid_coords.T]))
    # Combine coordinates into a DataFrame
    dapi_dilated_coords = pd.DataFrame(
        np.concatenate(stack, axis=1).T, columns=["label", "y", "x"]
    )
    return dapi_dilated_coords


def convert_df_to_sets(df):
    """
    Converts a pandas DataFrame into a list of sets of tuples, efficiently leveraging groupby operations.

    Args:
        df (pd.DataFrame): The input DataFrame containing label and feature columns.

    Returns:
        list: A list of sets of tuples, where each set corresponds to a group in the DataFrame and each tuple combines feature values.
    """
    return (
        df.groupby("label")[["x", "y"]]
        .apply(lambda x: set(zip(x["y"], x["x"])))
        .to_list()
    )


def get_intersections(marker_coords, dapi_coords):
    """
    Calculates the intersections between sets of tuples in two lists.

    Args:
        marker_coords (list): A list of sets of tuples, where each tuple represents a coordinate.
        dapi_coords (list): A list of sets of tuples, where each tuple represents a coordinate.

    Returns:
        list: An array, where each cell represents the intersection sizes
              between the corresponding sets in `coords` and `dapi`.
    """
    intersections = np.zeros((len(dapi_coords), len(marker_coords)))
    for i, obj_dapi in enumerate(dapi_coords):
        temp_intersections = list()
        for obj_marker in marker_coords:
            temp_set = obj_dapi.intersection(obj_marker)
            intersection_size = len(temp_set)
            temp_intersections.append(intersection_size)
        intersections[i, :] = temp_intersections
    return intersections


def get_non_zero_coords_array(arr):
    """
    Creates a 2D array from an array with row and column indices of non-zero cells along with their values.

    Args:
        arr: An array with numerical values.

    Returns:
        A 2D array with the following structure:
        [
            [row1_index, col1_index, row1_col1_value],
            [row2_index, col2_index, row2_col2_value],
            ...,
            [rowM_index, colN_index, rowM_colN_value]
        ]
    """
    # Find non-zero indices and values using vectorized operations
    non_zero_indices = np.nonzero(arr)
    non_zero_values = arr[non_zero_indices]
    # Combine information into a single array
    non_zero_array = np.column_stack(
        (non_zero_indices[1], non_zero_indices[0], non_zero_values)
    )
    # Sort the array based on the third column (values)
    non_zero_array = non_zero_array[non_zero_array[:, 2].argsort()[::-1]]
    return non_zero_array


def remove_duplicates_inorder(arr):
    """Removes duplicate rows from a NumPy array based on order, keeping only unqiue pairs.

    Args:
        arr (np.array): The NumPy array to process.

    Returns:
        np.ndarray: The NumPy array with duplicates removed.
    """
    visited_dapi = set()
    visited_marker = set()

    def remove_duplicate(row):
        """
        Removes duplicate rows from a NumPy array based on previous appearances.

        Args:
            row (np.ndarray): A single row from the NumPy array.

        Returns:
            np.ndarray: The modified row, replaced with NaNs if it's deemed a duplicate,
                    or the original row otherwise.
        """
        dapi_coord = row[0]
        marker_coord = row[1]
        if dapi_coord in visited_dapi or marker_coord in visited_marker:
            row = np.full(row.shape, np.nan, row.dtype)
        else:
            visited_dapi.add(dapi_coord)
            visited_marker.add(marker_coord)
        return row

    masked_array = np.apply_along_axis(remove_duplicate, 1, arr)
    non_duplicate_array = masked_array[~np.isnan(masked_array).all(axis=1)]
    return non_duplicate_array


def get_registered_marker(marker_coords, non_duplicate_array, img_shape):
    registered_marker = np.zeros(img_shape)
    for obj in range(non_duplicate_array.shape[0]):
        for coord in marker_coords[int(non_duplicate_array[obj, 0])]:
            registered_marker[coord[0], coord[1]] = 1
    return registered_marker


############   Main   ############
if __name__ == "__main__":
    reader = get_reader(snakemake.input[0])
    marker_img = get_image_data(reader, snakemake.wildcards.TARGETS)
    dapi_img = get_image_data(get_reader(snakemake.input[1]), "DAPI")
    dapi_img = dapi_img > 0
    dapi_dilated = get_image_data(get_reader(snakemake.input[2]), "DAPI")
    registered_img = remove_irrelevant_obj(marker_img > 0, dapi_dilated)
    labeled_registered_marker = label(registered_img > 0)
    marker_coords = get_coords(labeled_registered_marker, True)
    marker_coords = convert_array_to_sets(marker_coords)

    try:
        distance_filter_threshold = convert_physical_to_pixel_size(
            snakemake.config["colocalization"]["distance_filter_threshold"],
            reader.physical_pixel_sizes,
            dim="X",
        )
    except ZeroDivisionError:
        distance_filter_threshold = 1
    disk_dilation = disk(distance_filter_threshold)
    marker_dilated = binary_dilation(registered_img.copy(), disk_dilation)
    registered_dapi = remove_irrelevant_obj(dapi_img, marker_dilated)
    labeled_dapi, num = label(registered_dapi > 0, return_num=True)
    dapi_coords = get_coords(labeled_dapi, False)
    dapi_dilated_coords = get_individual_dilated_dapi(dapi_coords, disk_dilation)
    dapi_coords_sets = convert_df_to_sets(dapi_dilated_coords)

    intersections = get_intersections(marker_coords, dapi_coords_sets)
    non_zero_array = get_non_zero_coords_array(intersections)
    non_duplicate_array = remove_duplicates_inorder(non_zero_array)

    registered_marker = get_registered_marker(
        marker_coords, non_duplicate_array, marker_img.shape
    )

    if np.count_nonzero(registered_marker) == 0:
        raise ValueError(f"No overlapping signals detected.")

    save_image(
        registered_marker,
        snakemake.output[0],
        snakemake.wildcards.TARGETS,
        reader.physical_pixel_sizes.Y,
        reader.physical_pixel_sizes.X,
        asuint=True,
    )
