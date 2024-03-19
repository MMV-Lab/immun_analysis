import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.types import PhysicalPixelSizes
from skimage.morphology import remove_small_holes, remove_small_objects
from scipy.sparse import csr_matrix
import get_networkx_graph_from_array


def binarize(image: np.ndarray, asbool: bool = False) -> np.ndarray:
    """Convert an image of any dtype to a binary image.

    Parameters
    ----------
    image : np.ndarray
        Input image to be binarized
    asbool : bool, optional
        Convert binary image to boolean dtype. Defaults to False.

    Returns
    -------
    np.ndarray
        Binarized image, where all non-zero pixel values are set to 1
    """
    binary_image = np.where(image > 0, 1, 0).astype(np.uint8)
    if asbool:
        binary_image = binary_image.astype(np.bool_)
    return binary_image


def compute_sparse(array: np.ndarray) -> csr_matrix:
    """Returns a sparse matrix representation of the input array

    Parameters
    ----------
    array : np.ndarray
        data array

    Returns
    -------
    csr_matrix
        sparse matrix representation of the input array
    """
    cols = np.arange(array.size)
    data = array.ravel()
    row_indices = data
    shape = (array.max() + 1, array.size)
    return csr_matrix((cols, (row_indices, cols)), shape=shape)


def convert_physical_to_pixel_size(size, factor, dim="YX", rounded=True):
    """Convert user provided physical size inputs to pixel size.

    Parameters
    ----------
    size : float
        physical size
    factor : float
        conversion factor
    dim : str
        transform along these axes
    rounded : bool
        round resulting pixel size

    Returns
    -------
    float
        pixel size that matches the provided physical size
    """
    try:
        if dim == "Y":
            pixel_size = size / factor.Y
        elif dim == "X":
            pixel_size = size / factor.X
        elif dim == "YX":
            pixel_size = size / (factor.X * factor.Y)

        if rounded:
            pixel_size = int(round(pixel_size, 0))
    except ZeroDivisionError:
        pixel_size = 1
    return pixel_size


def convert_pixel_to_physical_size(size, factor, dim="YX", rounded=True):
    """Convert user provided pixel size inputs to  size.

    Parameters
    ----------
    size : float
        physical size
    factor : float
        conversion factor
    dim : str
        transform along these axes
    rounded : bool
        round resulting pixel size

    Returns
    -------
    float
        physical size that matches the provided pixel size
    """
    if dim == "Y":
        physical_size = size * factor.Y
    elif dim == "X":
        physical_size = size * factor.X
    elif dim == "YX":
        physical_size = size * (factor.X * factor.Y)

    if rounded:
        physical_size = int(round(physical_size, 0))
    if physical_size == 0:
        physical_size = 1
    return physical_size


# def get_coords(array):
#     matrix = compute_sparse(array)
#     return [np.unravel_index(row.data, array.shape) for row in matrix][1:]


def get_coords(array: np.ndarray):
    """Returns the coordinates of non-zero elements of the input array.

    Parameters
    ----------
    array : np.ndarray
        data array

    Returns
    -------
    list
        list of coordinates of the non-zero elements from the input array.
    """

    # Compute the sparse matrix representation of the input array
    matrix = compute_sparse(array)

    # Iterate over each row in the sparse matrix and
    # Convert the column indices of the elements back into their original coordinates
    coords = [np.unravel_index(row.data, array.shape) for row in matrix]

    # Exclude the first element which corresponds to the zero elements in the array
    return coords[1:]


# def get_image_data(reader: AICSImage, C = 0) -> np.ndarray:
#     """
#     Retrieves image data from a reader object.

#     Args:
#         reader (AICSImage): The reader object that contains the image data.
#         C (int, optional): The index of the channel to retrieve the image data from. Defaults to 0.

#     Returns:
#         np.ndarray: The image data for the specified channel.
#     """
#     # if C != 0:
#     #    C = reader.channel_names.index(C)
#     image = reader.get_image_data("YX", C=str(C), T=0, Z=0)
#     return image


def get_image_data(reader, channel=False):
    """_summary_

    Parameters
    ----------
    reader : _type_
        _description_
    channel : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    if channel:
        image = reader.get_image_data(
            "YX", C=reader.channel_names.index(channel), T=0, Z=0
        )
    else:
        image = reader.get_image_data("YX", C=0, T=0, Z=0)
    return image


def get_reader(path):
    """Returns reader for given image path.

    Parameters
    ----------
    path : str
        image path

    Returns
    -------
    reader
        reader for image file
    """
    return AICSImage(path)


def prune_short_branches(img):
    """_summary_

    Parameters
    ----------
    img : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    graph = get_networkx_graph_from_array.get_networkx_graph_from_array(img)
    degree_one_nodes = [node for node, degree in graph.degree() if degree == 1]
    degree_one_endpoints = []
    for node in degree_one_nodes:
        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 1 and graph.degree(neighbors[0]) > 2:
            degree_one_endpoints.append(node)
            # graph.remove_node(node)
    img[tuple(np.array(degree_one_endpoints).T)] = False
    return img


def remove_lo(img: np.ndarray, max_size: int, connectivity: int) -> np.ndarray:
    """Efficiently removes large foreground objects from a binary image.

    Parameters
    ----------
    img : np.ndarray
        A 2D binary image. 0 represents background, 1 represents foreground.
    max_size : int
        The maximum size (number of pixels) of objects to keep in the image. Objects
        larger than this size will be removed.
    connectivity : int
        The connectivity of objects. Only objects with this level of connection will be
        considered for removal. Valid options are 1 (4-connected) or 2 (8-connected).

    Returns
    -------
    np.ndarray
        The modified binary image with large objects removed
    """
    return img ^ remove_small_objects(img, max_size, connectivity=connectivity)


def remove_sh(img: np.ndarray, area_threshold: int, connectivity: int) -> np.ndarray:
    """
    Removes contiguous holes smaller than the specified size from a binary image.

    Parameters
    ----------
    img : numpy.ndarray
        A 2D binary image represented as a numpy array. 0 represents background, 1 represents foreground.
    min_size : int
        The minimum size (number of pixels) of holes to be removed.
    connectivity : int
        The connectivity of the objects. Only objects connected to this degree will be considered for removal.
        Usually 1 (4-connected) or 2 (8-connected).

    Returns
    -------
    numpy.ndarray
        The modified binary image with small holes removed.
    """
    return remove_small_holes(
        img, area_threshold=area_threshold, connectivity=connectivity
    )


def remove_so(img: np.ndarray, min_size: int, connectivity: int) -> np.ndarray:
    """Removes small objects from a binary image based on their size and connectivity.

    Parameters
    ----------
    img : np.ndarray
        A 2D binary image represented as a numpy array. 0 represents background, 1 represents foreground.
    min_size : int
        The minimum size (number of pixels) of objects to remain in the image.
    connectivity : int
        The connectivity of the objects. Only objects connected to this degree will be considered for removal.
        Usually 1 (4-connected) or 2 (8-connected).

    Returns
    -------
    np.ndarray
        The modified binary image with small objects removed.
    """

    return remove_small_objects(img, min_size=min_size, connectivity=connectivity)


def save_image(
    img: np.ndarray,
    path: str,
    channel_name: str,
    ysize: float,
    xsize: float,
    asuint: bool = False,
) -> None:
    """Saves an image to a specified path in OME-TIFF format, with the option to convert the image to uint8 format and set non-zero pixels to 255.

    Parameters
    ----------
    img : np.ndarray
        The input image to be saved
    path : str
        The path where the image will be saved
    channel_name : str
        The channel name
    ysize : float
        The pixel size in the Y dimension
    xsize : float
        The pixel size in the X dimension
    asuint : bool, optional
        convert the image to uint8 format and non-zero pixels will be set to 255, by default False
    """
    if asuint:
        img = img.astype(np.uint8)
        img[img > 0] = 255
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)
    OmeTiffWriter.save(
        img,
        path,
        dim_order="CYX",
        channel_names=[channel_name],
        physical_pixel_sizes=[PhysicalPixelSizes(0, ysize, xsize)],
    )
