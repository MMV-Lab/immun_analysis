from skimage import measure, io, img_as_bool
from aicsimageio import AICSImage
import argparse


############   Functions   ############
def get_image(dir):
    return AICSImage(dir).get_image_data("YX")


def get_binary_img(image):
    return img_as_bool(image / 255)


############   Main   ############
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loads .tiff images and segments them with OTSU-Threshold"
    )
    # Positional Arguments
    parser.add_argument(
        "--input", type=str, nargs="+", help="Paths to the otsu and dapi image"
    )
    parser.add_argument(
        "--output", type=str, help="Path to store the registered images"
    )

    args = parser.parse_args()
    for arg in vars(args):
        print(arg, "|>", getattr(args, arg))

    otsu = get_binary_img(get_image(args.input[0]))
    dapi = get_binary_img(get_image(args.input[1]))

    x1_labels = measure.label(otsu, connectivity=2)
    x1_regionprops = measure.regionprops(x1_labels)

    object_coords = [obj["coords"] for obj in x1_regionprops]

    for obj in object_coords:
        count = 0
        for coords in obj:
            if dapi[coords[0], coords[1]] == 1:
                count = count + 1
        percent = count / len(obj)
        if percent == 0:
            for coords in obj:
                otsu[coords[0], coords[1]] = 0

    io.imsave(
        args.output,
        img_as_bool(otsu),
        plugin="pil",
        optimize=True,
        bits=1,
    )
