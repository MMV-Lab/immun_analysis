from cellpose import models
from os.path import join, isfile
from utils import get_config, get_image_data, get_reader, save_image
import argparse

MODELPATH = "Snakemake/model/dapi"
DEFAULTMODEL = "cellpose_residual_on_style_on_concatenation_off_DAPI_CELLPOSE_TRAIN_2023_01_10_11_34_21.773728"
CHAN = 0
CHAN2 = 0
############   Main   ############
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loads .tiff images and segments them with Cellpose"
    )
    # Positional Arguments
    parser.add_argument("input", type=str, help="Path to the image")
    parser.add_argument("output", type=str, help="Path to output & name")
    # Required Arguments
    parser.add_argument("-c", type=str, help="Path to config.yaml")
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, "|>", getattr(args, arg))

    config = get_config(args.c)

    if config["dapi_segmentation"]["model_name"] != "default":
        if isfile(join(MODELPATH, config["dapi_segmentation"]["model_name"])):
            model_path = join(MODELPATH, config["dapi_segmentation"]["model_name"])
            diameter = config["dapi_segmentation"]["cellpose"]["diameter"]
        else:
            raise NotImplementedError(
                "pass in either default or the name of a pretrained model"
            )
    else:
        model_path = join(MODELPATH, DEFAULTMODEL)
        diameter = 21.194

    model = models.CellposeModel(
        gpu=config["dapi_segmentation"]["gpu"], pretrained_model=model_path
    )

    diameter = model.diam_labels if diameter == 0 else diameter

    reader = get_reader(args.input)
    image = get_image_data(reader, config["registration"])

    masks, _, _ = model.eval([image], channels=[CHAN, CHAN2], diameter=diameter)

    save_image(masks[0], args.output, asuint=True)
