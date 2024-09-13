import warnings
import numpy as np
from os.path import join
from os import listdir
from mmv_im2im.configs.config_base import (
    ProgramConfig,
    parse_adaptor,
    configuration_validation,
)
from mmv_im2im import ProjectTester
from utils import (
    convert_physical_to_pixel_size,
    get_image_data,
    get_reader,
    remove_lo,
    remove_sh,
    remove_so,
    save_image,
)
import torch.cuda as cuda

from skimage.morphology import label

warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


MODELPATH = "workflow/model/ml"
DEFAULTMODEL = "default"


def load_model_path(model_name):
    """imports the machine learning model

    Parameters
    ----------
    model_name : str
        name of the model

    Returns
    -------
    str
        path to the model

    Raises
    ------
    NotImplementedError
        checkpoint file of the model was not found
    """
    if model_name != "default":
        model_path = join(MODELPATH, model_name)
        for file in listdir(model_path):
            if not file.endswith(".ckpt"):
                raise NotImplementedError(
                    "The checkpoint file was not found.\nPass in either 'default' or the name of a pretrained model and add the checkpoint"
                )
    else:
        model_path = join(MODELPATH, DEFAULTMODEL)
    return model_path


def load_interference_config(model_path):
    """loads and configures the interference config for the ml model.

    Parameters
    ----------
    model_path : str
        path to the model

    Returns
    -------
    dict
        ml interference config
    """
    for file in listdir(model_path):
        if file.endswith(".yaml"):
            yaml_file_path = join(model_path, file)
        else:
            raise NotImplementedError(
                "The model config file was not found.\nPass in either 'default' or the name of a pretrained model and add the config"
            )
    cfg = parse_adaptor(
        config_class=ProgramConfig,
        config=yaml_file_path,
    )

    for file in listdir(model_path):
        if file.endswith(".ckpt"):
            ckpt_file_path = join(model_path, file)
    cfg.model.checkpoint = ckpt_file_path
    cfg = configuration_validation(cfg)

    # Use CPU or GPU

    if cuda.is_available():
        precision = cfg.trainer.params["precision"]
        gpus = cfg.trainer.params["gpus"]
        accelerator = (
            {"accelerator": "gpu", "device": 1, "precision": precision, "gpus": gpus}
            if snakemake.config["ml_segmentation"]["gpu"]
            else {"accelerator": "cpu", "precision": precision}
        )
        cpu_only = not snakemake.config["ml_segmentation"]["gpu"]
    else:
        precision = cfg.trainer.params["precision"]
        accelerator = {"accelerator": "cpu", "precision": precision}
        cpu_only = True

    cfg.trainer.params = accelerator
    cfg.model.model_extra["cpu_only"] = cpu_only
    print(cfg)
    return cfg


if __name__ == "__main__":
    reader = get_reader(snakemake.input[0])
    img = get_image_data(reader, snakemake.wildcards.TARGETS)

    model_path = load_model_path(
        snakemake.config["ml_segmentation"][snakemake.wildcards.TARGETS]["model_name"]
    )

    cfg = load_interference_config(model_path)

    # define the executor for inference
    executor = ProjectTester(cfg)
    executor.setup_model()
    executor.setup_data_processing()

    # process the data, get the segmentation
    # run inference
    pred = executor.process_one_image(img)
    seg = (
        pred
        > snakemake.config["ml_segmentation"][snakemake.wildcards.TARGETS][
            "cutoff_value"
        ]
    )

    img_labeled = label(seg.astype(np.uint16), connectivity=2)

    if snakemake.config["ml_segmentation"]["large_objects"]["active"]:
        max_size = convert_physical_to_pixel_size(
            snakemake.config["ml_segmentation"]["large_objects"]["max_size"],
            reader.physical_pixel_sizes,
        )
        seg = remove_lo(
            seg > 0,
            max_size,
            snakemake.config["ml_segmentation"]["large_objects"]["connectivity"],
        )

    img_labeled = label(seg.astype(np.uint16), connectivity=2)

    if snakemake.config["ml_segmentation"]["small_holes"]["active"]:
        area_threshold = convert_physical_to_pixel_size(
            snakemake.config["ml_segmentation"]["small_holes"]["area_threshold"],
            reader.physical_pixel_sizes,
        )
        seg = remove_sh(
            img_labeled,
            area_threshold,
            snakemake.config["ml_segmentation"]["small_holes"]["connectivity"],
        )

    img_labeled = label(seg.astype(np.uint16), connectivity=2)

    if snakemake.config["ml_segmentation"]["small_objects"]["active"]:
        min_size = convert_physical_to_pixel_size(
            snakemake.config["ml_segmentation"]["small_objects"]["min_size"],
            reader.physical_pixel_sizes,
        )
        seg = remove_so(
            img_labeled,
            min_size,
            snakemake.config["ml_segmentation"]["small_objects"]["connectivity"],
        )

    if np.count_nonzero(seg) == 0:
        raise ValueError(
            f'No signals detected. Please check the channel "{snakemake.wildcards.TARGETS}" of the input image for signals or revise the used segmentation model.'
        )

    # save the result
    save_image(
        seg,
        snakemake.output[0],
        snakemake.wildcards.TARGETS,
        reader.physical_pixel_sizes.Y,
        reader.physical_pixel_sizes.X,
        asuint=True,
    )
