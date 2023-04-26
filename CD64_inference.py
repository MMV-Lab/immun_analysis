import numpy as np

from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from mmv_im2im.configs.config_base import (ProgramConfig,parse_adaptor,configuration_validation,)
from mmv_im2im import ProjectTester
from skimage.morphology import remove_small_objects

# Configs
IM_input_dir = Path("/mnt/eternus/share/immun_project/segmentation/step1_tophat/")
seg_save_dir = Path("/mnt/eternus/share/immun_project/segmentation/step5_CD64/")
config_path = ("/mnt/eternus/share/immun_project/training_data/inference_semanticseg_2d.yaml")
checkpoint_path = "/mnt/eternus/share/immun_project/training_data/v3/best.ckpt"

size_filter_threshold = 45
cutoff_value = 0.1
use_gpu = True

seg_save_dir.mkdir(parents=True, exist_ok=True)

# load the inference configuration
cfg = parse_adaptor(config_class=ProgramConfig, config=config_path)
cfg.model.checkpoint = checkpoint_path
cfg = configuration_validation(cfg)

# Use CPU or GPU
if use_gpu:
    cfg.trainer.params = {"accelerator": "gpu", "device": 1}
    cfg.model.model_extra["cpu_only"] = False
else:
    cfg.trainer.params = {"accelerator": "cpu"}
    cfg.model.model_extra["cpu_only"] = True

# define the executor for inference
executor = ProjectTester(cfg)
executor.setup_model()
executor.setup_data_processing()

# process the data, get the segmentation
filenames = sorted(IM_input_dir.glob("*.tif"))
for fn in filenames:
    # get the data
    img = AICSImage(fn).get_image_data("YX", Z=0, C=0, T=0)
    # run inference
    pred = executor.process_one_image(img)
    seg = pred > cutoff_value
    # run size filter
    size_filter_seg = remove_small_objects(seg > 0, size_filter_threshold).astype(
        np.uint8
    )
    size_filter_seg[size_filter_seg > 0] = 1
    # save the result
    out_path = seg_save_dir / fn.name
    OmeTiffWriter.save(size_filter_seg, out_path, dim_orders="YX")
