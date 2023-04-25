import numpy as np
import os

from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from mmv_im2im.configs.config_base import ProgramConfig, parse_adaptor, configuration_validation
from mmv_im2im import ProjectTester
from skimage.morphology import remove_small_objects

# Configs
IM_input_dir =  Path('/mnt/eternus/share/immun_project/segmentation/step1_tophat/')
seg_save_dir = Path('/mnt/eternus/share/immun_project/segmentation/step5_CD64/')

config_path = "/mnt/eternus/share/immun_project/training_data/inference_semanticseg_2d.yaml"
checkpoint_path= "/mnt/eternus/share/immun_project/training_data/v3/best.ckpt"

size_filter_threshold = 45
cutoff_value = 0.1
use_gpu = True

if not seg_save_dir .exists(): seg_save_dir .makedirs()

# load the inference configuration
cfg = parse_adaptor(config_class=ProgramConfig, config = config_path)
cfg.data.inference_input.dir = IM_input_dir 
cfg.data.inference_output.path = seg_save_dir 
cfg.model.checkpoint = checkpoint_path
cfg.data.inference_input.data_type = 'tif'
for name in os.listdir(cfg.data.inference_input.dir):
    if name.endswith('.tiff'):
        cfg.data.inference_input.data_type = 'tiff'
    elif name.endswith('.tif'):
        cfg.data.inference_input.data_type = 'tif'

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

# get the data, run inference, run size filter, and save the result
filenames = sorted(IM_input_dir.glob("*."+cfg.data.inference_input.data_type))
for fn in filenames:
    img = AICSImage(fn).get_image_data("YX", Z=0, C=0, T=0)
    pred = executor.process_one_image(img)
    seg = pred > cutoff_value
    size_filter_seg = remove_small_objects(seg>0,size_filter_threshold).astype(np.uint8)
    size_filter_seg[size_filter_seg > 0] = 1
    out_path = seg_save_dir / fn.name
    OmeTiffWriter.save(size_filter_seg, out_path, dim_orders="YX")
