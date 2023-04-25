import numpy as np

from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from mmv_im2im.configs.config_base import ProgramConfig, parse_adaptor, configuration_validation
from mmv_im2im import ProjectTester
from skimage import morphology

# Configs
tophat_IM_input_dir =  Path('/mnt/eternus/share/immun_project/segmentation/step1_tophat/')
inference_input_savedir = Path('/mnt/eternus/share/immun_project/training_data/inference_input/')
Seg_input_dir =  inference_input_savedir 
Seg_savedir = Path('/mnt/eternus/share/immun_project/segmentation/step5_CD64/')

config_path = "/mnt/eternus/share/immun_project/training_data/inference_semanticseg_2d.yaml"
checkpoint_path= "/mnt/eternus/share/immun_project/training_data/v3/best.ckpt"

size_filter_threshold = 45
cutoff_value = 0.1
use_gpu = False

if not inference_input_savedir.exists(): inference_input_savedir.makedirs()
if not Seg_savedir.exists(): Seg_savedir.makedirs()

# read the file path and collect original input images
def copy_image(input_dir,savedir):
    filenames = sorted(input_dir.glob("*.tif"))
    for fn in filenames:
        image = AICSImage(fn).get_image_data("YX", Z=0, C=0, T=0)
        out_path = savedir / fn.name.replace('.tif',".tiff" ,-1)
        print(out_path)
        image = image.astype(np.float32)
        OmeTiffWriter.save(image , out_path, dim_order="YX") 

copy_image(tophat_IM_input_dir,inference_input_savedir)


# load the inference configuration
cfg = parse_adaptor(config_class=ProgramConfig, config = config_path)
cfg = configuration_validation(cfg)
cfg.data.inference_input.dir = Seg_input_dir
cfg.data.inference_output.path = Seg_savedir
cfg.model.checkpoint = checkpoint_path

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
filenames = sorted(Seg_input_dir.glob("*.tiff"))
for fn in filenames:
    print(fn)
    img = AICSImage(fn).get_image_data("YX", Z=0, C=0, T=0)
    pred = executor.process_one_image(img)
    seg = pred > cutoff_value
    size_filter_seg = morphology.remove_small_objects(seg>0,size_filter_threshold).astype(np.uint8)
    size_filter_seg[size_filter_seg > 0] = 1
    out_path = Seg_savedir / fn.name
    OmeTiffWriter.save(size_filter_seg, out_path, dim_orders="YX")
