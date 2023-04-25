import os
import numpy as np

from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from mmv_im2im.configs.config_base import ProgramConfig, parse_adaptor, configuration_validation
from mmv_im2im import ProjectTester
from skimage import morphology

tophat_IM_input_dir =  Path('/mnt/eternus/share/immun_project/segmentation/step1_tophat/')
inference_input_savedir = Path('/mnt/eternus/share/immun_project/training_data/inference_input/')
if not os.path.exists(inference_input_savedir): os.makedirs(inference_input_savedir)

size_filter_threshold = 45

input_dir =  inference_input_savedir 
savedir = Path('/mnt/eternus/share/immun_project/segmentation/step5_CD64/')
if not os.path.exists(savedir): os.makedirs(savedir)

config_path = "/mnt/eternus/share/immun_project/training_data/inference_semanticseg_2d.yaml"

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



# Size_Filter main function
def Size_Filter(IM):
    IM = IM.astype(np.uint8) 
    try:
        size_filter_IM = morphology.remove_small_objects( IM>0,size_filter_threshold )
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()

    size_filter_IM = size_filter_IM.astype(np.float64)
    # lb, num = morphology.label(size_filter_IM, return_num=True)
    return size_filter_IM

# load the inference configuration
cfg = parse_adaptor(config_class=ProgramConfig, config = config_path)
cfg = configuration_validation(cfg)

# define the executor for inference
executor = ProjectTester(cfg)
executor.setup_model()
executor.setup_data_processing()

# get the data, run inference, run size filter, and save the result
filenames = sorted(input_dir.glob("*.tiff"))
for fn in filenames:
    print(fn)
    img = AICSImage(fn).get_image_data("YX", Z=0, C=0, T=0)
    seg = executor.process_one_image(img)
    size_filter_seg = Size_Filter(seg)
    out_path = savedir / fn.name
    OmeTiffWriter.save(size_filter_seg, out_path, dim_orders="YX")






