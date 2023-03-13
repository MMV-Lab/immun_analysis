from cellpose import io, models
from aicsimageio.writers import OmeTiffWriter
from pathlib import Path

input_dir =  Path('/mnt/eternus/share/immun_project/cut/')
savedir = Path('/mnt/eternus/share/immun_project/segmentation/step3_DAPI/')
model_path = '/mnt/eternus/users/Shuo/cellpose/DAPI_CELLPOSE_TRAIN/models/cellpose_residual_on_style_on_concatenation_off_DAPI_CELLPOSE_TRAIN_2023_01_10_11_34_21.773728'

# parameters
diameter=21.194
chan=0
chan2=0

# declare the cellpose model
model = models.CellposeModel(gpu=True, 
                             pretrained_model=model_path)

# check diameter
diameter = model.diam_labels if diameter==0 else diameter

# step 1: read the file path
filenames = sorted(input_dir.glob("*DAPI.tif"))
for fn in filenames:
    image = io.imread(fn)
    masks, _, _ = model.eval([image], channels=[chan, chan2], diameter=diameter)
    out_path = savedir / fn.name
    OmeTiffWriter.save(masks[0], out_path, dim_order="YX") 
