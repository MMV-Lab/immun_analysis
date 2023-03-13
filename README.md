## immun_analysis

Use 

- configs
input_dir =  Path('/mnt/eternus/share/immun_project/cut/')  # the input DAPI raw tiff image folder path
savedir = Path('/mnt/eternus/share/immun_project/segmentation/step3_DAPI/') # the output DAPI mask tiff folder path
model_path = '/mnt/eternus/share/immun_project/segmentation/step3_DAPI/models/cellpose_residual_on_style_on_concatenation_off_DAPI_CELLPOSE_TRAIN_2023_01_10_11_34_21.773728'
# the pretrained cellpose model  

- install
pip install aicsimageio scikit-image cellpose

- usage
python From_raw_folder_cellpose_DAPI_new.py
