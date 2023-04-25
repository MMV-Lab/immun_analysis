# immun_analysis
  Use this code to get segmentation from CD64 image after tophat. 

## `Configs`

### the input tophat tiff image folder path  
  tophat_IM_input_dir =  Path('/mnt/eternus/share/immun_project/segmentation/step1_tophat/')
### the output folder path as inference input folder path 
  inference_input_savedir = Path('/mnt/eternus/share/immun_project/training_data/inference_input/')
### the threshold for size filter 
  size_filter_threshold = 45
### the output CD64 segment tiff folder path    
  savedir = Path('/mnt/eternus/share/immun_project/segmentation/step5_CD64/') 
### the config path
  config_path = "/mnt/eternus/share/immun_project/training_data/inference_semanticseg_2d.yaml"

## `Dependents`
  pip install aicsimageio scikit-image mmv_im2im os numpy pathlib

## `Usage`
  python CD64_inference.py
