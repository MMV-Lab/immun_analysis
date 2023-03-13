# immun_analysis
  Use this code to get cell mask from DAPI raw image. 

## `Configs`
### the input DAPI raw tiff image folder path  
  input_dir =  Path('/mnt/eternus/share/immun_project/cut/')  
### the output DAPI mask tiff folder path    
  savedir = Path('/mnt/eternus/share/immun_project/segmentation/step3_DAPI/') 
### the pretrained cellpose model
  model_path  = '/mnt/eternus/share/immun_project/segmentation/step3_DAPI/models/cellpose_residual_on_style_on_concatenation_off_DAPI_CELLPOSE_TRAIN_2023_01_10_11_34_21.773728' 

## `Dependents`

  pip install aicsimageio scikit-image cellpose

## `Usage`

  python Get_DAPI_Segmentation.py
