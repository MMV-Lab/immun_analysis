# immun_analysis
***
Immun_analysis is an open-soure ...
## Docker Version

Follow these instructions to perform the analysis:
1. Install Docker for your operating system from [here](https://docs.docker.com/get-docker/).
2. Start Docker.
3. Build the Docker image by calling ```docker build -t immunanalysis .``` from the immun_analysis directory.
4. Insert the CD64 und DAPI images into the input folder.
5. Adjust parameters in the config.yaml file.
6. Run the analysis by calling `docker run -v "path/to/immun_analysis:/home/user/immun_analysis/" immunanalysis`

## Local Version (without Docker)
Follow these instructions to perform the analysis without docker:
1. Install Anaconda following the [installation instructions](https://docs.conda.io/en/latest/miniconda.html).
2. Start the Anaconda Prompt.
3. Install the required packages by calling ```conda env update -n base --file environment.yaml``` from the immun_analysis directory.
4. Insert the CD64 und DAPI images into the input folder.
5. Adjust parameters in the config.yaml file.
6. Type `snakemake -c all --snakefile ./Snakemake/Snakefile`.


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
