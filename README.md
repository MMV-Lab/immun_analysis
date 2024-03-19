# immun_analysis
***
Immun_analysis is an open-source ...
## `Docker Version`

Follow these instructions to perform the analysis:
1. Install Docker for your operating system from [here](https://docs.docker.com/get-docker/).
2. Start Docker.
3. Build the Docker image by calling ```docker build -t immunanalysis .``` from the immun_analysis directory.
4. Insert the CD64 und DAPI images into the input folder.
5. Adjust parameters in the config.yaml file.
6. Run the analysis by calling `docker run -v "path/to/immun_analysis/:/home/user/immun_analysis/" immunanalysis`

## `Local Version (without Docker)`
Follow these instructions to perform the analysis without docker:
1. Install Anaconda following the [installation instructions](https://docs.conda.io/en/latest/miniconda.html).
2. Start the Anaconda Prompt.
3. Install the required packages by calling ```conda env update -n base --file environment.yaml``` from the immun_analysis directory.
4. Insert the CD64 und DAPI images into the input folder.
5. Adjust parameters in the config.yaml file.
6. Type `snakemake -c all --snakefile ./Snakemake/Snakefile`.

docker run -v "C:\Users\Devon\VSProjects\immun_analysis:/home/user/immun_analysis/" immunanalysis

***

# CD64_Seg
  Use this code to get segmentation from CD64 image after tophat. 

## `Configs`

### the input tophat tiff image folder path  
  IM_input_dir =  Path('/mnt/eternus/share/immun_project/segmentation/step1_tophat/')
### the output CD64 segment tiff folder path    
  seg_save_dir = Path('/mnt/eternus/share/immun_project/segmentation/step5_CD64/') 
### the config path
  config_path = "/mnt/eternus/share/immun_project/training_data/inference_semanticseg_2d.yaml"
### the checkpoint path
  checkpoint_path= "/mnt/eternus/share/immun_project/training_data/v3/best.ckpt"
### the threshold for size filter 
  size_filter_threshold = 45
### the cutoff value for segmentation
  cutoff_value = 0.1
### only use CPU or use GPU 
use_gpu = True

## `Usage`
  python CD64_inference.py
