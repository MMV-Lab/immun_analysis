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
6. Run the analysis by calling `docker run -v "path/to/immun_analysis:/home/user/immune_analysis/" immunanalysis`

## Local Version (without Docker)
Follow these instructions to perform the analysis without docker:
1. Install Anaconda following the [installation instructions](https://docs.conda.io/en/latest/miniconda.html).
2. Start the Anaconda Prompt.
3. Install the required packages by calling ```conda env update -n base --file environment.yaml``` from the immun_analysis directory.
4. Insert the CD64 und DAPI images into the input folder.
5. Adjust parameters in the config.yaml file.
6. Type `snakemake -c all --snakefile ./Snakemake/Snakefile`.
