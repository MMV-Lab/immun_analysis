# immun_analysis
***
Immun_analysis is an open-source ...
## `Python environment local install`
Follow these instructions to perform the analysis without docker:
1. Install miniconda following the [installation instructions](https://docs.conda.io/en/latest/miniconda.html).
2. Start the Anaconda Prompt.
3. Install the required packages by calling ```conda env update -n base --file environment.yaml``` from the immun_analysis directory.
4. Insert the images into the input folder.
5. Adjust parameters in the config/config.yaml file.
6. Type `snakemake -c all --snakefile ./Snakemake/Snakefile`.
