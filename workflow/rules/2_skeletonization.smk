rule skeletonization:
    input:
        join(OUTPUT, "{TARGETS}" + "_" + "{MARKERS}" + "_registered", "{image}"),
    output:
        join(OUTPUT, "{TARGETS}" + "_" + "{MARKERS}" + "_sekeletonization", "{image}"),
    script:
        "../scripts/2_skeletonization.py"


# shell:
#     "python Snakemake/scripts/skeletonization.py --input {input} --output {output} -c {CONFIG}"
