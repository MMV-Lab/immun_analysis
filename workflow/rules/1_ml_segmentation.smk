rule ml:
    input:
        join(OUTPUT, "{TARGETS}" + "_" + TOPHAT, "{image}"),
    output:
        join(OUTPUT, "{TARGETS}" + "_" + MLSEG, "{image}"),
    script:
        "../scripts/1_ml_segmentation.py"


# shell:
#    "python Snakemake/scripts/ml_segmentation.py {input} {output} '{CONFIG}'"
