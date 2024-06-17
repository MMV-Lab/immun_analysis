rule tophat:
    input:
        join(INPUT, "{image}"),
    output:
        join(OUTPUT, "{TARGETS}" + "_" + TOPHAT, "{image}"),
    script:
        "../scripts/0_tophat.py"


# shell:
#     "python Snakemake/scripts/0_otsu_segmentation.py --input {input} --output {output} -c {CONFIG}"
