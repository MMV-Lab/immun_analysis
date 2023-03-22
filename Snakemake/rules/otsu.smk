rule otsu:
    input:
        join(INPUT, "{image}"),
    output:
        join(OUTPUT, TOPHAT, "{image}"),
        join(OUTPUT, STEP1, "{image}"),
    shell:
        "python Snakemake/scripts/otsu_segmentation.py --input {input} --output {output} -c {CONFIG}"
