rule otsu:
    input:
        join(INPUT, "{image}" + TARGET + TYPE),
    output:
        join(OUTPUT, STEP1, "{image}" + TYPE),
    shell:
        "python Snakemake/scripts/otsu_segmentation.py {input} {output} -c {CONFIG}"
