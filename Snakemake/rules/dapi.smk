rule dapi:
    input:
        join(INPUT, "{image}"),
    output:
        join(OUTPUT, DAPI, "{image}"),
    shell:
        "python Snakemake/scripts/dapi_segmentation.py {input} {output} -c {CONFIG}"
