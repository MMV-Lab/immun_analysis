rule otsu:
    input:
        PATH + "input/" + "{images}" + ENDCD64,
    output:
        PATH + "result/" + "otsu-{images}" + ENDCD64,
    shell:
        "python Snakemake/scripts/otsu_segmentation.py --input {input} --output {output}"
