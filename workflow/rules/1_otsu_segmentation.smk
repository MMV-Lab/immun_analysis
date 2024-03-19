rule otsu:
    input:
        join(OUTPUT, "{TARGETS}" + "_" + TOPHAT, "{image}"),
    output:
        join(OUTPUT, "{TARGETS}" + "_" + OTSUSEG, "{image}"),
    script:
        "../scripts/1_otsu_segmentation.py"
