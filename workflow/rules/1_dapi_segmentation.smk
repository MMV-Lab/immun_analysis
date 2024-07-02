rule dapi:
    input:
        join(INPUT, "{image}"),
    output:
        join(OUTPUT, DAPI, "{image}"),
        join(OUTPUT, DAPIDILATE, "{image}"),
    script:
        "../scripts/1_dapi_segmentation.py"
