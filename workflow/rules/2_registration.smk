rule registration:
    input:
        join(OUTPUT, "{TARGETS}" + "_" + "{MARKERS}" + "_segmented", "{image}"),
        join(OUTPUT, DAPI, "{image}"),
        join(OUTPUT, DAPIDILATE, "{image}"),
    output:
        join(OUTPUT, "{TARGETS}" + "_" + "{MARKERS}" + "_registered", "{image}"),
    script:
        "../scripts/2_registration.py"
