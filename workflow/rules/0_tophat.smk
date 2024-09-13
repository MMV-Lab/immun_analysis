rule tophat:
    input:
        join(INPUT, "{image}"),
    output:
        join(OUTPUT, "{TARGETS}" + "_" + TOPHAT, "{image}"),
    script:
        "../scripts/0_tophat.py"
