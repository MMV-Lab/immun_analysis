rule longest_path:
    input:
        expand(
            join(
                OUTPUT, "{{TARGETS}}" + "_" + "{{MARKERS}}" + "_sekeletonization", "{image}"
            ),
            image=image,
        ),
    output:
        join(OUTPUT, "{TARGETS}" + "_" + "{MARKERS}" + "_longest_path", "longest_path_all.xlsx"),
        join(OUTPUT, "{TARGETS}" + "_" + "{MARKERS}" + "_longest_path", "longest_path_stats.xlsx"),
    script:
        "../scripts/3_longest_path.py"
