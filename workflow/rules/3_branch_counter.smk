rule branch_counter:
    input:
        expand(
            join(OUTPUT, "{{TARGETS}}" + "_" + "{{MARKERS}}" + "_sekeletonization", "{image}"), image=image
        ),
    output:
        join(OUTPUT, "{TARGETS}" + "_" + "{MARKERS}" + "_branch_counter", "branches_all.xlsx"),
        join(
            OUTPUT,
            "{TARGETS}" + "_" + "{MARKERS}" + "_branch_counter",
            "branches_stats.xlsx",
        ),
    script:
        "../scripts/3_branch_counter.py"
