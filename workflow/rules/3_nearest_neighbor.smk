rule neighborhood:
    input:
        expand(
            join(OUTPUT, "{{TARGETS}}" + "_" + "{{MARKERS}}" + "_registered", "{image}"),
            image=image,
        ),
    output:
        join(OUTPUT, "{TARGETS}" + "_" + "{MARKERS}" + "_neighborhood", "neighborhood_all.xlsx"),
        join(OUTPUT, "{TARGETS}" + "_" + "{MARKERS}" + "_neighborhood", "neighborhood_stats.xlsx"),
    threads: 3
    script:
        "../scripts/3_nearest_neighbor.py"
