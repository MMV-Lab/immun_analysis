# cell area
# average branch length

rule cell_area:
    input:
        expand(
            join(
                OUTPUT, "{{TARGETS}}" + "_" + "{{MARKERS}}" + "_registered", "{image}"
            ),
            image=image,
        ),
    output:
        join(OUTPUT, "{TARGETS}" + "_" + "{MARKERS}" + "_cell_area", "cell_area_all.xlsx"),
        join(OUTPUT, "{TARGETS}" + "_" + "{MARKERS}" + "_cell_area", "cell_area_stats.xlsx"),
    script:
        "../scripts/3_cell_area.py"
