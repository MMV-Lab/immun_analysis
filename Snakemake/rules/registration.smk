def get_input():
    input_list = []
    input_list.append(join(OUTPUT, STEP1, "{image}" + TYPE))
    input_list.append(join(INPUT, "{image}" + REGISTRATION + TYPE))
    return input_list


rule registration:
    input:
        get_input(),
    output:
        join(OUTPUT, STEP2, "{image}" + TYPE),
    shell:
        "python Snakemake/scripts/registration.py --input {input} --output {output}"
