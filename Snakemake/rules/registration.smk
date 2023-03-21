def get_input():
    input_list = []
    input_list.append(join(OUTPUT, STEP1, "{image}"))
    input_list.append(join(OUTPUT, DAPI, "{image}"))
    return input_list


rule registration:
    input:
        get_input(),
    output:
        join(OUTPUT, DAPIDILATE, "{image}"),
        join(OUTPUT, STEP2, "{image}"),
    shell:
        "python Snakemake/scripts/registration.py --input {input} --output {output} -c {CONFIG}"
