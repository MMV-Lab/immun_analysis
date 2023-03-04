def get_input():
    input_list = []
    input_list.append(PATH + "result/" + "otsu-{images}" + ENDCD64)
    input_list.append(PATH + "input/" + "{images}" + ENDDAPI)
    return input_list


rule registration:
    input:
        get_input(),
    output:
        PATH + "result/" + "otsu-reg-{images}" + ENDCD64,
    shell:
        "python Snakemake/scripts/registration.py --input {input} --output {output}"
