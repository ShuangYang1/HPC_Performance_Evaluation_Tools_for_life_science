import re

factor = {
    "BWA": 27759.79112,
    "SPAdes": 81961.13074,
    "Bismark": 93644.88129,
    "STAR": 20202.64362,
    "Cellranger": 121445.2168,
    "GATK": 696156.6642,
}


def convert_to_sec(time_str):
    minutes, seconds = time_str.split("m")
    seconds = seconds.replace("s", "")
    total_seconds = int(minutes) * 60 + float(seconds)
    return total_seconds


def cal_score(software, filename):
    with open(filename, "r") as infile:
        content = infile.read()
        pattern = re.compile(r"real\t([0-9]+m[0-9]+.[0-9]+s)")
        time = pattern.search(content).group(1)
        sec = convert_to_sec(time)
        score = round((factor[software] / sec), 2)
        return score
