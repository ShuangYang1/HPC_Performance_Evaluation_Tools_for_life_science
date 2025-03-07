factor = {
    "SPONGE": 3.2144,
    "GROMACS": 0.2265,
    "AMBER": 0.2629,
    "DSDP": 245183.325,
}


def convert_to_sec(time_str):
    minutes, seconds = time_str.split("m")
    seconds = seconds.replace("s", "")
    total_seconds = int(minutes) * 60 + float(seconds)
    return total_seconds


def cal_score(software):
    if software == "SPONGE":
        with open("result/FastMode/SPONGE/mdinfo.txt", "r") as f:
            for line in f:
                if "Core Run Speed:" in line:
                    performance = float(line.split()[3])
        score = factor["SPONGE"] * performance
    elif software == "GROMACS":
        with open("result/FastMode/GROMACS/md.log", "r") as f:
            for line in f:
                if "Performance:" in line:
                    performance = float(line.split()[1])
        score = factor["GROMACS"] * performance
    elif software == "AMBER":
        with open("result/FastMode/AMBER/mdout", "r") as f:
            ready = False
            for line in f:
                if "Average timings for all steps:" in line:
                    ready = True
                elif "ns/day" in line and ready:
                    performance = float(line.split()[3])
                else:
                    continue
        score = factor["AMBER"] * performance
    elif software == "DSDP":
        with open("log/FastMode/DSDP/test.log", "r") as f:
            for line in f:
                if "Time:" in line:
                    time = float(line.split()[1])
        score = factor["DSDP"] / time
    return round(score, 2)
