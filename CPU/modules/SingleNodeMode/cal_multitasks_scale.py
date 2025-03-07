def cal_multitasks_scale(cpus):
    factors = {}
    for t in range(1, cpus + 1):
        if cpus % t == 0:
            factors[t] = cpus // t
    threads = sorted(factors.keys())[:-1]
    parallelmode = [str(t) + "_" + str(factors[t]) for t in threads]
    return parallelmode


def cal_gatk_multitasks_scale(cpus, mem):
    parallel = min(cpus, int(mem / 1000 / 10))
    return ["1_" + str(parallel)]
