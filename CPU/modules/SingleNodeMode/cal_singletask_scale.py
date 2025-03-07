def cal_singletask_scale(cpu_cores,software):
    threads = [1]
    if software=="GATK":
        linear_part = list(range(2, min(4, cpu_cores) + 1))
    else:
        linear_part = list(range(2, min(8, cpu_cores) + 1))
    threads.extend(linear_part)
    if software=="BWA":
        current = 4
    else:
        current = 8
    while current < cpu_cores:
        next_value = min(cpu_cores, current * 2)
        if next_value not in threads:
            threads.append(next_value)
        current = next_value
    if cpu_cores not in threads:
        threads.append(cpu_cores)
    return threads
