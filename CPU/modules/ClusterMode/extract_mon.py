import math
import os


def split_timestamp(start_timestamp, end_timestamp, step):
    total_duration = end_timestamp - start_timestamp
    num_bins = math.ceil(total_duration / step)
    time_bins = []
    for i in range(num_bins):
        bin_start = start_timestamp + i * step
        bin_end = min(bin_start + step, end_timestamp)
        time_bins.append((bin_start, bin_end - 1))
    return time_bins


def extract_mon(
    server_ip,
    server_port,
    node,
    instance,
    start_timestamp,
    end_timestamp,
    outdir,
    bin_num,
):
    query_instance = '\{instance="' + instance + '"\}'
    cpu_query_instance = '\{instance="' + instance + '",mode="idle"\}'
    extract_cpu = f"curl -G 'http://{server_ip}:{server_port}/api/v1/query_range?query=node_cpu_seconds_total{cpu_query_instance}&start={start_timestamp}&end={end_timestamp}&step=1s' > {outdir}/{node}_cpu_{bin_num}.json"
    extract_mem_total = f"curl -G 'http://{server_ip}:{server_port}/api/v1/query_range?query=node_memory_MemTotal_bytes{query_instance}&start={start_timestamp}&end={start_timestamp}&step=1s' > {outdir}/memTotal.json"
    # extract_mem_available = f"curl -G 'http://{server_ip}:{server_port}/api/v1/query_range?query=node_memory_MemAvailable_bytes{query_instance}&start={start_timestamp}&end={end_timestamp}&step=1s' > {outdir}/{node}_memAvailable_{bin_num}.json"
    extract_mem_free = f"curl -G 'http://{server_ip}:{server_port}/api/v1/query_range?query=node_memory_MemFree_bytes{query_instance}&start={start_timestamp}&end={end_timestamp}&step=1s' > {outdir}/{node}_memFree_{bin_num}.json"
    extract_mem_cached = f"curl -G 'http://{server_ip}:{server_port}/api/v1/query_range?query=node_memory_Cached_bytes{query_instance}&start={start_timestamp}&end={end_timestamp}&step=1s' > {outdir}/{node}_memCached_{bin_num}.json"
    extract_mem_buffers = f"curl -G 'http://{server_ip}:{server_port}/api/v1/query_range?query=node_memory_Buffers_bytes{query_instance}&start={start_timestamp}&end={end_timestamp}&step=1s' > {outdir}/{node}_memBuffers_{bin_num}.json"
    extract_read = f"curl -G 'http://{server_ip}:{server_port}/api/v1/query_range?query=node_infiniband_port_data_received_bytes_total{query_instance}&start={start_timestamp}&end={end_timestamp}&step=1s' > {outdir}/{node}_read_{bin_num}.json"
    extract_write = f"curl -G 'http://{server_ip}:{server_port}/api/v1/query_range?query=node_infiniband_port_data_transmitted_bytes_total{query_instance}&start={start_timestamp}&end={end_timestamp}&step=1s' > {outdir}/{node}_write_{bin_num}.json"
    extract_power = f"curl -G 'http://{server_ip}:{server_port}/api/v1/query_range?query=node_hwmon_power_average_watt{query_instance}&start={start_timestamp}&end={end_timestamp}&step=1s' > {outdir}/{node}_power_{bin_num}.json"

    for cmd in [
        extract_cpu,
        extract_mem_total,
        extract_mem_free,
        extract_mem_cached,
        extract_mem_buffers,
        extract_read,
        extract_write,
        extract_power,
    ]:
        os.system(cmd)


def auto_extract(
    node_exporter_port,
    server_ip,
    server_port,
    nodelist,
    start_timestamp,
    end_timestamp,
    outdir,
    step,
):
    for node in nodelist:
        instance = f"{node}:{node_exporter_port}"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        time_bins = split_timestamp(start_timestamp, end_timestamp, step)
        for i, (bin_start, bin_end) in enumerate(time_bins):
            extract_mon(
                server_ip,
                server_port,
                node,
                instance,
                bin_start,
                bin_end,
                outdir,
                i,
            )
    return time_bins


if __name__ == "__main__":
    node_exporter_port = 9100
    server_ip = "10.129.239.60"
    server_port = 9090
    nodelist = ["c05b26n04"]
    step = 11000
    start_timestamp = 1736836352
    end_timestamp = 1736836387
    outdir = "tmp/GROMACS_1_8/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for node in nodelist:
        instance = f"{node}:{node_exporter_port}"
        step = 11000
        time_bins = split_timestamp(start_timestamp, end_timestamp, step)
        print(time_bins)
        for i, (bin_start, bin_end) in enumerate(time_bins):
            extract_mon(
                server_ip,
                server_port,
                node,
                instance,
                bin_start,
                bin_end,
                outdir,
                i,
            )
