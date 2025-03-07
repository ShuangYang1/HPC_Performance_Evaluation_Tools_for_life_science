import os
import re
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from modules.SingleNodeMode.extract_mon import auto_extract
from modules.SingleNodeMode.hpc_resource_radar import extract_and_draw

singletask_factor = {
    "SPONGE": 3.2144,
    "GROMACS": 0.2265,
    "AMBER": 0.2629,
    "DSDP": 245183.325,
}

multitasks_time_factor = {
    "SPONGE": 4318,
    "GROMACS": 748,
    "AMBER": 622,
    "DSDP": 95540,
}

multitasks_energy_factor = {
    "SPONGE": 14735847.19,
    "GROMACS": 2170683.796,
    "AMBER": 1670268.829,
    "DSDP": 201118688.5,
}

dataset = {
    "SPONGE": "https://doi.org/10.5281/zenodo.12200627 夏 义杰. Test Data for Comparing Simulation Speed between SPONGE and AMBER. Zenodo; 2024.",
    "GROMACS": "1WDN谷氨酰胺结合蛋白在水环境中的模型（61K原子），进行50,000 步（共 100 ps）的 NPT 模拟（恒压恒温），模拟温度为 300 K，压力为 1 atm",
    "AMBER": "1WDN谷氨酰胺结合蛋白在水环境中的模型（61K原子），进行50,000 步（共 100 ps）的 NPT 模拟（恒压恒温），模拟温度为 300 K，压力为 1 atm",
    "DSDP": "包含995对蛋白质和配体的结构文件的测试数据集",
}

font_path = "modules/SingleNodeMode/SimHei.ttf"
my_font = matplotlib.font_manager.FontProperties(fname=font_path)


def get_version(software):
    try:
        if software == "SPONGE":
            result = subprocess.run(
                ["SPONGE", "--version"], text=True, capture_output=True
            ).stdout
            return result.splitlines()[1].split()[0]
        elif software == "GROMACS":
            result = subprocess.run(
                ["gmx", "--version"], text=True, capture_output=True
            ).stdout
            for line in result.splitlines():
                if "GROMACS version:" in line:
                    return line.split()[-1]
        elif software == "AMBER":
            result = subprocess.run(
                ["pmemd", "--version"], text=True, capture_output=True
            ).stdout
            return result.split()[-1]
        elif software == "DSDP":
            return "v1.0"
    except subprocess.CalledProcessError as e:
        print(f"获取{software}版本信息时出错: {e.output.decode('utf-8')}")
        return None


def get_gpu_info(partition, node):
    try:
        result = subprocess.run(
            [
                "srun",
                f"--partition={partition}",
                f"--nodelist={node}",
                "--gres=gpu:1",
                "nvidia-smi",
                "--query-gpu=gpu_name,memory.total,memory.free,memory.used,count",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        gpu_info = result.stdout.strip().splitlines()
        gpu_details = gpu_info[0].split(", ")
        gpus = {}
        gpus["gpu_name"] = gpu_details[0]
        gpus["memory"] = float(gpu_details[1]) / 1024
        output = subprocess.run(
            ["scontrol", "show", "node", node],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        for line in output.splitlines():
            if "Gres=" in line:
                gres_field = line.split("Gres=")[-1].strip()
                if "gpu:" in gres_field:
                    gpu_count_info = gres_field.split("gpu:")[-1]
                    if "(" in gpu_count_info:
                        gpu_count = gpu_count_info.split("(")[0]
                        gpus["gpu_count"] = gpu_count
                        break
                else:
                    continue
            else:
                continue
        return gpus
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None


def get_cpu_info(partition, node):
    cpu_info = {}
    try:
        cpu_data = (
            subprocess.check_output(
                [
                    "srun",
                    f"--partition={partition}",
                    f"--nodelist={node}",
                    "cat",
                    "/proc/cpuinfo",
                ]
            )
            .decode("utf-8")
            .strip()
        )
        model_match = re.search(r"model name\s+:\s+(.*)", cpu_data)
        if model_match:
            cpu_info["model"] = model_match.group(1)
        cpu_info["cores"] = cpu_data.count("processor\t:")
    except Exception as e:
        print(f"获取CPU信息时出错: {e}")
    return cpu_info


def get_os_version(partition, node):
    try:
        os_data = (
            subprocess.check_output(
                [
                    "srun",
                    f"--partition={partition}",
                    f"--nodelist={node}",
                    "lsb_release",
                    "-d",
                ]
            )
            .decode("utf-8")
            .strip()
        )
        os_version = os_data.replace("Description:", "").strip()
    except Exception as e:
        print(f"获取OS信息时出错: {e}")
    return os_version


def get_kernel_version(partition, node):
    try:
        kernel_version = (
            subprocess.check_output(
                [
                    "srun",
                    f"--partition={partition}",
                    f"--nodelist={node}",
                    "uname",
                    "-r",
                ]
            )
            .decode("utf-8")
            .strip()
        )
    except Exception as e:
        print(f"获取内核信息时出错: {e}")
    return kernel_version


def get_slurm_version():
    try:
        slurm_version = (
            subprocess.check_output(
                [
                    "srun",
                    "--version",
                ]
            )
            .decode("utf-8")
            .strip()
        )
    except Exception as e:
        print(f"获取slurm信息时出错: {e}")
    return slurm_version


def get_memory_info(partition, node):
    memory_info = {}
    try:
        mem_data = (
            subprocess.check_output(
                [
                    "srun",
                    f"--partition={partition}",
                    f"--nodelist={node}",
                    "cat",
                    "/proc/meminfo",
                ]
            )
            .decode("utf-8")
            .strip()
        )
        total_match = re.search(r"MemTotal:\s+(\d+)\skB", mem_data)
        if total_match:
            total_memory_kb = int(total_match.group(1))
            memory_info["total"] = round(total_memory_kb / (1024**2), 2)
    except Exception as e:
        print(f"获取内存信息时出错: {e}")
    return memory_info


def get_filesystem_info():
    filesystem_info = {}
    try:
        script_path = os.path.abspath(__file__)
        df_output = subprocess.check_output(["df", "-P", script_path]).decode("utf-8")
        parts = df_output.split("\n")[1].split()
        filesystem_info["mount_point"] = parts[5]
        with open("/proc/mounts", "r") as f:
            for line in f:
                info = line.split()
                if info[1] == filesystem_info["mount_point"]:
                    filesystem_info["fs_type"] = info[2]
                    break
        filesystem_info["size"] = round(int(parts[1]) / (1024**2), 2)
        filesystem_info["used"] = round(int(parts[2]) / (1024**2), 2)
        filesystem_info["available"] = round(int(parts[3]) / (1024**2), 2)
        filesystem_info["use_percentage"] = parts[4]
    except Exception as e:
        print(f"获取文件系统容量信息时出错: {e}")
    return filesystem_info


def cal_speedup(gpu_list, time_list, gputime_list, performance_list):
    table = "| 使用的GPU数量 | 分子动力学模拟性能（ns/day） | 加速比 | 计算效率 | 计算用时（s） | GPU卡时（s） |\n| ---- | ---- | ---- | ---- | ---- | ---- |\n"
    for i in range(len(gpu_list)):
        table += f"| {gpu_list[i]} | {round(performance_list[i], 2)} | {round(performance_list[i] / performance_list[0], 2)} | {round(performance_list[i] / performance_list[0] / gpu_list[i] * 100, 2)}% | {round(time_list[i], 2)} | {round(gputime_list[i], 2)} |\n"
    return table


def cal_multitasks_time_score(software, taskcount, sec):
    score = round((multitasks_time_factor[software] * taskcount / sec), 2)
    return score


def cal_multitasks_energy_score(software, taskcount, energy):
    score = round((multitasks_energy_factor[software] * taskcount / energy), 2)
    return score


def multitasks_best_mode(modes, time_result):
    best_time = min(time_result)
    best_mode = modes[time_result.index(best_time)]
    return best_mode, best_time


def make_multitasks_linechart(modes, time_result, xlabel, outpath):
    best_mode, best_time = multitasks_best_mode(modes, time_result)
    plt.figure(figsize=(10, 5), dpi=1000)
    plt.plot(modes, time_result, marker="o")
    plt.xlabel(xlabel, fontproperties=my_font)
    plt.ylabel("Time(s)")
    plt.savefig(outpath)
    plt.close()
    return best_mode


def make_singletask_violin(software, singletask_gpus, time):
    path = "image/SingleNodeMode/" + software
    if not os.path.exists(path):
        os.makedirs(path)
    plt.figure(figsize=(10, 5), dpi=1000)
    violin_parts = plt.violinplot(time, showmeans=True, showmedians=True, widths=0.3)
    violin_parts["cmeans"].set_color("green")
    violin_parts["cmedians"].set_color("red")
    if singletask_gpus == 1:
        plt.title(
            f"{software} 单任务单卡{len(time)}次计算用时分布", fontproperties=my_font
        )
    else:
        plt.title(
            f"{software} 单任务{singletask_gpus}卡{len(time)}次计算用时分布",
            fontproperties=my_font,
        )
    plt.ylabel("Time(s)")
    plt.xticks([])
    violin_path = path + "/singletask_violin.png"
    plt.savefig(violin_path)
    plt.close()
    mean = np.mean(time)
    std = np.std(time)
    score = max(round((1 - (std / mean)) * 100, 2), 0)
    return violin_path, score


def make_score_bar(softwares, scores):
    bar_path = "image/SingleNodeMode/score_bar.png"
    y_pos = np.arange(len(softwares))
    plt.figure(figsize=(10, 5), dpi=1000)
    plt.barh(y_pos, scores, height=0.4, color="skyblue", align="center")
    plt.title("单节点深度测评软件得分", fontproperties=my_font)
    plt.xlim(0, max(max(scores) + 20, 110))
    plt.xlabel("Score")
    plt.yticks(y_pos, softwares)
    plt.axvline(x=100, color="r", linestyle="--", linewidth=1.5)
    for i, v in enumerate(scores):
        plt.text(v + 1, i, str(v), color="black", va="center")
    plt.savefig(bar_path)
    plt.close()
    return bar_path


def get_multitasks_timestamp(timestamp_dir):
    start_timestamps = []
    end_timestamps = []
    for filename in os.listdir(timestamp_dir):
        if filename.endswith(".start.timestamp"):
            with open(os.path.join(timestamp_dir, filename), "r") as f:
                start_timestamps.append(int(f.read().strip()))
        elif filename.endswith(".end.timestamp"):
            with open(os.path.join(timestamp_dir, filename), "r") as f:
                end_timestamps.append(int(f.read().strip()))
    start_timestamp = min(start_timestamps)
    end_timestamp = max(end_timestamps)
    return start_timestamp, end_timestamp


def cal_singletask_score(software, singletask_best_mode, repeat):
    perfs = []
    for repeat_id in range(1, repeat + 1):
        resultdir = f"result/SingleNodeMode/{software}/singletask/{singletask_best_mode}_{repeat_id}/"
        if software == "SPONGE":
            result_file = f"{resultdir}/mdinfo.txt"
            with open(result_file, "r") as f:
                for line in f:
                    if "Core Run Speed:" in line:
                        performance = float(line.split()[3])
            perfs.append(performance)
        elif software == "GROMACS":
            result_file = f"{resultdir}/md.log"
            with open(result_file, "r") as f:
                for line in f:
                    if "Performance:" in line:
                        performance = float(line.split()[1])
            perfs.append(performance)
        elif software == "AMBER":
            result_file = f"{resultdir}/mdout"
            with open(result_file, "r") as f:
                ready = False
                for line in f:
                    if "Average timings for all steps:" in line:
                        ready = True
                    elif "ns/day" in line and ready:
                        performance = float(line.split()[3])
                    else:
                        continue
            perfs.append(performance)
        elif software == "DSDP":
            result_file = f"log/SingleNodeMode/{software}/singletask/{singletask_best_mode}_{repeat_id}.log"
            with open(result_file, "r") as f:
                for line in f:
                    if "Time:" in line:
                        time = float(line.split()[1])
            perfs.append(time)
    if software == "DSDP":
        score = singletask_factor["DSDP"] / np.mean(perfs)
    else:
        score = np.mean(perfs) * singletask_factor[software]
    return round(score, 2)


def make_boxplot(data, xticks, xlabel, datatype, outpath):
    ylabel = {
        "time": "Time(s)",
        "cputime": "CPU Time(s)",
        "gputime": "GPU Time(s)",
        "performance": "Performance(ns/day)",
    }
    fig, ax = plt.subplots(figsize=(10, 5), dpi=1000)
    box = ax.boxplot(data, patch_artist=True, notch=True, showmeans=False)
    medians = [item.get_ydata()[0] for item in box["medians"]]
    positions = range(1, len(data) + 1)
    ax.plot(positions, medians, linestyle="-", color="orange", label="Median Line")
    ax.set_xticks(positions, xticks)
    ax.set_xlabel(xlabel, fontproperties=my_font)
    ax.set_ylabel(ylabel[datatype])
    plt.savefig(outpath)
    plt.close()
    return True


def extract_gromacs_singletask_result(gpus, cpus, repeat):
    modes = []
    for i in range(1, gpus + 1):
        if cpus % i == 0:
            cpus_per_task = cpus // i
            testmode = f"{i}_{cpus_per_task}"
            modes.append(testmode)
    time_result = [[] for _ in range(len(modes))]
    gputime_result = [[] for _ in range(len(modes))]
    performance_result = [[] for _ in range(len(modes))]
    for mode in modes:
        gpu_per_task = int(mode.split("_")[0])
        for repeat_id in range(1, repeat + 1):
            resultdir = f"result/SingleNodeMode/GROMACS/singletask/{mode}_{repeat_id}"
            with open(f"{resultdir}/md.log", "r") as f:
                for line in f:
                    if "Performance:" in line:
                        performance_result[modes.index(mode)].append(
                            float(line.split()[1])
                        )
                    elif "Time:" in line:
                        time_result[modes.index(mode)].append(float(line.split()[2]))
                        gputime_result[modes.index(mode)].append(
                            float(line.split()[2]) * gpu_per_task
                        )
                    else:
                        continue
    return modes, time_result, gputime_result, performance_result


def extract_amber_singletask_result(gpus, cpus, repeat):
    modes = list(range(1, gpus + 1))
    time_result = [[] for _ in range(len(modes))]
    gputime_result = [[] for _ in range(len(modes))]
    performance_result = [[] for _ in range(len(modes))]
    for mode in modes:
        gpu_per_task = mode
        cpu_per_task = cpus // mode
        for repeat_id in range(1, repeat + 1):
            resultdir = f"result/SingleNodeMode/AMBER/singletask/{gpu_per_task}_{cpu_per_task}_{repeat_id}"
            with open(f"{resultdir}/mdout", "r") as f:
                ready = False
                for line in f:
                    if "Average timings for all steps:" in line:
                        ready = True
                    elif "ns/day" in line and ready:
                        performance_result[modes.index(mode)].append(
                            float(line.split()[3])
                        )
                    elif "Elapsed(s)" in line and ready:
                        time_result[modes.index(mode)].append(float(line.split()[3]))
                        gputime_result[modes.index(mode)].append(
                            float(line.split()[3]) * gpu_per_task
                        )
                    else:
                        continue
    return modes, time_result, gputime_result, performance_result


def extract_gromacs_multitasks_result(gpus, cpus):
    modes = []
    for i in range(1, gpus + 1):
        if gpus % i == 0:
            parallel = gpus // i
            cpus_per_task = cpus // gpus
            testmode = f"{i}_{cpus_per_task}_{parallel}"
            modes.append(testmode)
    time_result = []
    for mode in modes:
        logdir = f"log/SingleNodeMode/GROMACS/multitasks/{mode}/"
        start_timestamp, end_timestamp = get_multitasks_timestamp(logdir)
        time_result.append(end_timestamp - start_timestamp)
    return modes, time_result


def extract_amber_multitasks_result(gpus, cpus):
    modes = []
    time_result = []
    for i in range(1, gpus + 1):
        if gpus % i == 0:
            parallel = gpus // i
            cpus_per_task = cpus // gpus
            testmode = f"{i}_{parallel}"
            modes.append(testmode)
            logdir = (
                f"log/SingleNodeMode/AMBER/multitasks/{i}_{cpus_per_task}_{parallel}/"
            )
            start_timestamp, end_timestamp = get_multitasks_timestamp(logdir)
            time_result.append(end_timestamp - start_timestamp)
    return modes, time_result


def sponge_report(
    content,
    multitasks_count,
    repeat,
    gpus,
    cpus,
    node_exporter_port,
    nvidia_gpu_exporter_port,
    server_ip,
    server_port,
    partition,
    node,
):
    software = "SPONGE"
    version = get_version(software)
    cpu_info = get_cpu_info(partition, node)
    memory_info = get_memory_info(partition, node)
    filesystem_info = get_filesystem_info()
    gpu_info = get_gpu_info(partition, node)
    os_version = get_os_version(partition, node)
    kernel_version = get_kernel_version(partition, node)
    slurm_version = get_slurm_version()
    image_path = f"image/SingleNodeMode/{software}/"
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # 单任务
    singletask_gpus = 1
    singletask_cpus_per_task = cpus
    singletask_best_mode = f"{singletask_gpus}_{singletask_cpus_per_task}"
    with open(
        f"log/SingleNodeMode/{software}/singletask/{singletask_best_mode}_1.start.timestamp",
        "r",
    ) as f:
        singletask_start_timestamp = int(f.read().strip())
    with open(
        f"log/SingleNodeMode/{software}/singletask/{singletask_best_mode}_1.end.timestamp",
        "r",
    ) as f:
        singletask_end_timestamp = int(f.read().strip())
    singletask_mondir = f"mon/SingleNodeMode/{software}/singletask/"
    step = 11000
    time_bins = auto_extract(
        node_exporter_port,
        nvidia_gpu_exporter_port,
        server_ip,
        server_port,
        node,
        singletask_start_timestamp,
        singletask_end_timestamp,
        singletask_mondir,
        step,
    )
    singletask_radar_path = f"image/SingleNodeMode/{software}/singletask_radar.png"
    singletask_radar_gpu_path = (
        f"image/SingleNodeMode/{software}/singletask_radar_gpu.png"
    )
    singletask_radar_split_path = (
        f"image/SingleNodeMode/{software}/singletask_radar_split.png"
    )
    (
        singletask_cpu_summary,
        singletask_mem_summary,
        singletask_read_summary,
        singletask_write_summary,
        singletask_power_summary,
        singletask_gpu_summary,
        singletask_gpu_mem_summary,
    ) = extract_and_draw(
        singletask_mondir,
        time_bins,
        cpus,
        gpus,
        singletask_radar_path,
        singletask_radar_gpu_path,
        singletask_radar_split_path,
        [software, "singletask", singletask_gpus],
    )
    singletask_repeat_time = []
    for repeat_id in range(1, repeat + 1):
        with open(
            f"result/SingleNodeMode/{software}/singletask/{singletask_best_mode}_{repeat_id}/mdinfo.txt",
            "r",
        ) as f:
            for line in f:
                if "Core Run Wall Time:" in line:
                    singletask_repeat_time.append(float(line.split()[4]))
    singletask_violin_path, stability_score = make_singletask_violin(
        software, singletask_gpus, singletask_repeat_time
    )

    # 多任务
    multitasks_best_mode = (
        f"{singletask_gpus}_{int(cpus / gpus)}_{int(gpus / singletask_gpus)}"
    )
    multitasks_start_timestamp, multitasks_end_timestamp = get_multitasks_timestamp(
        f"log/SingleNodeMode/{software}/multitasks/{multitasks_best_mode}"
    )
    multitasks_mondir = f"mon/SingleNodeMode/{software}/multitasks/"
    step = 11000
    time_bins = auto_extract(
        node_exporter_port,
        nvidia_gpu_exporter_port,
        server_ip,
        server_port,
        node,
        multitasks_start_timestamp,
        multitasks_end_timestamp,
        multitasks_mondir,
        step,
    )
    multitasks_radar_path = f"image/SingleNodeMode/{software}/multitasks_radar.png"
    multitasks_radar_gpu_path = (
        f"image/SingleNodeMode/{software}/multitasks_radar_gpu.png"
    )
    multitasks_radar_split_path = (
        f"image/SingleNodeMode/{software}/multitasks_radar_split.png"
    )
    (
        multitasks_cpu_summary,
        multitasks_mem_summary,
        multitasks_read_summary,
        multitasks_write_summary,
        multitasks_power_summary,
        multitasks_gpu_summary,
        multitasks_gpu_mem_summary,
    ) = extract_and_draw(
        multitasks_mondir,
        time_bins,
        cpus,
        gpus,
        multitasks_radar_path,
        multitasks_radar_gpu_path,
        multitasks_radar_split_path,
        [software, "multitasks", multitasks_best_mode],
    )

    singletask_score = cal_singletask_score(software, singletask_best_mode, repeat)
    multitasks_time_score = cal_multitasks_time_score(
        software,
        multitasks_count,
        multitasks_end_timestamp - multitasks_start_timestamp,
    )
    multitasks_energy_score = cal_multitasks_energy_score(
        software, multitasks_count, multitasks_power_summary["sum"]
    )
    total_score = round(
        np.mean(
            [
                singletask_score,
                multitasks_time_score,
                multitasks_energy_score,
            ]
        )
        * (stability_score / 100),
        2,
    )
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"## {software}\n"
    content += f"软件版本：{version}  \n"
    content += f"数据集：{dataset[software]}  \n"
    content += f"测试环境：GPU型号{gpu_info['gpu_name']}，GPU卡数{gpu_info['gpu_count']}，单卡显存{gpu_info['memory']}GB，CPU型号{cpu_info['model']}，CPU核心数{cpu_info['cores']}，内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']}，文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。操作系统版本{os_version}，内核版本{kernel_version}，作业调度系统{slurm_version}。  \n\n"
    content += "&nbsp;\n"

    content += "测评内容：  \n"
    content += "1. 单任务运行特征分析。记录软件计算时对GPU、CPU、内存、I/O读写带宽等资源的使用情况和功耗，输出从任务运行到结束的资源使用情况雷达图，帮助用户了解软件资源需求特征、分析被测节点配置是否存在瓶颈。\n"
    content += "2. 单任务计算用时稳定性分析。多次测试相同单任务的计算用时，统计计算用时分布，输出小提琴图。本指标是为了测试被测节点的软硬件环境是否存在较大的性能波动。\n"
    content += "3. 大批量任务运行特征分析。记录计算软件在大批量任务并行模式下的GPU、CPU、内存、I/O读写带宽等资源的使用情况和功耗，输出从任务运行到结束的资源使用情况雷达图，帮助用户了解软件在大批量任务并行模式下的资源需求特征、分析被测节点配置是否存在瓶颈。  \n\n"
    content += "&nbsp;\n"
    content += f"计算当前测试环境下软件性能得分。本测评根据单任务性能得分、计算用时稳定性得分、大批量任务计算用时得分、大批量任务计算能耗得分等四个细分指标的综合计算，得到单节点的{software}性能总分。  \n"
    content += "$$\n"
    content += "总得分=avg(单任务性能得分+大批量任务计算用时得分+大批量任务计算能耗得分)\\times\\frac{计算用时稳定性得分}{100}\n"
    content += "$$\n"
    content += "四个细分指标详述如下：  \n"
    content += "① 单任务性能得分：根据多次相同单任务的平均分子动力学模拟性能计算。  \n"
    content += "$$\n"
    content += "单任务性能得分={系数}\\times{分子动力学模拟性能}\n"
    content += "$$\n"
    content += "② 计算用时稳定性得分：根据多次相同单任务的用时分布计算。  \n"
    content += "$$\n"
    content += "单任务计算用时稳定性得分=max((1-\\frac{计算用时标准差}{计算用时均值})\\times100,0)\n"
    content += "$$\n"
    content += "③ 大批量任务计算用时得分：根据大批量任务并行的总用时计算。  \n"
    content += "$$\n"
    content += "大批量任务计算用时得分=系数\\times\\frac{任务数量}{计算总用时}\n"
    content += "$$\n"
    content += "④ 大批量任务计算能耗得分：根据大批量任务并行的总能耗计算。  \n"
    content += "$$\n"
    content += "大批量任务计算能耗得分=系数\\times\\frac{任务数量}{计算总能耗}\n"
    content += "$$\n"
    content += "注：系数以一个标准配置为基准（100分）来确定，测试环境的最终得分为相对于标准配置的相对得分。分子动力学模拟性能为分子动力学模拟软件常用的性能指标，单位为ns/day，表示在当前系统配置和环境以及输入参数条件下每天能够模拟的纳秒数，该值越大代表单位时间内能完成更长时间的分子动力学模拟，代表整体系统性能更好。  \n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 单任务运行特征分析：\n"
    content += f"![singletask_radar]({singletask_radar_path})  \n"
    content += "注：从顶端顺时针一周为软件单任务运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括CPU使用率、 内存使用量、I/O读带宽、I/O写带宽、功耗)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![singletask_radar]({singletask_radar_gpu_path})  \n"
    content += "注：从顶端顺时针一周为软件单任务运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括GPU使用率和全部显存使用量)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![singletask_radar_split]({singletask_radar_split_path})  \n"
    content += "注：从顶端顺时针一周为软件单任务运行期间各指标使用情况，从内向外不同层级分别展示一个指标的变化情况，每个层级内侧线为0，外侧线为该指标全程的最大值，指标范围见图例。  \n\n"
    content += "&nbsp;\n"
    content += f"GPU使用率峰值：{singletask_gpu_summary['max']}%，中位数{singletask_gpu_summary['median']}%，平均值{singletask_gpu_summary['mean']}%  \n"
    content += f"显存使用峰值：{singletask_gpu_mem_summary['max']}GB，中位数{singletask_gpu_mem_summary['median']}GB，平均值{singletask_gpu_mem_summary['mean']}GB  \n"
    content += f"CPU使用率峰值：{singletask_cpu_summary['max']}%，中位数{singletask_cpu_summary['median']}%，平均值{singletask_cpu_summary['mean']}%  \n"
    content += f"内存使用峰值：{singletask_mem_summary['max']}GB，中位数{singletask_mem_summary['median']}GB，平均值{singletask_mem_summary['mean']}GB  \n"
    if singletask_read_summary["max"] < 1024:
        content += f"I/O读峰值速率：{round(singletask_read_summary['max'], 2)}B/s，"
    elif (
        singletask_read_summary["max"] >= 1024
        and singletask_read_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O读峰值速率：{round(singletask_read_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        singletask_read_summary["max"] >= 1024 * 1024
        and singletask_read_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O读峰值速率：{round(singletask_read_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif singletask_read_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O读峰值速率：{round(singletask_read_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_read_summary["median"] < 1024:
        content += f"中位数{round(singletask_read_summary['median'], 2)}B/s，"
    elif (
        singletask_read_summary["median"] >= 1024
        and singletask_read_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(singletask_read_summary['median'] / 1024, 2)}KB/s，"
    elif (
        singletask_read_summary["median"] >= 1024 * 1024
        and singletask_read_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(singletask_read_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif singletask_read_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(singletask_read_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_read_summary["mean"] < 1024:
        content += f"平均值{round(singletask_read_summary['mean'], 2)}B/s  \n"
    elif (
        singletask_read_summary["mean"] >= 1024
        and singletask_read_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(singletask_read_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        singletask_read_summary["mean"] >= 1024 * 1024
        and singletask_read_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(singletask_read_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif singletask_read_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(singletask_read_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    if singletask_write_summary["max"] < 1024:
        content += f"I/O写峰值速率：{round(singletask_write_summary['max'], 2)}B/s，"
    elif (
        singletask_write_summary["max"] >= 1024
        and singletask_write_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O写峰值速率：{round(singletask_write_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        singletask_write_summary["max"] >= 1024 * 1024
        and singletask_write_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O写峰值速率：{round(singletask_write_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif singletask_write_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O写峰值速率：{round(singletask_write_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_write_summary["median"] < 1024:
        content += f"中位数{round(singletask_write_summary['median'], 2)}B/s，"
    elif (
        singletask_write_summary["median"] >= 1024
        and singletask_write_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(singletask_write_summary['median'] / 1024, 2)}KB/s，"
    elif (
        singletask_write_summary["median"] >= 1024 * 1024
        and singletask_write_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(singletask_write_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif singletask_write_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(singletask_write_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_write_summary["mean"] < 1024:
        content += f"平均值{round(singletask_write_summary['mean'], 2)}B/s  \n"
    elif (
        singletask_write_summary["mean"] >= 1024
        and singletask_write_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(singletask_write_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        singletask_write_summary["mean"] >= 1024 * 1024
        and singletask_write_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(singletask_write_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif singletask_write_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(singletask_write_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    # content += f"I/O读峰值速率：{singletask_read_summary['max']}GB，中位数{singletask_read_summary['median']}GB，平均值{singletask_read_summary['mean']}GB  \n"
    # content += f"I/O写峰值速率：{singletask_write_summary['max']}GB，中位数{singletask_write_summary['median']}GB，平均值{singletask_write_summary['mean']}GB  \n"
    content += f"功耗峰值：{singletask_power_summary['max']}W，中位数{singletask_power_summary['median']}W，平均值{singletask_power_summary['mean']}W  \n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 单任务计算用时稳定性分析：\n"
    content += f"相同算例测试{repeat}次，统计用时分布。  \n"
    content += f"![singletask_violin]({singletask_violin_path})  \n"
    content += "注：纵坐标为计算用时。小提琴图代表多次测试的计算用时分布，小提琴图越宽的地方代表在该范围内的数据分布密度越高，也就是说越多的数据点集中在这个区间。红线代表多次测试的计算用时的中位数，绿线代表平均值。  \n\n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 大批量任务运行特征分析：\n"
    content += f"![multitasks_radar]({multitasks_radar_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括CPU使用率、 内存使用量、I/O读带宽、I/O写带宽、功耗)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar]({multitasks_radar_gpu_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括GPU使用率和全部显存使用量)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar_split]({multitasks_radar_split_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间各指标使用情况，从内向外不同层级分别展示一个指标的变化情况，每个层级内侧线为0，外侧线为该指标全程的最大值，指标范围见图例。  \n\n"
    content += "&nbsp;\n"
    content += f"GPU使用率峰值：{multitasks_gpu_summary['max']}%，中位数{multitasks_gpu_summary['median']}%，平均值{multitasks_gpu_summary['mean']}%  \n"
    content += f"显存使用峰值：{multitasks_gpu_mem_summary['max']}GB，中位数{multitasks_gpu_mem_summary['median']}GB，平均值{multitasks_gpu_mem_summary['mean']}GB  \n"
    content += f"CPU使用率峰值：{multitasks_cpu_summary['max']}%，中位数{multitasks_cpu_summary['median']}%，平均值{multitasks_cpu_summary['mean']}%  \n"
    content += f"内存使用峰值：{multitasks_mem_summary['max']}GB，中位数{multitasks_mem_summary['median']}GB，平均值{multitasks_mem_summary['mean']}GB  \n"
    if multitasks_read_summary["max"] < 1024:
        content += f"I/O读峰值速率：{round(multitasks_read_summary['max'], 2)}B/s，"
    elif (
        multitasks_read_summary["max"] >= 1024
        and multitasks_read_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O读峰值速率：{round(multitasks_read_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        multitasks_read_summary["max"] >= 1024 * 1024
        and multitasks_read_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O读峰值速率：{round(multitasks_read_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif multitasks_read_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O读峰值速率：{round(multitasks_read_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_read_summary["median"] < 1024:
        content += f"中位数{round(multitasks_read_summary['median'], 2)}B/s，"
    elif (
        multitasks_read_summary["median"] >= 1024
        and multitasks_read_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(multitasks_read_summary['median'] / 1024, 2)}KB/s，"
    elif (
        multitasks_read_summary["median"] >= 1024 * 1024
        and multitasks_read_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(multitasks_read_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif multitasks_read_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(multitasks_read_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_read_summary["mean"] < 1024:
        content += f"平均值{round(multitasks_read_summary['mean'], 2)}B/s  \n"
    elif (
        multitasks_read_summary["mean"] >= 1024
        and multitasks_read_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(multitasks_read_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        multitasks_read_summary["mean"] >= 1024 * 1024
        and multitasks_read_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(multitasks_read_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif multitasks_read_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(multitasks_read_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    if multitasks_write_summary["max"] < 1024:
        content += f"I/O写峰值速率：{round(multitasks_write_summary['max'], 2)}B/s，"
    elif (
        multitasks_write_summary["max"] >= 1024
        and multitasks_write_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O写峰值速率：{round(multitasks_write_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        multitasks_write_summary["max"] >= 1024 * 1024
        and multitasks_write_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O写峰值速率：{round(multitasks_write_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif multitasks_write_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O写峰值速率：{round(multitasks_write_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_write_summary["median"] < 1024:
        content += f"中位数{round(multitasks_write_summary['median'], 2)}B/s，"
    elif (
        multitasks_write_summary["median"] >= 1024
        and multitasks_write_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(multitasks_write_summary['median'] / 1024, 2)}KB/s，"
    elif (
        multitasks_write_summary["median"] >= 1024 * 1024
        and multitasks_write_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(multitasks_write_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif multitasks_write_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(multitasks_write_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_write_summary["mean"] < 1024:
        content += f"平均值{round(multitasks_write_summary['mean'], 2)}B/s  \n"
    elif (
        multitasks_write_summary["mean"] >= 1024
        and multitasks_write_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(multitasks_write_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        multitasks_write_summary["mean"] >= 1024 * 1024
        and multitasks_write_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(multitasks_write_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif multitasks_write_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(multitasks_write_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    # content += f"I/O读峰值速率：{multitasks_read_summary['max']}GB，中位数{multitasks_read_summary['median']}GB，平均值{multitasks_read_summary['mean']}GB  \n"
    # content += f"I/O写峰值速率：{multitasks_write_summary['max']}GB，中位数{multitasks_write_summary['median']}GB，平均值{multitasks_write_summary['mean']}GB  \n"
    content += f"功耗峰值：{multitasks_power_summary['max']}W，中位数{multitasks_power_summary['median']}W，平均值{multitasks_power_summary['mean']}W  \n"
    content += "&nbsp;\n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 节点性能打分：\n"
    content += f"测试软件：{software}  \n"
    content += f"单任务性能得分：{singletask_score}  \n"
    content += f"单任务计算用时稳定性得分：{stability_score}  \n"
    content += f"大批量任务计算用时得分：{multitasks_time_score}  \n"
    content += f"大批量任务计算能耗得分：{multitasks_energy_score}  \n"
    content += f"性能综合得分：{total_score}  \n"
    return content, total_score


def gromacs_report(
    content,
    multitasks_count,
    repeat,
    gpus,
    cpus,
    node_exporter_port,
    nvidia_gpu_exporter_port,
    server_ip,
    server_port,
    partition,
    node,
):
    software = "GROMACS"
    version = get_version(software)
    cpu_info = get_cpu_info(partition, node)
    memory_info = get_memory_info(partition, node)
    filesystem_info = get_filesystem_info()
    gpu_info = get_gpu_info(partition, node)
    os_version = get_os_version(partition, node)
    kernel_version = get_kernel_version(partition, node)
    slurm_version = get_slurm_version()
    image_path = f"image/SingleNodeMode/{software}/"
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # 单任务
    singletask_gpus = gpus
    singletask_cpus_per_task = cpus // gpus
    singletask_best_mode = f"{singletask_gpus}_{singletask_cpus_per_task}"
    with open(
        f"log/SingleNodeMode/{software}/singletask/{singletask_best_mode}_1.start.timestamp",
        "r",
    ) as f:
        singletask_start_timestamp = int(f.read().strip())
    with open(
        f"log/SingleNodeMode/{software}/singletask/{singletask_best_mode}_1.end.timestamp",
        "r",
    ) as f:
        singletask_end_timestamp = int(f.read().strip())
    singletask_mondir = f"mon/SingleNodeMode/{software}/singletask/"
    step = 11000
    time_bins = auto_extract(
        node_exporter_port,
        nvidia_gpu_exporter_port,
        server_ip,
        server_port,
        node,
        singletask_start_timestamp,
        singletask_end_timestamp,
        singletask_mondir,
        step,
    )
    singletask_radar_path = f"image/SingleNodeMode/{software}/singletask_radar.png"
    singletask_radar_gpu_path = (
        f"image/SingleNodeMode/{software}/singletask_radar_gpu.png"
    )
    singletask_radar_split_path = (
        f"image/SingleNodeMode/{software}/singletask_radar_split.png"
    )
    (
        singletask_cpu_summary,
        singletask_mem_summary,
        singletask_read_summary,
        singletask_write_summary,
        singletask_power_summary,
        singletask_gpu_summary,
        singletask_gpu_mem_summary,
    ) = extract_and_draw(
        singletask_mondir,
        time_bins,
        cpus,
        gpus,
        singletask_radar_path,
        singletask_radar_gpu_path,
        singletask_radar_split_path,
        [software, "singletask", singletask_gpus],
    )
    singletask_repeat_time = []
    for repeat_id in range(1, repeat + 1):
        with open(
            f"result/SingleNodeMode/{software}/singletask/{singletask_best_mode}_{repeat_id}/md.log",
            "r",
        ) as f:
            for line in f:
                if "Time:" in line:
                    singletask_repeat_time.append(float(line.split()[2]))
    singletask_violin_path, stability_score = make_singletask_violin(
        software, singletask_gpus, singletask_repeat_time
    )

    (
        singletask_modes,
        singletask_time_result,
        singletask_gputime_result,
        singletask_performance_result,
    ) = extract_gromacs_singletask_result(gpus, cpus, repeat)
    singletask_time_boxplot_path = (
        f"image/SingleNodeMode/{software}/singletask_time_boxplot.png"
    )
    singletask_gputime_boxplot_path = (
        f"image/SingleNodeMode/{software}/singletask_gputime_boxplot.png"
    )
    singletask_performance_boxplot_path = (
        f"image/SingleNodeMode/{software}/singletask_performance_boxplot.png"
    )
    make_boxplot(
        singletask_time_result,
        singletask_modes,
        "进程数（GPU卡数）_线程数",
        "time",
        singletask_time_boxplot_path,
    )
    make_boxplot(
        singletask_gputime_result,
        singletask_modes,
        "进程数（GPU卡数）_线程数",
        "gputime",
        singletask_gputime_boxplot_path,
    )
    make_boxplot(
        singletask_performance_result,
        singletask_modes,
        "进程数（GPU卡数）_线程数",
        "performance",
        singletask_performance_boxplot_path,
    )
    singletask_gpu_list = [int(s.split("_")[0]) for s in singletask_modes]
    singletask_time_list = [np.median(s) for s in singletask_time_result]
    singletask_gputime_list = [np.median(s) for s in singletask_gputime_result]
    singletask_performance_list = [np.median(s) for s in singletask_performance_result]
    singletask_best_time_gpu = singletask_gpu_list[
        singletask_time_list.index(min(singletask_time_list))
    ]
    singletask_best_gputime_gpu = singletask_gpu_list[
        singletask_gputime_list.index(min(singletask_gputime_list))
    ]
    singletask_best_performance_gpu = singletask_gpu_list[
        singletask_performance_list.index(max(singletask_performance_list))
    ]

    # 多任务
    multitasks_modes, multitasks_time_result = extract_gromacs_multitasks_result(
        gpus, cpus
    )
    multitasks_time_linechart_path = (
        f"image/SingleNodeMode/{software}/multitasks_time_linechart.png"
    )
    multitasks_best_mode = make_multitasks_linechart(
        multitasks_modes,
        multitasks_time_result,
        "进程数（GPU卡数）_线程数_并行任务数",
        multitasks_time_linechart_path,
    )
    multitasks_start_timestamp, multitasks_end_timestamp = get_multitasks_timestamp(
        f"log/SingleNodeMode/{software}/multitasks/{multitasks_best_mode}"
    )
    multitasks_mondir = f"mon/SingleNodeMode/{software}/multitasks/"
    step = 11000
    time_bins = auto_extract(
        node_exporter_port,
        nvidia_gpu_exporter_port,
        server_ip,
        server_port,
        node,
        multitasks_start_timestamp,
        multitasks_end_timestamp,
        multitasks_mondir,
        step,
    )
    multitasks_radar_path = f"image/SingleNodeMode/{software}/multitasks_radar.png"
    multitasks_radar_gpu_path = (
        f"image/SingleNodeMode/{software}/multitasks_radar_gpu.png"
    )
    multitasks_radar_split_path = (
        f"image/SingleNodeMode/{software}/multitasks_radar_split.png"
    )
    (
        multitasks_cpu_summary,
        multitasks_mem_summary,
        multitasks_read_summary,
        multitasks_write_summary,
        multitasks_power_summary,
        multitasks_gpu_summary,
        multitasks_gpu_mem_summary,
    ) = extract_and_draw(
        multitasks_mondir,
        time_bins,
        cpus,
        gpus,
        multitasks_radar_path,
        multitasks_radar_gpu_path,
        multitasks_radar_split_path,
        [software, "multitasks", multitasks_best_mode],
    )

    # 计算得分
    singletask_score = cal_singletask_score(software, singletask_best_mode, repeat)
    multitasks_time_score = cal_multitasks_time_score(
        software,
        multitasks_count,
        multitasks_end_timestamp - multitasks_start_timestamp,
    )
    multitasks_energy_score = cal_multitasks_energy_score(
        software, multitasks_count, multitasks_power_summary["sum"]
    )
    total_score = round(
        np.mean(
            [
                singletask_score,
                multitasks_time_score,
                multitasks_energy_score,
            ]
        )
        * (stability_score / 100),
        2,
    )
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"## {software}\n"
    content += f"软件版本：{version}  \n"
    content += f"数据集：{dataset[software]}  \n"
    content += f"测试环境：GPU型号{gpu_info['gpu_name']}，GPU卡数{gpu_info['gpu_count']}，单卡显存{gpu_info['memory']}GB，CPU型号{cpu_info['model']}，CPU核心数{cpu_info['cores']}，内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']}，文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。操作系统版本{os_version}，内核版本{kernel_version}，作业调度系统{slurm_version}。  \n\n"
    content += "&nbsp;\n"

    content += "测评内容：  \n"
    content += "1. 单任务运行特征分析。记录软件计算时对GPU、CPU、内存、I/O读写带宽等资源的使用情况和功耗，输出从任务运行到结束的资源使用情况雷达图，帮助用户了解软件资源需求特征、分析被测节点配置是否存在瓶颈。\n"
    content += "2. 单任务并行效率分析。使用不同数量的GPU卡运行相同算例，统计计算用时、GPU卡时和分子动力学模拟性能，计算使用不同数量GPU卡的加速比和计算效率。通过本指标的评测，可以得知计算用时最少的单任务并行策略以及计算卡时最少的单任务并行策略。  \n\n"
    content += "$$\n"
    content += "\\begin{align}\n"
    content += (
        "加速比&=\\frac{该GPU卡数的分子动力学模拟性能}{单卡分子动力学模拟性能}\\\\\n"
    )
    content += "计算效率&=\\frac{加速比}{GPU卡数}*100\\%\n"
    content += "\\end{align}\n"
    content += "$$\n"
    content += "3. 单任务计算用时稳定性分析。多次测试相同单任务的计算用时，统计计算用时分布，输出小提琴图。本指标是为了测试被测节点的软硬件环境是否存在较大的性能波动。\n"
    content += "4. 大批量任务并行效率分析。使用不同GPU卡数×并行任务数的组合测试计算软件大批量任务并行计算。输出不同运行模式计算用时的折线图，以及在目前节点配置下大批量任务并行时最佳并行模式。通过本指标的评测，可以得知在何种并行策略下大批量任务的计算总用时最少。\n"
    content += "5. 大批量任务运行特征分析。记录计算软件在大批量任务并行模式下的GPU、CPU、内存、I/O读写带宽等资源的使用情况和功耗，输出从任务运行到结束的资源使用情况雷达图，帮助用户了解软件在大批量任务并行模式下的资源需求特征、分析被测节点配置是否存在瓶颈。  \n\n"
    content += "&nbsp;\n"
    content += f"计算当前测试环境下软件性能得分。本测评根据单任务性能得分、计算用时稳定性得分、大批量任务计算用时得分、大批量任务计算能耗得分等四个细分指标的综合计算，得到单节点的{software}性能总分。  \n"
    content += "$$\n"
    content += "总得分=avg(单任务性能得分+大批量任务计算用时得分+大批量任务计算能耗得分)\\times\\frac{计算用时稳定性得分}{100}\n"
    content += "$$\n"
    content += "四个细分指标详述如下：  \n"
    content += "① 单任务性能得分：根据多次相同单任务的平均分子动力学模拟性能计算。  \n"
    content += "$$\n"
    content += "单任务性能得分={系数}\\times{分子动力学模拟性能}\n"
    content += "$$\n"
    content += "② 计算用时稳定性得分：根据多次相同单任务的用时分布计算。  \n"
    content += "$$\n"
    content += "单任务计算用时稳定性得分=max((1-\\frac{计算用时标准差}{计算用时均值})\\times100,0)\n"
    content += "$$\n"
    content += "③ 大批量任务计算用时得分：根据大批量任务并行的总用时计算。  \n"
    content += "$$\n"
    content += "大批量任务计算用时得分=系数\\times\\frac{任务数量}{计算总用时}\n"
    content += "$$\n"
    content += "④ 大批量任务计算能耗得分：根据大批量任务并行的总能耗计算。  \n"
    content += "$$\n"
    content += "大批量任务计算能耗得分=系数\\times\\frac{任务数量}{计算总能耗}\n"
    content += "$$\n"
    content += "注：系数以一个标准配置为基准（100分）来确定，测试环境的最终得分为相对于标准配置的相对得分。分子动力学模拟性能为分子动力学模拟软件常用的性能指标，单位为ns/day，表示在当前系统配置和环境以及输入参数条件下每天能够模拟的纳秒数，该值越大代表单位时间内能完成更长时间的分子动力学模拟，代表整体系统性能更好。  \n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 单任务运行特征分析：\n"
    content += f"![singletask_radar]({singletask_radar_path})  \n"
    content += "注：从顶端顺时针一周为软件单任务运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括CPU使用率、 内存使用量、I/O读带宽、I/O写带宽、功耗)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![singletask_radar]({singletask_radar_gpu_path})  \n"
    content += "注：从顶端顺时针一周为软件单任务运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括GPU使用率和全部显存使用量)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![singletask_radar_split]({singletask_radar_split_path})  \n"
    content += "注：从顶端顺时针一周为软件单任务运行期间各指标使用情况，从内向外不同层级分别展示一个指标的变化情况，每个层级内侧线为0，外侧线为该指标全程的最大值，指标范围见图例。  \n\n"
    content += "&nbsp;\n"
    content += f"GPU使用率峰值：{singletask_gpu_summary['max']}%，中位数{singletask_gpu_summary['median']}%，平均值{singletask_gpu_summary['mean']}%  \n"
    content += f"显存使用峰值：{singletask_gpu_mem_summary['max']}GB，中位数{singletask_gpu_mem_summary['median']}GB，平均值{singletask_gpu_mem_summary['mean']}GB  \n"
    content += f"CPU使用率峰值：{singletask_cpu_summary['max']}%，中位数{singletask_cpu_summary['median']}%，平均值{singletask_cpu_summary['mean']}%  \n"
    content += f"内存使用峰值：{singletask_mem_summary['max']}GB，中位数{singletask_mem_summary['median']}GB，平均值{singletask_mem_summary['mean']}GB  \n"
    if singletask_read_summary["max"] < 1024:
        content += f"I/O读峰值速率：{round(singletask_read_summary['max'], 2)}B/s，"
    elif (
        singletask_read_summary["max"] >= 1024
        and singletask_read_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O读峰值速率：{round(singletask_read_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        singletask_read_summary["max"] >= 1024 * 1024
        and singletask_read_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O读峰值速率：{round(singletask_read_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif singletask_read_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O读峰值速率：{round(singletask_read_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_read_summary["median"] < 1024:
        content += f"中位数{round(singletask_read_summary['median'], 2)}B/s，"
    elif (
        singletask_read_summary["median"] >= 1024
        and singletask_read_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(singletask_read_summary['median'] / 1024, 2)}KB/s，"
    elif (
        singletask_read_summary["median"] >= 1024 * 1024
        and singletask_read_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(singletask_read_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif singletask_read_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(singletask_read_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_read_summary["mean"] < 1024:
        content += f"平均值{round(singletask_read_summary['mean'], 2)}B/s  \n"
    elif (
        singletask_read_summary["mean"] >= 1024
        and singletask_read_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(singletask_read_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        singletask_read_summary["mean"] >= 1024 * 1024
        and singletask_read_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(singletask_read_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif singletask_read_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(singletask_read_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    if singletask_write_summary["max"] < 1024:
        content += f"I/O写峰值速率：{round(singletask_write_summary['max'], 2)}B/s，"
    elif (
        singletask_write_summary["max"] >= 1024
        and singletask_write_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O写峰值速率：{round(singletask_write_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        singletask_write_summary["max"] >= 1024 * 1024
        and singletask_write_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O写峰值速率：{round(singletask_write_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif singletask_write_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O写峰值速率：{round(singletask_write_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_write_summary["median"] < 1024:
        content += f"中位数{round(singletask_write_summary['median'], 2)}B/s，"
    elif (
        singletask_write_summary["median"] >= 1024
        and singletask_write_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(singletask_write_summary['median'] / 1024, 2)}KB/s，"
    elif (
        singletask_write_summary["median"] >= 1024 * 1024
        and singletask_write_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(singletask_write_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif singletask_write_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(singletask_write_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_write_summary["mean"] < 1024:
        content += f"平均值{round(singletask_write_summary['mean'], 2)}B/s  \n"
    elif (
        singletask_write_summary["mean"] >= 1024
        and singletask_write_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(singletask_write_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        singletask_write_summary["mean"] >= 1024 * 1024
        and singletask_write_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(singletask_write_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif singletask_write_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(singletask_write_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    # content += f"I/O读峰值速率：{singletask_read_summary['max']}GB，中位数{singletask_read_summary['median']}GB，平均值{singletask_read_summary['mean']}GB  \n"
    # content += f"I/O写峰值速率：{singletask_write_summary['max']}GB，中位数{singletask_write_summary['median']}GB，平均值{singletask_write_summary['mean']}GB  \n"
    content += f"功耗峰值：{singletask_power_summary['max']}W，中位数{singletask_power_summary['median']}W，平均值{singletask_power_summary['mean']}W  \n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 单任务并行效率分析：\n"
    content += "计算用时：  \n"
    content += f"![singletask_time_boxplot]({singletask_time_boxplot_path})  \n"
    content += "注：横坐标为不同的进程数×线程数组合，纵坐标为任务运行时间，用于对比发现GROMACS单机多卡并行时计算用时最少的运行模式。进程数代表GROMACS开启的进程数，一个进程使用一张GPU卡，所以进程数可以理解为等价于使用的GPU卡数，因此进程数应<=GPU卡的总数。线程数代表每个进程使用线程数，通常使得进程数×线程数=CPU核心数能够充分利用到全部的CPU核心发挥最佳性能，因此选用进程数<=GPU卡数且能被CPU核心数整除的几种情况进行测试。  \n"
    content += f"**使用{singletask_best_time_gpu}张GPU卡并行计算时计算用时最少。**  \n"
    content += "&nbsp;\n"
    content += "计算所用GPU卡时：  \n"
    content += f"![singletask_gputime_boxplot]({singletask_gputime_boxplot_path})  \n"
    content += "注：横坐标为不同的进程数×线程数组合，纵坐标为任务计算所用的GPU卡时（GPU卡时=使用的GPU卡数×计算用时），用于对比发现GROMACS单机多卡并行时GPU卡时最少（GPU最经济）的运行模式。进程数代表GROMACS开启的进程数，一个进程使用一张GPU卡，所以进程数可以理解为等价于使用的GPU卡数，因此进程数应<=GPU卡的总数。线程数代表每个进程使用线程数，通常使得进程数×线程数=CPU核心数能够充分利用到全部的CPU核心发挥最佳性能，因此选用进程数<=GPU卡数且能被CPU核心数整除的几种情况进行测试。  \n"
    content += (
        f"**使用{singletask_best_gputime_gpu}张GPU卡并行计算时GPU卡时最少。**  \n"
    )
    content += "&nbsp;\n"
    content += "分子动力学模拟性能：  \n"
    content += (
        f"![singletask_performance_boxplot]({singletask_performance_boxplot_path})  \n"
    )
    content += "注：横坐标为不同的进程数×线程数组合，纵坐标为分子动力学模拟性能，用于对比发现GROMACS单机多卡并行时分子动力学模拟性能的运行模式。进程数代表GROMACS开启的进程数，一个进程使用一张GPU卡，所以进程数可以理解为等价于使用的GPU卡数，因此进程数应<=GPU卡的总数。线程数代表每个进程使用线程数，通常使得进程数×线程数=CPU核心数能够充分利用到全部的CPU核心发挥最佳性能，因此选用进程数<=GPU卡数且能被CPU核心数整除的几种情况进行测试。分子动力学模拟性能为分子动力学模拟软件常用的性能指标，单位为ns/day，表示在当前系统配置和环境以及输入参数条件下每天能够模拟的纳秒数，该值越大代表单位时间内能完成更长时间的分子动力学模拟，代表整体系统性能更好。  \n"
    content += f"**使用{singletask_best_performance_gpu}张GPU卡并行计算时分子动力学模拟性能最高。**  \n"
    content += "&nbsp;\n"
    content += cal_speedup(
        singletask_gpu_list,
        singletask_time_list,
        singletask_gputime_list,
        singletask_performance_list,
    )

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 单任务计算用时稳定性分析：\n"
    content += f"使用全部{singletask_best_mode.split('_')[0]}张GPU卡并行计算相同算例，测试{repeat}次，统计用时分布。  \n"
    content += f"![singletask_violin]({singletask_violin_path})  \n"
    content += "注：纵坐标为计算用时。小提琴图代表多次测试的计算用时分布，小提琴图越宽的地方代表在该范围内的数据分布密度越高，也就是说越多的数据点集中在这个区间。红线代表多次测试的计算用时的中位数，绿线代表平均值。  \n\n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 大批量任务并行效率分析：\n"
    content += f"![multitasks_time_linechart]({multitasks_time_linechart_path})  \n"
    content += "注：横坐标为不同的进程数×线程数×并行任务数组合，纵坐标为计算总用时，用于对比发现GROMACS大批量任务并行时计算总用时最少的运行模式。进程数代表GROMACS开启的进程数，一个进程使用一张GPU卡，所以进程数可以理解为等价于使用的GPU卡数，因此进程数应<=GPU卡的总数。并行任务数代表同时最多并行的任务数量，最大等于GPU卡数（每个任务使用一张GPU卡），最小为1（每个任务使用全部GPU卡，任务间串行）。线程数代表每个进程使用线程数，通常使得进程数×线程数×并行任务数=CPU核心数能够充分利用到全部的CPU核心发挥最佳性能，因此选用满足以下条件的几种情况进行测试：  \n"
    content += "$$\n"
    content += "\\begin{align*}\n"
    content += "进程数\\times并行任务数&=GPU卡数\\\\\n"
    content += "进程数\\times线程数\\times并行任务数&=CPU核心数\n"
    content += "\\end{align*}\n"
    content += "$$\n"
    content += f"**每个任务使用{multitasks_best_mode.split('_')[0]}张GPU卡，同时最多并行{multitasks_best_mode.split('_')[-1]}个任务时计算总用时最少。**  \n\n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 大批量任务运行特征分析：\n"
    content += f"![multitasks_radar]({multitasks_radar_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括CPU使用率、 内存使用量、I/O读带宽、I/O写带宽、功耗)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar]({multitasks_radar_gpu_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括GPU使用率和全部显存使用量)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar_split]({multitasks_radar_split_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间各指标使用情况，从内向外不同层级分别展示一个指标的变化情况，每个层级内侧线为0，外侧线为该指标全程的最大值，指标范围见图例。  \n\n"
    content += "&nbsp;\n"
    content += f"GPU使用率峰值：{multitasks_gpu_summary['max']}%，中位数{multitasks_gpu_summary['median']}%，平均值{multitasks_gpu_summary['mean']}%  \n"
    content += f"显存使用峰值：{multitasks_gpu_mem_summary['max']}GB，中位数{multitasks_gpu_mem_summary['median']}GB，平均值{multitasks_gpu_mem_summary['mean']}GB  \n"
    content += f"CPU使用率峰值：{multitasks_cpu_summary['max']}%，中位数{multitasks_cpu_summary['median']}%，平均值{multitasks_cpu_summary['mean']}%  \n"
    content += f"内存使用峰值：{multitasks_mem_summary['max']}GB，中位数{multitasks_mem_summary['median']}GB，平均值{multitasks_mem_summary['mean']}GB  \n"
    if multitasks_read_summary["max"] < 1024:
        content += f"I/O读峰值速率：{round(multitasks_read_summary['max'], 2)}B/s，"
    elif (
        multitasks_read_summary["max"] >= 1024
        and multitasks_read_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O读峰值速率：{round(multitasks_read_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        multitasks_read_summary["max"] >= 1024 * 1024
        and multitasks_read_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O读峰值速率：{round(multitasks_read_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif multitasks_read_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O读峰值速率：{round(multitasks_read_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_read_summary["median"] < 1024:
        content += f"中位数{round(multitasks_read_summary['median'], 2)}B/s，"
    elif (
        multitasks_read_summary["median"] >= 1024
        and multitasks_read_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(multitasks_read_summary['median'] / 1024, 2)}KB/s，"
    elif (
        multitasks_read_summary["median"] >= 1024 * 1024
        and multitasks_read_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(multitasks_read_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif multitasks_read_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(multitasks_read_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_read_summary["mean"] < 1024:
        content += f"平均值{round(multitasks_read_summary['mean'], 2)}B/s  \n"
    elif (
        multitasks_read_summary["mean"] >= 1024
        and multitasks_read_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(multitasks_read_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        multitasks_read_summary["mean"] >= 1024 * 1024
        and multitasks_read_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(multitasks_read_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif multitasks_read_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(multitasks_read_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    if multitasks_write_summary["max"] < 1024:
        content += f"I/O写峰值速率：{round(multitasks_write_summary['max'], 2)}B/s，"
    elif (
        multitasks_write_summary["max"] >= 1024
        and multitasks_write_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O写峰值速率：{round(multitasks_write_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        multitasks_write_summary["max"] >= 1024 * 1024
        and multitasks_write_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O写峰值速率：{round(multitasks_write_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif multitasks_write_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O写峰值速率：{round(multitasks_write_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_write_summary["median"] < 1024:
        content += f"中位数{round(multitasks_write_summary['median'], 2)}B/s，"
    elif (
        multitasks_write_summary["median"] >= 1024
        and multitasks_write_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(multitasks_write_summary['median'] / 1024, 2)}KB/s，"
    elif (
        multitasks_write_summary["median"] >= 1024 * 1024
        and multitasks_write_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(multitasks_write_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif multitasks_write_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(multitasks_write_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_write_summary["mean"] < 1024:
        content += f"平均值{round(multitasks_write_summary['mean'], 2)}B/s  \n"
    elif (
        multitasks_write_summary["mean"] >= 1024
        and multitasks_write_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(multitasks_write_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        multitasks_write_summary["mean"] >= 1024 * 1024
        and multitasks_write_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(multitasks_write_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif multitasks_write_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(multitasks_write_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    # content += f"I/O读峰值速率：{multitasks_read_summary['max']}GB，中位数{multitasks_read_summary['median']}GB，平均值{multitasks_read_summary['mean']}GB  \n"
    # content += f"I/O写峰值速率：{multitasks_write_summary['max']}GB，中位数{multitasks_write_summary['median']}GB，平均值{multitasks_write_summary['mean']}GB  \n"
    content += f"功耗峰值：{multitasks_power_summary['max']}W，中位数{multitasks_power_summary['median']}W，平均值{multitasks_power_summary['mean']}W  \n"
    content += "&nbsp;\n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 节点性能打分：\n"
    content += f"测试软件：{software}  \n"
    content += f"单任务性能得分：{singletask_score}  \n"
    content += f"单任务计算用时稳定性得分：{stability_score}  \n"
    content += f"大批量任务计算用时得分：{multitasks_time_score}  \n"
    content += f"大批量任务计算能耗得分：{multitasks_energy_score}  \n"
    content += f"性能综合得分：{total_score}  \n"
    return content, total_score


def amber_report(
    content,
    multitasks_count,
    repeat,
    gpus,
    cpus,
    node_exporter_port,
    nvidia_gpu_exporter_port,
    server_ip,
    server_port,
    partition,
    node,
):
    software = "AMBER"
    version = get_version(software)
    cpu_info = get_cpu_info(partition, node)
    memory_info = get_memory_info(partition, node)
    filesystem_info = get_filesystem_info()
    gpu_info = get_gpu_info(partition, node)
    os_version = get_os_version(partition, node)
    kernel_version = get_kernel_version(partition, node)
    slurm_version = get_slurm_version()
    image_path = f"image/SingleNodeMode/{software}/"
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # 单任务
    singletask_gpus = gpus
    singletask_cpus_per_task = cpus // gpus
    singletask_best_mode = f"{singletask_gpus}_{singletask_cpus_per_task}"
    with open(
        f"log/SingleNodeMode/{software}/singletask/{singletask_best_mode}_1.start.timestamp",
        "r",
    ) as f:
        singletask_start_timestamp = int(f.read().strip())
    with open(
        f"log/SingleNodeMode/{software}/singletask/{singletask_best_mode}_1.end.timestamp",
        "r",
    ) as f:
        singletask_end_timestamp = int(f.read().strip())
    singletask_mondir = f"mon/SingleNodeMode/{software}/singletask/"
    step = 11000
    time_bins = auto_extract(
        node_exporter_port,
        nvidia_gpu_exporter_port,
        server_ip,
        server_port,
        node,
        singletask_start_timestamp,
        singletask_end_timestamp,
        singletask_mondir,
        step,
    )
    singletask_radar_path = f"image/SingleNodeMode/{software}/singletask_radar.png"
    singletask_radar_gpu_path = (
        f"image/SingleNodeMode/{software}/singletask_radar_gpu.png"
    )
    singletask_radar_split_path = (
        f"image/SingleNodeMode/{software}/singletask_radar_split.png"
    )
    (
        singletask_cpu_summary,
        singletask_mem_summary,
        singletask_read_summary,
        singletask_write_summary,
        singletask_power_summary,
        singletask_gpu_summary,
        singletask_gpu_mem_summary,
    ) = extract_and_draw(
        singletask_mondir,
        time_bins,
        cpus,
        gpus,
        singletask_radar_path,
        singletask_radar_gpu_path,
        singletask_radar_split_path,
        [software, "singletask", singletask_gpus],
    )
    singletask_repeat_time = []
    for repeat_id in range(1, repeat + 1):
        with open(
            f"result/SingleNodeMode/{software}/singletask/{singletask_best_mode}_{repeat_id}/mdout",
            "r",
        ) as f:
            ready = False
            for line in f:
                if "Average timings for all steps:" in line:
                    ready = True
                elif "Elapsed(s)" in line and ready:
                    singletask_repeat_time.append(float(line.split()[3]))
                else:
                    continue
    singletask_violin_path, stability_score = make_singletask_violin(
        software, singletask_gpus, singletask_repeat_time
    )

    (
        singletask_modes,
        singletask_time_result,
        singletask_gputime_result,
        singletask_performance_result,
    ) = extract_amber_singletask_result(gpus, cpus, repeat)
    singletask_time_boxplot_path = (
        f"image/SingleNodeMode/{software}/singletask_time_boxplot.png"
    )
    singletask_gputime_boxplot_path = (
        f"image/SingleNodeMode/{software}/singletask_gputime_boxplot.png"
    )
    singletask_performance_boxplot_path = (
        f"image/SingleNodeMode/{software}/singletask_performance_boxplot.png"
    )
    make_boxplot(
        singletask_time_result,
        singletask_modes,
        "进程数（GPU卡数）",
        "time",
        singletask_time_boxplot_path,
    )
    make_boxplot(
        singletask_gputime_result,
        singletask_modes,
        "进程数（GPU卡数）",
        "gputime",
        singletask_gputime_boxplot_path,
    )
    make_boxplot(
        singletask_performance_result,
        singletask_modes,
        "进程数（GPU卡数）",
        "performance",
        singletask_performance_boxplot_path,
    )
    singletask_gpu_list = singletask_modes
    singletask_time_list = [np.median(s) for s in singletask_time_result]
    singletask_gputime_list = [np.median(s) for s in singletask_gputime_result]
    singletask_performance_list = [np.median(s) for s in singletask_performance_result]
    singletask_best_time_gpu = singletask_gpu_list[
        singletask_time_list.index(min(singletask_time_list))
    ]
    singletask_best_gputime_gpu = singletask_gpu_list[
        singletask_gputime_list.index(min(singletask_gputime_list))
    ]
    singletask_best_performance_gpu = singletask_gpu_list[
        singletask_performance_list.index(max(singletask_performance_list))
    ]

    # 多任务
    multitasks_modes, multitasks_time_result = extract_amber_multitasks_result(
        gpus, cpus
    )
    multitasks_time_linechart_path = (
        f"image/SingleNodeMode/{software}/multitasks_time_linechart.png"
    )
    multitasks_best_mode = make_multitasks_linechart(
        multitasks_modes,
        multitasks_time_result,
        "进程数（GPU卡数）_并行任务数",
        multitasks_time_linechart_path,
    )
    multitasks_best_mode_gpu = multitasks_best_mode.split("_")[0]
    multitasks_best_mode_parallel = multitasks_best_mode.split("_")[1]
    multitasks_best_mode_cpu = cpus // gpus
    multitasks_start_timestamp, multitasks_end_timestamp = get_multitasks_timestamp(
        f"log/SingleNodeMode/{software}/multitasks/{multitasks_best_mode_gpu}_{multitasks_best_mode_cpu}_{multitasks_best_mode_parallel}/"
    )
    multitasks_mondir = f"mon/SingleNodeMode/{software}/multitasks/"
    step = 11000
    time_bins = auto_extract(
        node_exporter_port,
        nvidia_gpu_exporter_port,
        server_ip,
        server_port,
        node,
        multitasks_start_timestamp,
        multitasks_end_timestamp,
        multitasks_mondir,
        step,
    )
    multitasks_radar_path = f"image/SingleNodeMode/{software}/multitasks_radar.png"
    multitasks_radar_gpu_path = (
        f"image/SingleNodeMode/{software}/multitasks_radar_gpu.png"
    )
    multitasks_radar_split_path = (
        f"image/SingleNodeMode/{software}/multitasks_radar_split.png"
    )
    (
        multitasks_cpu_summary,
        multitasks_mem_summary,
        multitasks_read_summary,
        multitasks_write_summary,
        multitasks_power_summary,
        multitasks_gpu_summary,
        multitasks_gpu_mem_summary,
    ) = extract_and_draw(
        multitasks_mondir,
        time_bins,
        cpus,
        gpus,
        multitasks_radar_path,
        multitasks_radar_gpu_path,
        multitasks_radar_split_path,
        [software, "multitasks", multitasks_best_mode],
    )

    # 计算得分
    singletask_score = cal_singletask_score(software, singletask_best_mode, repeat)
    multitasks_time_score = cal_multitasks_time_score(
        software,
        multitasks_count,
        multitasks_end_timestamp - multitasks_start_timestamp,
    )
    multitasks_energy_score = cal_multitasks_energy_score(
        software, multitasks_count, multitasks_power_summary["sum"]
    )
    total_score = round(
        np.mean(
            [
                singletask_score,
                multitasks_time_score,
                multitasks_energy_score,
            ]
        )
        * (stability_score / 100),
        2,
    )

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"## {software}\n"
    content += f"软件版本：{version}  \n"
    content += f"数据集：{dataset[software]}  \n"
    content += f"测试环境：GPU型号{gpu_info['gpu_name']}，GPU卡数{gpu_info['gpu_count']}，单卡显存{gpu_info['memory']}GB，CPU型号{cpu_info['model']}，CPU核心数{cpu_info['cores']}，内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']}，文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。操作系统版本{os_version}，内核版本{kernel_version}，作业调度系统{slurm_version}。  \n\n"
    content += "&nbsp;\n"

    content += "测评内容：  \n"
    content += "1. 单任务运行特征分析。记录软件计算时对GPU、CPU、内存、I/O读写带宽等资源的使用情况和功耗，输出从任务运行到结束的资源使用情况雷达图，帮助用户了解软件资源需求特征、分析被测节点配置是否存在瓶颈。\n"
    content += "2. 单任务并行效率分析。使用不同数量的GPU卡运行相同算例，统计计算用时、GPU卡时和分子动力学模拟性能，计算使用不同数量GPU卡的加速比和计算效率。通过本指标的评测，可以得知计算用时最少的单任务并行策略以及计算卡时最少的单任务并行策略。  \n\n"
    content += "$$\n"
    content += "\\begin{align}\n"
    content += (
        "加速比&=\\frac{该GPU卡数的分子动力学模拟性能}{单卡分子动力学模拟性能}\\\\\n"
    )
    content += "计算效率&=\\frac{加速比}{GPU卡数}*100\\%\n"
    content += "\\end{align}\n"
    content += "$$\n"
    content += "3. 单任务计算用时稳定性分析。多次测试相同单任务的计算用时，统计计算用时分布，输出小提琴图。本指标是为了测试被测节点的软硬件环境是否存在较大的性能波动。\n"
    content += "4. 大批量任务并行效率分析。使用不同GPU卡数×并行任务数的组合测试计算软件大批量任务并行计算。输出不同运行模式计算用时的折线图，以及在目前节点配置下大批量任务并行时最佳并行模式。通过本指标的评测，可以得知在何种并行策略下大批量任务的计算总用时最少。\n"
    content += "5. 大批量任务运行特征分析。记录计算软件在大批量任务并行模式下的GPU、CPU、内存、I/O读写带宽等资源的使用情况和功耗，输出从任务运行到结束的资源使用情况雷达图，帮助用户了解软件在大批量任务并行模式下的资源需求特征、分析被测节点配置是否存在瓶颈。  \n\n"
    content += "&nbsp;\n"
    content += f"计算当前测试环境下软件性能得分。本测评根据单任务性能得分、计算用时稳定性得分、大批量任务计算用时得分、大批量任务计算能耗得分等四个细分指标的综合计算，得到单节点的{software}性能总分。  \n"
    content += "$$\n"
    content += "总得分=avg(单任务性能得分+大批量任务计算用时得分+大批量任务计算能耗得分)\\times\\frac{计算用时稳定性得分}{100}\n"
    content += "$$\n"
    content += "四个细分指标详述如下：  \n"
    content += "① 单任务性能得分：根据多次相同单任务的平均分子动力学模拟性能计算。  \n"
    content += "$$\n"
    content += "单任务性能得分={系数}\\times{分子动力学模拟性能}\n"
    content += "$$\n"
    content += "② 计算用时稳定性得分：根据多次相同单任务的用时分布计算。  \n"
    content += "$$\n"
    content += "单任务计算用时稳定性得分=max((1-\\frac{计算用时标准差}{计算用时均值})\\times100,0)\n"
    content += "$$\n"
    content += "③ 大批量任务计算用时得分：根据大批量任务并行的总用时计算。  \n"
    content += "$$\n"
    content += "大批量任务计算用时得分=系数\\times\\frac{任务数量}{计算总用时}\n"
    content += "$$\n"
    content += "④ 大批量任务计算能耗得分：根据大批量任务并行的总能耗计算。  \n"
    content += "$$\n"
    content += "大批量任务计算能耗得分=系数\\times\\frac{任务数量}{计算总能耗}\n"
    content += "$$\n"
    content += "注：系数以一个标准配置为基准（100分）来确定，测试环境的最终得分为相对于标准配置的相对得分。分子动力学模拟性能为分子动力学模拟软件常用的性能指标，单位为ns/day，表示在当前系统配置和环境以及输入参数条件下每天能够模拟的纳秒数，该值越大代表单位时间内能完成更长时间的分子动力学模拟，代表整体系统性能更好。  \n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 单任务运行特征分析：\n"
    content += f"![singletask_radar]({singletask_radar_path})  \n"
    content += "注：从顶端顺时针一周为软件单任务运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括CPU使用率、 内存使用量、I/O读带宽、I/O写带宽、功耗)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![singletask_radar]({singletask_radar_gpu_path})  \n"
    content += "注：从顶端顺时针一周为软件单任务运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括GPU使用率和全部显存使用量)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![singletask_radar_split]({singletask_radar_split_path})  \n"
    content += "注：从顶端顺时针一周为软件单任务运行期间各指标使用情况，从内向外不同层级分别展示一个指标的变化情况，每个层级内侧线为0，外侧线为该指标全程的最大值，指标范围见图例。  \n\n"
    content += "&nbsp;\n"
    content += f"GPU使用率峰值：{singletask_gpu_summary['max']}%，中位数{singletask_gpu_summary['median']}%，平均值{singletask_gpu_summary['mean']}%  \n"
    content += f"显存使用峰值：{singletask_gpu_mem_summary['max']}GB，中位数{singletask_gpu_mem_summary['median']}GB，平均值{singletask_gpu_mem_summary['mean']}GB  \n"
    content += f"CPU使用率峰值：{singletask_cpu_summary['max']}%，中位数{singletask_cpu_summary['median']}%，平均值{singletask_cpu_summary['mean']}%  \n"
    content += f"内存使用峰值：{singletask_mem_summary['max']}GB，中位数{singletask_mem_summary['median']}GB，平均值{singletask_mem_summary['mean']}GB  \n"
    if singletask_read_summary["max"] < 1024:
        content += f"I/O读峰值速率：{round(singletask_read_summary['max'], 2)}B/s，"
    elif (
        singletask_read_summary["max"] >= 1024
        and singletask_read_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O读峰值速率：{round(singletask_read_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        singletask_read_summary["max"] >= 1024 * 1024
        and singletask_read_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O读峰值速率：{round(singletask_read_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif singletask_read_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O读峰值速率：{round(singletask_read_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_read_summary["median"] < 1024:
        content += f"中位数{round(singletask_read_summary['median'], 2)}B/s，"
    elif (
        singletask_read_summary["median"] >= 1024
        and singletask_read_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(singletask_read_summary['median'] / 1024, 2)}KB/s，"
    elif (
        singletask_read_summary["median"] >= 1024 * 1024
        and singletask_read_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(singletask_read_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif singletask_read_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(singletask_read_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_read_summary["mean"] < 1024:
        content += f"平均值{round(singletask_read_summary['mean'], 2)}B/s  \n"
    elif (
        singletask_read_summary["mean"] >= 1024
        and singletask_read_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(singletask_read_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        singletask_read_summary["mean"] >= 1024 * 1024
        and singletask_read_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(singletask_read_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif singletask_read_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(singletask_read_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    if singletask_write_summary["max"] < 1024:
        content += f"I/O写峰值速率：{round(singletask_write_summary['max'], 2)}B/s，"
    elif (
        singletask_write_summary["max"] >= 1024
        and singletask_write_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O写峰值速率：{round(singletask_write_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        singletask_write_summary["max"] >= 1024 * 1024
        and singletask_write_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O写峰值速率：{round(singletask_write_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif singletask_write_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O写峰值速率：{round(singletask_write_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_write_summary["median"] < 1024:
        content += f"中位数{round(singletask_write_summary['median'], 2)}B/s，"
    elif (
        singletask_write_summary["median"] >= 1024
        and singletask_write_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(singletask_write_summary['median'] / 1024, 2)}KB/s，"
    elif (
        singletask_write_summary["median"] >= 1024 * 1024
        and singletask_write_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(singletask_write_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif singletask_write_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(singletask_write_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_write_summary["mean"] < 1024:
        content += f"平均值{round(singletask_write_summary['mean'], 2)}B/s  \n"
    elif (
        singletask_write_summary["mean"] >= 1024
        and singletask_write_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(singletask_write_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        singletask_write_summary["mean"] >= 1024 * 1024
        and singletask_write_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(singletask_write_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif singletask_write_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(singletask_write_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    # content += f"I/O读峰值速率：{singletask_read_summary['max']}GB，中位数{singletask_read_summary['median']}GB，平均值{singletask_read_summary['mean']}GB  \n"
    # content += f"I/O写峰值速率：{singletask_write_summary['max']}GB，中位数{singletask_write_summary['median']}GB，平均值{singletask_write_summary['mean']}GB  \n"
    content += f"功耗峰值：{singletask_power_summary['max']}W，中位数{singletask_power_summary['median']}W，平均值{singletask_power_summary['mean']}W  \n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 单任务并行效率分析：\n"
    content += "计算用时：  \n"
    content += f"![singletask_time_boxplot]({singletask_time_boxplot_path})  \n"
    content += "注：横坐标为不同的进程数（GPU卡数），纵坐标为任务运行时间，用于对比发现AMBER单机多卡并行时计算用时最少的运行模式。进程数代表AMBER开启的进程数，一个进程使用一张GPU卡，所以进程数可以理解为等价于使用的GPU卡数，因此选用进程数<=GPU卡总数的几种情况进行测试。  \n"
    content += f"**使用{singletask_best_time_gpu}张GPU卡并行计算时计算用时最少。**  \n"
    content += "&nbsp;\n"
    content += "计算所用GPU卡时：  \n"
    content += f"![singletask_gputime_boxplot]({singletask_gputime_boxplot_path})  \n"
    content += "注：横坐标为不同的进程数（GPU卡数），纵坐标为GPU卡时（GPU卡时=所用GPU数量×计算用时），用于对比发现AMBER单机多卡并行时GPU卡时最少（GPU最经济）的运行模式。进程数代表AMBER开启的进程数，一个进程使用一张GPU卡，所以进程数可以理解为等价于使用的GPU卡数，因此选用进程数<=GPU卡总数的几种情况进行测试。  \n"
    content += (
        f"**使用{singletask_best_gputime_gpu}张GPU卡并行计算时GPU卡时最少。**  \n"
    )
    content += "&nbsp;\n"
    content += "分子动力学模拟性能：  \n"
    content += (
        f"![singletask_performance_boxplot]({singletask_performance_boxplot_path})  \n"
    )
    content += "注：横坐标为不同的进程数（GPU卡数），纵坐标为分子动力学模拟性能，用于对比发现AMBER单机多卡并行时分子动力学模拟性能最高的运行模式。进程数代表AMBER开启的进程数，一个进程使用一张GPU卡，所以进程数可以理解为等价于使用的GPU卡数，因此选用进程数<=GPU卡总数的几种情况进行测试。分子动力学模拟性能为分子动力学模拟软件常用的性能指标，单位为ns/day，表示在当前系统配置和环境以及输入参数条件下每天能够模拟的纳秒数，该值越大代表单位时间内能完成更长时间的分子动力学模拟，代表整体系统性能更好。  \n"
    content += f"**使用{singletask_best_performance_gpu}张GPU卡并行计算时分子动力学模拟性能最高。**  \n"
    content += "&nbsp;\n"
    content += cal_speedup(
        singletask_gpu_list,
        singletask_time_list,
        singletask_gputime_list,
        singletask_performance_list,
    )

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 单任务计算用时稳定性分析：\n"
    content += f"使用全部{singletask_best_mode.split('_')[0]}张GPU卡并行计算相同算例，测试{repeat}次，统计用时分布。  \n"
    content += f"![singletask_violin]({singletask_violin_path})  \n"
    content += "注：纵坐标为计算用时。小提琴图代表多次测试的计算用时分布，小提琴图越宽的地方代表在该范围内的数据分布密度越高，也就是说越多的数据点集中在这个区间。红线代表多次测试的计算用时的中位数，绿线代表平均值。  \n\n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 大批量任务并行效率分析：\n"
    content += f"![multitasks_time_linechart]({multitasks_time_linechart_path})  \n"
    content += "注：横坐标为不同的进程数（GPU卡数）×并行任务数组合，纵坐标为计算总用时，用于对比发现AMBER大批量任务并行时计算总用时最少的运行模式。进程数代表AMBER开启的进程数，一个进程使用一张GPU卡，所以进程数可以理解为等价于使用的GPU卡数，因此进程数应<=GPU卡的总数。并行任务数代表同时最多并行的任务数量，最大等于GPU卡数（每个任务使用一张GPU卡），最小为1（每个任务使用全部GPU卡，任务间串行）。因此选用满足 $进程数×并行任务数=GPU卡数$ 的几种情况进行测试。  \n"
    content += f"**每个任务使用{multitasks_best_mode.split('_')[0]}张GPU卡，同时最多并行{multitasks_best_mode.split('_')[-1]}个任务时计算总用时最少。**  \n\n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 大批量任务运行特征分析：\n"
    content += f"![multitasks_radar]({multitasks_radar_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括CPU使用率、 内存使用量、I/O读带宽、I/O写带宽、功耗)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar]({multitasks_radar_gpu_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括GPU使用率和全部显存使用量)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar_split]({multitasks_radar_split_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间各指标使用情况，从内向外不同层级分别展示一个指标的变化情况，每个层级内侧线为0，外侧线为该指标全程的最大值，指标范围见图例。  \n\n"
    content += "&nbsp;\n"
    content += f"GPU使用率峰值：{multitasks_gpu_summary['max']}%，中位数{multitasks_gpu_summary['median']}%，平均值{multitasks_gpu_summary['mean']}%  \n"
    content += f"显存使用峰值：{multitasks_gpu_mem_summary['max']}GB，中位数{multitasks_gpu_mem_summary['median']}GB，平均值{multitasks_gpu_mem_summary['mean']}GB  \n"
    content += f"CPU使用率峰值：{multitasks_cpu_summary['max']}%，中位数{multitasks_cpu_summary['median']}%，平均值{multitasks_cpu_summary['mean']}%  \n"
    content += f"内存使用峰值：{multitasks_mem_summary['max']}GB，中位数{multitasks_mem_summary['median']}GB，平均值{multitasks_mem_summary['mean']}GB  \n"
    if multitasks_read_summary["max"] < 1024:
        content += f"I/O读峰值速率：{round(multitasks_read_summary['max'], 2)}B/s，"
    elif (
        multitasks_read_summary["max"] >= 1024
        and multitasks_read_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O读峰值速率：{round(multitasks_read_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        multitasks_read_summary["max"] >= 1024 * 1024
        and multitasks_read_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O读峰值速率：{round(multitasks_read_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif multitasks_read_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O读峰值速率：{round(multitasks_read_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_read_summary["median"] < 1024:
        content += f"中位数{round(multitasks_read_summary['median'], 2)}B/s，"
    elif (
        multitasks_read_summary["median"] >= 1024
        and multitasks_read_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(multitasks_read_summary['median'] / 1024, 2)}KB/s，"
    elif (
        multitasks_read_summary["median"] >= 1024 * 1024
        and multitasks_read_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(multitasks_read_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif multitasks_read_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(multitasks_read_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_read_summary["mean"] < 1024:
        content += f"平均值{round(multitasks_read_summary['mean'], 2)}B/s  \n"
    elif (
        multitasks_read_summary["mean"] >= 1024
        and multitasks_read_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(multitasks_read_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        multitasks_read_summary["mean"] >= 1024 * 1024
        and multitasks_read_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(multitasks_read_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif multitasks_read_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(multitasks_read_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    if multitasks_write_summary["max"] < 1024:
        content += f"I/O写峰值速率：{round(multitasks_write_summary['max'], 2)}B/s，"
    elif (
        multitasks_write_summary["max"] >= 1024
        and multitasks_write_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O写峰值速率：{round(multitasks_write_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        multitasks_write_summary["max"] >= 1024 * 1024
        and multitasks_write_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O写峰值速率：{round(multitasks_write_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif multitasks_write_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O写峰值速率：{round(multitasks_write_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_write_summary["median"] < 1024:
        content += f"中位数{round(multitasks_write_summary['median'], 2)}B/s，"
    elif (
        multitasks_write_summary["median"] >= 1024
        and multitasks_write_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(multitasks_write_summary['median'] / 1024, 2)}KB/s，"
    elif (
        multitasks_write_summary["median"] >= 1024 * 1024
        and multitasks_write_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(multitasks_write_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif multitasks_write_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(multitasks_write_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_write_summary["mean"] < 1024:
        content += f"平均值{round(multitasks_write_summary['mean'], 2)}B/s  \n"
    elif (
        multitasks_write_summary["mean"] >= 1024
        and multitasks_write_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(multitasks_write_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        multitasks_write_summary["mean"] >= 1024 * 1024
        and multitasks_write_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(multitasks_write_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif multitasks_write_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(multitasks_write_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    # content += f"I/O读峰值速率：{multitasks_read_summary['max']}GB，中位数{multitasks_read_summary['median']}GB，平均值{multitasks_read_summary['mean']}GB  \n"
    # content += f"I/O写峰值速率：{multitasks_write_summary['max']}GB，中位数{multitasks_write_summary['median']}GB，平均值{multitasks_write_summary['mean']}GB  \n"
    content += f"功耗峰值：{multitasks_power_summary['max']}W，中位数{multitasks_power_summary['median']}W，平均值{multitasks_power_summary['mean']}W  \n"
    content += "&nbsp;\n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 节点性能打分：\n"
    content += f"测试软件：{software}  \n"
    content += f"单任务性能得分：{singletask_score}  \n"
    content += f"单任务计算用时稳定性得分：{stability_score}  \n"
    content += f"大批量任务计算用时得分：{multitasks_time_score}  \n"
    content += f"大批量任务计算能耗得分：{multitasks_energy_score}  \n"
    content += f"性能综合得分：{total_score}  \n"
    return content, total_score


def dsdp_report(
    content,
    multitasks_count,
    repeat,
    gpus,
    cpus,
    node_exporter_port,
    nvidia_gpu_exporter_port,
    server_ip,
    server_port,
    partition,
    node,
):
    software = "DSDP"
    version = get_version(software)
    cpu_info = get_cpu_info(partition, node)
    memory_info = get_memory_info(partition, node)
    filesystem_info = get_filesystem_info()
    gpu_info = get_gpu_info(partition, node)
    os_version = get_os_version(partition, node)
    kernel_version = get_kernel_version(partition, node)
    slurm_version = get_slurm_version()
    image_path = f"image/SingleNodeMode/{software}/"
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # 单任务
    singletask_gpus = 1
    singletask_cpus_per_task = cpus
    singletask_best_mode = f"{singletask_gpus}_{singletask_cpus_per_task}"
    with open(
        f"log/SingleNodeMode/{software}/singletask/{singletask_best_mode}_1.start.timestamp",
        "r",
    ) as f:
        singletask_start_timestamp = int(f.read().strip())
    with open(
        f"log/SingleNodeMode/{software}/singletask/{singletask_best_mode}_1.end.timestamp",
        "r",
    ) as f:
        singletask_end_timestamp = int(f.read().strip())
    singletask_mondir = f"mon/SingleNodeMode/{software}/singletask/"
    step = 11000
    time_bins = auto_extract(
        node_exporter_port,
        nvidia_gpu_exporter_port,
        server_ip,
        server_port,
        node,
        singletask_start_timestamp,
        singletask_end_timestamp,
        singletask_mondir,
        step,
    )
    singletask_radar_path = f"image/SingleNodeMode/{software}/singletask_radar.png"
    singletask_radar_gpu_path = (
        f"image/SingleNodeMode/{software}/singletask_radar_gpu.png"
    )
    singletask_radar_split_path = (
        f"image/SingleNodeMode/{software}/singletask_radar_split.png"
    )
    (
        singletask_cpu_summary,
        singletask_mem_summary,
        singletask_read_summary,
        singletask_write_summary,
        singletask_power_summary,
        singletask_gpu_summary,
        singletask_gpu_mem_summary,
    ) = extract_and_draw(
        singletask_mondir,
        time_bins,
        cpus,
        gpus,
        singletask_radar_path,
        singletask_radar_gpu_path,
        singletask_radar_split_path,
        [software, "singletask", singletask_gpus],
    )
    singletask_repeat_time = []
    for repeat_id in range(1, repeat + 1):
        with open(
            f"log/SingleNodeMode/{software}/singletask/{singletask_best_mode}_{repeat_id}.log",
            "r",
        ) as f:
            for line in f:
                if "Time:" in line:
                    singletask_repeat_time.append(float(line.split()[1]))
    singletask_violin_path, stability_score = make_singletask_violin(
        software, singletask_gpus, singletask_repeat_time
    )

    # 多任务
    multitasks_best_mode = (
        f"{singletask_gpus}_{int(cpus / gpus)}_{int(gpus / singletask_gpus)}"
    )
    multitasks_start_timestamp, multitasks_end_timestamp = get_multitasks_timestamp(
        f"log/SingleNodeMode/{software}/multitasks/{multitasks_best_mode}"
    )
    multitasks_mondir = f"mon/SingleNodeMode/{software}/multitasks/"
    step = 11000
    time_bins = auto_extract(
        node_exporter_port,
        nvidia_gpu_exporter_port,
        server_ip,
        server_port,
        node,
        multitasks_start_timestamp,
        multitasks_end_timestamp,
        multitasks_mondir,
        step,
    )
    multitasks_radar_path = f"image/SingleNodeMode/{software}/multitasks_radar.png"
    multitasks_radar_gpu_path = (
        f"image/SingleNodeMode/{software}/multitasks_radar_gpu.png"
    )
    multitasks_radar_split_path = (
        f"image/SingleNodeMode/{software}/multitasks_radar_split.png"
    )
    (
        multitasks_cpu_summary,
        multitasks_mem_summary,
        multitasks_read_summary,
        multitasks_write_summary,
        multitasks_power_summary,
        multitasks_gpu_summary,
        multitasks_gpu_mem_summary,
    ) = extract_and_draw(
        multitasks_mondir,
        time_bins,
        cpus,
        gpus,
        multitasks_radar_path,
        multitasks_radar_gpu_path,
        multitasks_radar_split_path,
        [software, "multitasks", multitasks_best_mode],
    )

    singletask_score = cal_singletask_score(software, singletask_best_mode, repeat)
    multitasks_time_score = cal_multitasks_time_score(
        software,
        multitasks_count,
        multitasks_end_timestamp - multitasks_start_timestamp,
    )
    multitasks_energy_score = cal_multitasks_energy_score(
        software, multitasks_count, multitasks_power_summary["sum"]
    )
    total_score = round(
        np.mean(
            [
                singletask_score,
                multitasks_time_score,
                multitasks_energy_score,
            ]
        )
        * (stability_score / 100),
        2,
    )

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"## {software}\n"
    content += f"软件版本：{version}  \n"
    content += f"数据集：{dataset[software]}  \n"
    content += f"测试环境：GPU型号{gpu_info['gpu_name']}，GPU卡数{gpu_info['gpu_count']}，单卡显存{gpu_info['memory']}GB，CPU型号{cpu_info['model']}，CPU核心数{cpu_info['cores']}，内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']}，文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。操作系统版本{os_version}，内核版本{kernel_version}，作业调度系统{slurm_version}。  \n\n"
    content += "&nbsp;\n"

    content += "测评内容：  \n"
    content += "1. 单任务运行特征分析。记录软件计算时对GPU、CPU、内存、I/O读写带宽等资源的使用情况和功耗，输出从任务运行到结束的资源使用情况雷达图，帮助用户了解软件资源需求特征、分析被测节点配置是否存在瓶颈。\n"
    content += "2. 单任务计算用时稳定性分析。多次测试相同单任务的计算用时，统计计算用时分布，输出小提琴图。本指标是为了测试被测节点的软硬件环境是否存在较大的性能波动。\n"
    content += "3. 大批量任务运行特征分析。记录计算软件在大批量任务并行模式下的GPU、CPU、内存、I/O读写带宽等资源的使用情况和功耗，输出从任务运行到结束的资源使用情况雷达图，帮助用户了解软件在大批量任务并行模式下的资源需求特征、分析被测节点配置是否存在瓶颈。  \n\n"
    content += "&nbsp;\n"
    content += f"计算当前测试环境下软件性能得分。本测评根据单任务性能得分、计算用时稳定性得分、大批量任务计算用时得分、大批量任务计算能耗得分等四个细分指标的综合计算，得到单节点的{software}性能总分。  \n"
    content += "$$\n"
    content += "总得分=avg(单任务性能得分+大批量任务计算用时得分+大批量任务计算能耗得分)\\times\\frac{计算用时稳定性得分}{100}\n"
    content += "$$\n"
    content += "四个细分指标详述如下：  \n"
    content += "① 单任务性能得分：根据多次相同单任务的平均分子动力学模拟性能计算。  \n"
    content += "$$\n"
    content += "单任务性能得分=\\frac{系数}{单任务计算用时}\n"
    content += "$$\n"
    content += "② 计算用时稳定性得分：根据多次相同单任务的用时分布计算。  \n"
    content += "$$\n"
    content += "单任务计算用时稳定性得分=max((1-\\frac{计算用时标准差}{计算用时均值})\\times100,0)\n"
    content += "$$\n"
    content += "③ 大批量任务计算用时得分：根据大批量任务并行的总用时计算。  \n"
    content += "$$\n"
    content += "大批量任务计算用时得分=系数\\times\\frac{任务数量}{计算总用时}\n"
    content += "$$\n"
    content += "④ 大批量任务计算能耗得分：根据大批量任务并行的总能耗计算。  \n"
    content += "$$\n"
    content += "大批量任务计算能耗得分=系数\\times\\frac{任务数量}{计算总能耗}\n"
    content += "$$\n"
    content += "注：系数以一个标准配置为基准（100分）来确定，测试环境的最终得分为相对于标准配置的相对得分。  \n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 单任务运行特征分析：\n"
    content += f"![singletask_radar]({singletask_radar_path})  \n"
    content += "注：从顶端顺时针一周为软件单任务运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括CPU使用率、 内存使用量、I/O读带宽、I/O写带宽、功耗)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![singletask_radar]({singletask_radar_gpu_path})  \n"
    content += "注：从顶端顺时针一周为软件单任务运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括GPU使用率和全部显存使用量)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![singletask_radar_split]({singletask_radar_split_path})  \n"
    content += "注：从顶端顺时针一周为软件单任务运行期间各指标使用情况，从内向外不同层级分别展示一个指标的变化情况，每个层级内侧线为0，外侧线为该指标全程的最大值，指标范围见图例。  \n\n"
    content += "&nbsp;\n"
    content += f"GPU使用率峰值：{singletask_gpu_summary['max']}%，中位数{singletask_gpu_summary['median']}%，平均值{singletask_gpu_summary['mean']}%  \n"
    content += f"显存使用峰值：{singletask_gpu_mem_summary['max']}GB，中位数{singletask_gpu_mem_summary['median']}GB，平均值{singletask_gpu_mem_summary['mean']}GB  \n"
    content += f"CPU使用率峰值：{singletask_cpu_summary['max']}%，中位数{singletask_cpu_summary['median']}%，平均值{singletask_cpu_summary['mean']}%  \n"
    content += f"内存使用峰值：{singletask_mem_summary['max']}GB，中位数{singletask_mem_summary['median']}GB，平均值{singletask_mem_summary['mean']}GB  \n"
    if singletask_read_summary["max"] < 1024:
        content += f"I/O读峰值速率：{round(singletask_read_summary['max'], 2)}B/s，"
    elif (
        singletask_read_summary["max"] >= 1024
        and singletask_read_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O读峰值速率：{round(singletask_read_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        singletask_read_summary["max"] >= 1024 * 1024
        and singletask_read_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O读峰值速率：{round(singletask_read_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif singletask_read_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O读峰值速率：{round(singletask_read_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_read_summary["median"] < 1024:
        content += f"中位数{round(singletask_read_summary['median'], 2)}B/s，"
    elif (
        singletask_read_summary["median"] >= 1024
        and singletask_read_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(singletask_read_summary['median'] / 1024, 2)}KB/s，"
    elif (
        singletask_read_summary["median"] >= 1024 * 1024
        and singletask_read_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(singletask_read_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif singletask_read_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(singletask_read_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_read_summary["mean"] < 1024:
        content += f"平均值{round(singletask_read_summary['mean'], 2)}B/s  \n"
    elif (
        singletask_read_summary["mean"] >= 1024
        and singletask_read_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(singletask_read_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        singletask_read_summary["mean"] >= 1024 * 1024
        and singletask_read_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(singletask_read_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif singletask_read_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(singletask_read_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    if singletask_write_summary["max"] < 1024:
        content += f"I/O写峰值速率：{round(singletask_write_summary['max'], 2)}B/s，"
    elif (
        singletask_write_summary["max"] >= 1024
        and singletask_write_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O写峰值速率：{round(singletask_write_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        singletask_write_summary["max"] >= 1024 * 1024
        and singletask_write_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O写峰值速率：{round(singletask_write_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif singletask_write_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O写峰值速率：{round(singletask_write_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_write_summary["median"] < 1024:
        content += f"中位数{round(singletask_write_summary['median'], 2)}B/s，"
    elif (
        singletask_write_summary["median"] >= 1024
        and singletask_write_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(singletask_write_summary['median'] / 1024, 2)}KB/s，"
    elif (
        singletask_write_summary["median"] >= 1024 * 1024
        and singletask_write_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(singletask_write_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif singletask_write_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(singletask_write_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if singletask_write_summary["mean"] < 1024:
        content += f"平均值{round(singletask_write_summary['mean'], 2)}B/s  \n"
    elif (
        singletask_write_summary["mean"] >= 1024
        and singletask_write_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(singletask_write_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        singletask_write_summary["mean"] >= 1024 * 1024
        and singletask_write_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(singletask_write_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif singletask_write_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(singletask_write_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    # content += f"I/O读峰值速率：{singletask_read_summary['max']}GB，中位数{singletask_read_summary['median']}GB，平均值{singletask_read_summary['mean']}GB  \n"
    # content += f"I/O写峰值速率：{singletask_write_summary['max']}GB，中位数{singletask_write_summary['median']}GB，平均值{singletask_write_summary['mean']}GB  \n"
    content += f"功耗峰值：{singletask_power_summary['max']}W，中位数{singletask_power_summary['median']}W，平均值{singletask_power_summary['mean']}W  \n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 单任务计算用时稳定性分析：\n"
    content += f"相同算例测试{repeat}次，统计用时分布。  \n"
    content += f"![singletask_violin]({singletask_violin_path})  \n"
    content += "注：纵坐标为计算用时。小提琴图代表多次测试的计算用时分布，小提琴图越宽的地方代表在该范围内的数据分布密度越高，也就是说越多的数据点集中在这个区间。红线代表多次测试的计算用时的中位数，绿线代表平均值。  \n\n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 大批量任务运行特征分析：\n"
    content += f"![multitasks_radar]({multitasks_radar_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括CPU使用率、 内存使用量、I/O读带宽、I/O写带宽、功耗)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar]({multitasks_radar_gpu_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括GPU使用率和全部显存使用量)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar_split]({multitasks_radar_split_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间各指标使用情况，从内向外不同层级分别展示一个指标的变化情况，每个层级内侧线为0，外侧线为该指标全程的最大值，指标范围见图例。  \n\n"
    content += "&nbsp;\n"
    content += f"GPU使用率峰值：{multitasks_gpu_summary['max']}%，中位数{multitasks_gpu_summary['median']}%，平均值{multitasks_gpu_summary['mean']}%  \n"
    content += f"显存使用峰值：{multitasks_gpu_mem_summary['max']}GB，中位数{multitasks_gpu_mem_summary['median']}GB，平均值{multitasks_gpu_mem_summary['mean']}GB  \n"
    content += f"CPU使用率峰值：{multitasks_cpu_summary['max']}%，中位数{multitasks_cpu_summary['median']}%，平均值{multitasks_cpu_summary['mean']}%  \n"
    content += f"内存使用峰值：{multitasks_mem_summary['max']}GB，中位数{multitasks_mem_summary['median']}GB，平均值{multitasks_mem_summary['mean']}GB  \n"
    if multitasks_read_summary["max"] < 1024:
        content += f"I/O读峰值速率：{round(multitasks_read_summary['max'], 2)}B/s，"
    elif (
        multitasks_read_summary["max"] >= 1024
        and multitasks_read_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O读峰值速率：{round(multitasks_read_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        multitasks_read_summary["max"] >= 1024 * 1024
        and multitasks_read_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O读峰值速率：{round(multitasks_read_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif multitasks_read_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O读峰值速率：{round(multitasks_read_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_read_summary["median"] < 1024:
        content += f"中位数{round(multitasks_read_summary['median'], 2)}B/s，"
    elif (
        multitasks_read_summary["median"] >= 1024
        and multitasks_read_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(multitasks_read_summary['median'] / 1024, 2)}KB/s，"
    elif (
        multitasks_read_summary["median"] >= 1024 * 1024
        and multitasks_read_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(multitasks_read_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif multitasks_read_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(multitasks_read_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_read_summary["mean"] < 1024:
        content += f"平均值{round(multitasks_read_summary['mean'], 2)}B/s  \n"
    elif (
        multitasks_read_summary["mean"] >= 1024
        and multitasks_read_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(multitasks_read_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        multitasks_read_summary["mean"] >= 1024 * 1024
        and multitasks_read_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(multitasks_read_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif multitasks_read_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(multitasks_read_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    if multitasks_write_summary["max"] < 1024:
        content += f"I/O写峰值速率：{round(multitasks_write_summary['max'], 2)}B/s，"
    elif (
        multitasks_write_summary["max"] >= 1024
        and multitasks_write_summary["max"] < 1024 * 1024
    ):
        content += (
            f"I/O写峰值速率：{round(multitasks_write_summary['max'] / 1024, 2)}KB/s，"
        )
    elif (
        multitasks_write_summary["max"] >= 1024 * 1024
        and multitasks_write_summary["max"] < 1024 * 1024 * 1024
    ):
        content += f"I/O写峰值速率：{round(multitasks_write_summary['max'] / 1024 / 1024, 2)}MB/s，"
    elif multitasks_write_summary["max"] >= 1024 * 1024 * 1024:
        content += f"I/O写峰值速率：{round(multitasks_write_summary['max'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_write_summary["median"] < 1024:
        content += f"中位数{round(multitasks_write_summary['median'], 2)}B/s，"
    elif (
        multitasks_write_summary["median"] >= 1024
        and multitasks_write_summary["median"] < 1024 * 1024
    ):
        content += f"中位数{round(multitasks_write_summary['median'] / 1024, 2)}KB/s，"
    elif (
        multitasks_write_summary["median"] >= 1024 * 1024
        and multitasks_write_summary["median"] < 1024 * 1024 * 1024
    ):
        content += (
            f"中位数{round(multitasks_write_summary['median'] / 1024 / 1024, 2)}MB/s，"
        )
    elif multitasks_write_summary["median"] >= 1024 * 1024 * 1024:
        content += f"中位数{round(multitasks_write_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
    if multitasks_write_summary["mean"] < 1024:
        content += f"平均值{round(multitasks_write_summary['mean'], 2)}B/s  \n"
    elif (
        multitasks_write_summary["mean"] >= 1024
        and multitasks_write_summary["mean"] < 1024 * 1024
    ):
        content += f"平均值{round(multitasks_write_summary['mean'] / 1024, 2)}KB/s  \n"
    elif (
        multitasks_write_summary["mean"] >= 1024 * 1024
        and multitasks_write_summary["mean"] < 1024 * 1024 * 1024
    ):
        content += (
            f"平均值{round(multitasks_write_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        )
    elif multitasks_write_summary["mean"] >= 1024 * 1024 * 1024:
        content += f"平均值{round(multitasks_write_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
    # content += f"I/O读峰值速率：{multitasks_read_summary['max']}GB，中位数{multitasks_read_summary['median']}GB，平均值{multitasks_read_summary['mean']}GB  \n"
    # content += f"I/O写峰值速率：{multitasks_write_summary['max']}GB，中位数{multitasks_write_summary['median']}GB，平均值{multitasks_write_summary['mean']}GB  \n"
    content += f"功耗峰值：{multitasks_power_summary['max']}W，中位数{multitasks_power_summary['median']}W，平均值{multitasks_power_summary['mean']}W  \n"
    content += "&nbsp;\n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 节点性能打分：\n"
    content += f"测试软件：{software}  \n"
    content += f"单任务性能得分：{singletask_score}  \n"
    content += f"单任务计算用时稳定性得分：{stability_score}  \n"
    content += f"大批量任务计算用时得分：{multitasks_time_score}  \n"
    content += f"大批量任务计算能耗得分：{multitasks_energy_score}  \n"
    content += f"性能综合得分：{total_score}  \n"
    return content, total_score


def generate_report(
    softwares,
    multitasks_count,
    repeat,
    gpus,
    cpus,
    node_exporter_port,
    nvidia_gpu_exporter_port,
    serverip,
    server_port,
    partition,
    node,
):
    content = "# 单节点深度测评报告\n\n"
    content += "针对生命科学计算的高性能集群性能分析及评测框架，使用代表性生物学计算软件集合，通过Prometheus监控软件对运行时CPU、GPU、内存、IO等资源需求特征进行监控，使用计算用时、计算效率、CPU核时、GPU卡时等关键评价指标对集群性能进行测评和打分，帮助指导生命科学计算集群的建设、提升生命科学计算生产效率。  \n"
    content += "单节点深度测评报告中每款计算软件的测试内容和测试结果分为两大部分：  \n"
    content += "1. 运行特征及运行模式分析。包括单任务和大批量任务作业的运行特征分析、单任务使用不同数量GPU卡的并行计算效率分析（部分不支持多卡并行的软件除外）、大批量任务不同并行规模性能分析（部分不支持多卡并行的软件除外）、单任务计算用时稳定性分析等。这些分析结果用于帮助用户评估相关计算软件在测试节点上运行是否存在瓶颈、分析在测试节点配置下最高效或最经济的运行模式。\n"
    content += "2. 节点配置性能打分。使用运行模式分析中的计算用时、计算能耗、计算用时稳定性、分子动力学模拟性能（仅限分子动力学模拟软件）等指标进行多维度打分，并计算该软件在测试节点配置下的性能综合得分。  \n\n"
    content += "&nbsp;\n"
    content += "&nbsp;\n"

    scores = []
    weights = {}
    if os.path.exists("softwaresWeights.txt"):
        with open("softwaresWeights.txt", "r") as f:
            for line in f:
                line = line.strip().split()
                weights[line[0]] = float(line[1])
        for software in softwares:
            if software not in weights:
                weights[software] = 1.0
    else:
        for software in softwares:
            weights[software] = 1.0
    if "SPONGE" in softwares:
        content, sponge_score = sponge_report(
            content,
            multitasks_count,
            repeat,
            gpus,
            cpus,
            node_exporter_port,
            nvidia_gpu_exporter_port,
            serverip,
            server_port,
            partition,
            node,
        )
        scores.append(round((sponge_score * weights["SPONGE"]), 2))
    if "GROMACS" in softwares:
        content, gromacs_score = gromacs_report(
            content,
            multitasks_count,
            repeat,
            gpus,
            cpus,
            node_exporter_port,
            nvidia_gpu_exporter_port,
            serverip,
            server_port,
            partition,
            node,
        )
        scores.append(round((gromacs_score * weights["GROMACS"]), 2))
    if "AMBER" in softwares:
        content, amber_score = amber_report(
            content,
            multitasks_count,
            repeat,
            gpus,
            cpus,
            node_exporter_port,
            nvidia_gpu_exporter_port,
            serverip,
            server_port,
            partition,
            node,
        )
        scores.append(round((amber_score * weights["AMBER"]), 2))
    if "DSDP" in softwares:
        content, dsdp_score = dsdp_report(
            content,
            multitasks_count,
            repeat,
            gpus,
            cpus,
            node_exporter_port,
            nvidia_gpu_exporter_port,
            serverip,
            server_port,
            partition,
            node,
        )
        scores.append(round((dsdp_score * weights["DSDP"]), 2))

    cpu_info = get_cpu_info(partition, node)
    memory_info = get_memory_info(partition, node)
    filesystem_info = get_filesystem_info()
    gpu_info = get_gpu_info(partition, node)
    os_version = get_os_version(partition, node)
    kernel_version = get_kernel_version(partition, node)
    slurm_version = get_slurm_version()
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"\n\n### 单节点测评总得分:{round(sum(scores), 2)}\n\n"
    score_bar_path = make_score_bar(softwares, scores)
    content += f"![score_bar]({score_bar_path})  \n"
    content += "&nbsp;\n"
    content += "#### 测试环境：\n"
    content += f"GPU型号{gpu_info['gpu_name']}，GPU卡数{gpu_info['gpu_count']}，单卡显存{gpu_info['memory']}GB，CPU型号{cpu_info['model']}，CPU核心数{cpu_info['cores']}，内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']}，文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。操作系统版本{os_version}，内核版本{kernel_version}，作业调度系统{slurm_version}。  \n"
    content += "&nbsp;\n"
    for software in softwares:
        version = get_version(software)
        content += f"#### {software}\n"
        content += f"软件版本：{version}  \n"
        content += f"数据集：{dataset[software]}  \n"
        content += "&nbsp;\n"
    with open("SingleNodeMode_report.md", "w", encoding="utf-8") as md_file:
        md_file.write(content)


if __name__ == "__main__":
    # softwares = ["SPONGE", "GROMACS", "AMBER", "DSDP"]
    # scores = [99.96, 96.99, 99.92, 101.45]
    # print(round(sum(scores), 2))
    # make_score_bar(softwares, scores)

    gpus = 8
    cpus = 64
    multitasks_modes, multitasks_time_result = extract_gromacs_multitasks_result(
        gpus, cpus
    )
    print(multitasks_modes, multitasks_time_result)
    multitasks_modes, multitasks_time_result = extract_amber_multitasks_result(
        gpus, cpus
    )
    print(multitasks_modes, multitasks_time_result)
