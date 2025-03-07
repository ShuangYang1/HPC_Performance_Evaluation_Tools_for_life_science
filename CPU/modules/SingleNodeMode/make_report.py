import os
import re
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from modules.SingleNodeMode.cal_multitasks_scale import (
    cal_gatk_multitasks_scale,
    cal_multitasks_scale,
)
from modules.SingleNodeMode.cal_singletask_scale import cal_singletask_scale
from modules.SingleNodeMode.extract_mon import auto_extract
from modules.SingleNodeMode.hpc_resource_radar import extract_and_draw

factor = {
    "BWA": 23423.42708,
    "SPAdes": 45043.43444,
    "Bismark": 81520.67891,
    "STAR": 14087.23871,
    "Cellranger": 51593.33481,
    "GATK": 41140.9086,
}

energy_factor = {
    "BWA": 9289570.773,
    "SPAdes": 17179839.95,
    "Bismark": 32916832.48,
    "STAR": 5734608.912,
    "Cellranger": 20161501.01,
    "GATK": 17078688.93,
}

dataset = {
    "BWA": "千人基因组计划个体NA12750全基因组测序（4.8GB）；参考基因组：NCBI36（3.0GB）",
    "SPAdes": "大肠杆菌K-12菌株MG1655基因组测序（6.1GB）",
    "Bismark": "IMR90细胞系全基因组shotgun-bisulfite测序（4.0GB）；参考基因组NCBI36（3.0GB）",
    "STAR": "小鼠胚胎LINE1抑制后RNA-seq测序（11GB）；参考基因组GRCm39（2.6GB）",
    "Cellranger": "人外周血单核细胞（PBMC）的1000个细胞的单细胞测序数据，包含淋巴细胞（T细胞、B细胞和NK细胞）和单核细胞（5.17GB）；参考基因组GRCh38（11GB）",
    "GATK": "千人基因组计划个体NA12750全基因组测序比对数据（4.7GB）；参考基因组GRCh38（3.1GB）",
}

softwares = ["BWA", "SPAdes", "Bismark", "STAR", "Cellranger", "GATK"]
font_path = "modules/SingleNodeMode/SimHei.ttf"
my_font = matplotlib.font_manager.FontProperties(fname=font_path)


def get_version(software):
    try:
        if software == "SPAdes":
            result = subprocess.check_output(
                ["spades.py", "--version"], stderr=subprocess.STDOUT
            ).decode()
            return result.split()[-1].strip()
        elif software == "STAR":
            result = subprocess.check_output(
                ["STAR", "--version"], stderr=subprocess.STDOUT
            ).decode()
            return result.strip()
        elif software == "GATK":
            result = subprocess.check_output(
                ["gatk", "--version"], stderr=subprocess.STDOUT
            ).decode()
            for line in result.splitlines():
                if "The Genome Analysis Toolkit" in line:
                    return line.split()[-1].strip()
        elif software == "Bismark":
            result = subprocess.check_output(
                ["bismark", "--version"], stderr=subprocess.STDOUT
            ).decode()
            for line in result.splitlines():
                if "Version" in line:
                    return line.split()[-1].strip()
        elif software == "BWA":
            result = subprocess.run(
                ["bwa"], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True
            ).stderr
            for line in result.splitlines():
                if "Version" in line:
                    return line.split()[-1].strip()
        elif software == "Cellranger":
            result = subprocess.check_output(
                ["cellranger", "--version"], stderr=subprocess.STDOUT
            ).decode()
            return result.split("-")[-1].strip()
    except subprocess.CalledProcessError as e:
        print(f"获取{software}版本信息时出错: {e.output.decode('utf-8')}")
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


def convert_to_sec(time_str):
    minutes, seconds = time_str.split("m")
    seconds = seconds.replace("s", "")
    total_seconds = int(minutes) * 60 + float(seconds)
    return total_seconds


def cal_sec(filename):
    with open(filename, "r") as infile:
        content = infile.read()
        pattern = re.compile(r"real\t([0-9]+m[0-9]+.[0-9]+s)")
        time = pattern.search(content).group(1)
        sec = convert_to_sec(time)
        return sec


def make_singletask_linechart(software):
    logdir = "log/SingleNodeMode/" + software + "/singletask/"
    files = os.listdir(logdir)
    logfiles = [i for i in files if i.endswith(".log")]
    logs = sorted(logfiles)
    path = "image/SingleNodeMode/" + software
    if not os.path.exists(path):
        os.makedirs(path)
    time_dict = {}
    time, cputime = [], []
    for f in logs:
        if f.count("_") == 0:
            if software == "STAR":
                with open(logdir + f, "r") as infile:
                    content = infile.read()
                    if "ERROR" in content:
                        continue
                    else:
                        sec = cal_sec(logdir + f)
                        thread = int(f.split("_")[-1].split(".")[0])
                        time_dict[thread] = sec
            else:
                sec = cal_sec(logdir + f)
                thread = int(f.split("_")[-1].split(".")[0])
                time_dict[thread] = sec
    threads = sorted(time_dict.keys())
    for t in threads:
        time.append(time_dict[t])
        if software == "Bismark":
            cputime.append(t * time_dict[t] * 4)
        else:
            cputime.append(t * time_dict[t])
    plt.figure(figsize=(10, 5), dpi=1000)
    plt.plot(threads, time, marker="o")
    plt.title(software + " 单任务不同线程数计算用时", fontproperties=my_font)
    plt.xlabel("threads")
    plt.ylabel("Time(s)")
    time_linechart_path = path + "/singletask_time_linechart.png"
    plt.savefig(time_linechart_path)
    plt.close()
    plt.figure(figsize=(10, 5), dpi=1000)
    plt.plot(threads, cputime, marker="o")
    plt.title(software + " 单任务不同线程数CPU核时", fontproperties=my_font)
    plt.xlabel("threads")
    plt.ylabel("CPU time(s)")
    cputime_linechart_path = path + "/singletask_cputime_linechart.png"
    plt.savefig(cputime_linechart_path)
    plt.close()
    return time_linechart_path, cputime_linechart_path, threads, time, cputime


def cal_speedup(threads, time, cputime):
    table = "| 线程数 | 计算用时（s） | 加速比 | 计算效率 | CPU核时（s） |\n| ---- | ---- | ---- | ---- | ---- |\n"
    for i in range(len(threads)):
        table += f"| {threads[i]} | {round(time[i], 2)} | {round(time[0] / time[i], 2)} | {round(time[0] / time[i] / threads[i] * 100, 2)}% | {round(cputime[i], 2)} |\n"
    return table


def cal_multi_best_score(software, taskcount, sec):
    score = round((factor[software] * taskcount / sec), 2)
    return score


def cal_energy_score(software, taskcount, energy):
    score = round((energy_factor[software] * taskcount / energy), 2)
    return score


def find_mode_without_err(directory, software, taskcount, cpus, mem):
    file_without_err = []
    error_keyword = {
        "BWA": "Killed",
        "Bismark": "ERR",
        "Cellranger": "killed",
        "GATK": "done",
        "SPAdes": "Error",
        "STAR": "Killed",
    }
    parallelmodes = sorted(
        [entry.name for entry in Path(directory).iterdir() if entry.is_dir()]
    )
    for parallelmode in parallelmodes:
        root = os.path.join(directory, parallelmode)
        listdir = os.listdir(root)
        for file in listdir:
            if file.endswith(".log"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    if software == "GATK":
                        if "done" in f.read():
                            file_without_err.append(parallelmode)
                    elif software == "STAR":
                        file_content = f.read()
                        if (
                            error_keyword[software] not in file_content
                            and "ERROR" not in file_content
                        ):
                            file_without_err.append(parallelmode)
                    else:
                        if error_keyword[software] not in f.read():
                            file_without_err.append(parallelmode)
    count_dict = {}
    for item in file_without_err:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1
    mode_without_err = [key for key, value in count_dict.items() if value == taskcount]
    mode_with_err = []
    if software == "GATK":
        modes = [cal_gatk_multitasks_scale(cpus, mem)]
    elif software == "Bismark":
        modes = cal_multitasks_scale(cpus // 4)
    else:
        modes = cal_multitasks_scale(cpus)
    for mode in modes:
        if mode not in mode_without_err:
            mode_with_err.append(mode)
    return mode_without_err, mode_with_err


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


def multitasks_best_parallelmode(software, taskcount, cpus, mem):
    logdir = "log/SingleNodeMode/" + software + "/multitasks/"
    mode_without_err, mode_with_err = find_mode_without_err(
        logdir, software, taskcount, cpus, mem
    )
    parallel, time = [], []
    parallelmodes = sorted(
        [entry.name for entry in Path(logdir).iterdir() if entry.is_dir()]
    )
    for parallelmode in parallelmodes:
        if parallelmode in mode_without_err:
            parallel.append(parallelmode)
            start_timestamp, end_timestamp = get_multitasks_timestamp(
                logdir + parallelmode + "/"
            )
            time.append(end_timestamp - start_timestamp)
    print(software, mode_without_err, mode_with_err)
    best_time = min(time)
    best_mode = parallel[time.index(best_time)]
    return best_mode, best_time, parallel, time, mode_with_err


def make_multitasks_linechart(software, taskcount, cpus, mem):
    best_mode, best_time, parallel, time, mode_with_err = multitasks_best_parallelmode(
        software, taskcount, cpus, mem
    )
    path = "image/SingleNodeMode/" + software
    if not os.path.exists(path):
        os.makedirs(path)
    plt.figure(figsize=(10, 5), dpi=1000)
    plt.plot(parallel, time, marker="o")
    plt.title(software + " 多任务不同并行模式计算用时", fontproperties=my_font)
    plt.xlabel("threads_parallel")
    plt.ylabel("Time(s)")
    time_linechart_path = path + "/multitasks_time_linechart.png"
    plt.savefig(time_linechart_path)
    plt.close()
    return time_linechart_path, best_mode, best_time, parallel, time, mode_with_err


def make_singletask_violin(software, repeats):
    logdir = "log/SingleNodeMode/" + software + "/singletask/"
    files = os.listdir(logdir)
    logs = [i for i in files if i.endswith(".log")]
    path = "image/SingleNodeMode/" + software
    if not os.path.exists(path):
        os.makedirs(path)
    time = []
    for f in logs:
        if f.count("_") == 1:
            sec = cal_sec(logdir + f)
            time.append(sec)
            thread = f.split("_")[0]
    time.append(cal_sec(f"{logdir}{thread}.log"))
    plt.figure(figsize=(10, 5), dpi=1000)
    violin_parts = plt.violinplot(time, showmeans=True, showmedians=True, widths=0.3)
    violin_parts["cmeans"].set_color("green")
    violin_parts["cmedians"].set_color("red")
    plt.title(
        f"{software} 单任务{thread}线程{repeats}次计算用时分布", fontproperties=my_font
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


def hostname_and_timestamp(directory):
    listdir = os.listdir(directory)
    for file in listdir:
        if file.endswith("_2.hostname"):
            with open(directory + "/" + file, "r") as f:
                hostname = f.read().strip()
            with open(
                directory + "/" + file.replace("_2.hostname", "_2.start.timestamp"), "r"
            ) as f:
                start_timestamp = int(f.read())
            with open(
                directory + "/" + file.replace("_2.hostname", "_2.end.timestamp"), "r"
            ) as f:
                end_timestamp = int(f.read())
            best_thread = int(file.split("_")[0])
            return hostname, start_timestamp, end_timestamp, best_thread


def generate_report(
    multitasks_count,
    repeats,
    cpus,
    mem,
    partition,
    nodelist,
    node_exporter_port,
    server_ip,
    server_port,
):
    content = "# 单节点深度测评报告\n\n"
    content += "针对生命科学计算的高性能集群性能分析及评测框架，使用代表性生物学计算软件集合，通过Qmon监控软件对运行时CPU、内存、IO等资源需求特征进行监控，使用计算用时、计算效率、CPU核时等关键评价指标对集群性能进行测评和打分，帮助指导生命科学计算集群的建设、提升生命科学计算生产效率。  \n"
    content += "单节点深度测评模式对每款计算软件的运行特征、运行模式进行分析，用于评估计算软件在目前节点配置下运行是否存在瓶颈、分析在目前节点配置下最高效的运行模式。  \n\n"
    content += "每款软件包括以下分析：  \n"
    content += "1. 软件运行特征分析。使用Qmon监控软件记录计算软件使用全部CPU核心时的运行特征，分析CPU、内存、I/O读写带宽的使用情况，输出计算软件一次计算全程的资源使用情况雷达图。每款计算软件的单任务运行特征分析会输出两张雷达图，第一张图是将各个指标缩放到0-100并在同一比例尺下对比展示，第二张图将雷达图分成若干层级，从内向外不同层级分别展示一个指标的变化情况。两幅图的指标和数据都是相同的，都是从顶端顺时针方向为起始，只是展示形式不同。\n"
    content += "2. 单任务不同线程加速比分析。使用不同线程数运行计算软件，统计计算用时，输出不同线程数计算用时的折线图。统计CPU核时（计算用时×线程数），输出不同线程数计算CPU核时的折线图。统计不同线程数计算的加速比和计算效率。  \n"
    content += "$$\n"
    content += "\\begin{align*}\n"
    content += "加速比&=\\frac{单线程计算用时}{该线程数计算用时}\\\\\n"
    content += "计算效率&=\\frac{加速比}{线程数}*100\\%\n"
    content += "\\end{align*}\n"
    content += "$$\n"
    content += "3. 多任务并行模式分析。使用不同CPU核心数×并行任务数的组合测试计算软件多任务并行通量，输出不同运行模式计算用时的折线图。提供在目前节点配置下多任务并行时最佳的运行模式和节点配置得分。  \n"
    content += "$$\n"
    content += "\\begin{align*}\n"
    content += "计算用时得分&=系数×\\frac{任务数量}{最佳并行模式计算用时}  \n"
    content += "\\end{align*}\n"
    content += "$$\n"
    content += "系数以目前测试的主流配置为基准（100分）来确定，得分是一个相对值且没有上限。  \n"
    content += "4. 多任务并行资源使用情况分析。使用Qmon监控软件记录计算软件在最佳并行模式下的资源使用情况，分析CPU、内存、I/O读写带宽的使用情况和功耗（部分平台暂不支持功耗统计），输出雷达图，计算能耗得分。（雷达图的说明同单任务运行特征分析）  \n"
    content += "$$\n"
    content += "\\begin{align*}\n"
    content += "能耗得分&=系数×\\frac{任务数量}{最佳并行模式能耗}  \n"
    content += "\\end{align*}\n"
    content += "$$\n"
    content += "5. 计算用时稳定性分析。多次测试单任务使用全部CPU核心的计算用时，统计多次测试结果的分布，输出小提琴图和计算用时稳定性得分。  \n"
    content += "$$\n"
    content += "\\begin{align*}\n"
    content += "计算用时稳定性得分=max((1-\\frac{标准差}{均值})*100,0)  \n"
    content += "\\end{align*}\n"
    content += "$$\n"
    content += "每款软件最后会有一个综合得分  \n"
    content += "$$\n"
    content += "\\begin{align*}\n"
    content += (
        "综合得分=\\frac{计算用时得分+能耗得分}{2}×\\frac{计算用时稳定性得分}{100}  \n"
    )
    content += "\\end{align*}\n"
    content += "$$\n"
    content += "部分平台暂不支持功耗统计，没有能耗得分，则  \n"
    content += "$$\n"
    content += "\\begin{align*}\n"
    content += "综合得分=计算用时得分×\\frac{计算用时稳定性得分}{100}  \n"
    content += "\\end{align*}\n"
    content += "$$\n"
    content += "报告最终会计算总得分，总得分为各软件综合得分之和。  \n"
    content += "&nbsp;\n"
    content += "&nbsp;\n"

    cpu_info = get_cpu_info(partition, nodelist[0])
    memory_info = get_memory_info(partition, nodelist[0])
    filesystem_info = get_filesystem_info()
    os_version = get_os_version(partition, nodelist[0])
    kernel_version = get_kernel_version(partition, nodelist[0])
    slurm_version = get_slurm_version()

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

    singletask_threads_count = {}
    singletask_best_threads = {}
    singletask_best_time = {}
    singletask_best_cputime = {}
    multitasks_mode_without_error_count = {}
    multitasks_mode_with_error_count = {}
    multitasks_best_mode = {}
    multitasks_best_time = {}
    multitasks_best_cputime = {}

    for software in softwares:
        version = get_version(software)
        (
            singletask_hostname,
            singletask_start_timestamp,
            singletask_end_timestamp,
            singletask_best_thread,
        ) = hostname_and_timestamp(f"log/SingleNodeMode/{software}/singletask")
        singletask_mondir = f"mon/SingleNodeMode/{software}/singletask"
        if not os.path.exists(singletask_mondir):
            os.makedirs(singletask_mondir)
        time_bins = auto_extract(
            node_exporter_port,
            server_ip,
            server_port,
            singletask_hostname,
            singletask_start_timestamp,
            singletask_end_timestamp,
            singletask_mondir,
            11000,
        )
        image_path = f"image/SingleNodeMode/{software}/"
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        singletask_radar_path = f"image/SingleNodeMode/{software}/singletask_radar.png"
        singletask_radar_split_path = (
            f"image/SingleNodeMode/{software}/singletask_radar_split.png"
        )
        (
            singletask_cpu_summary,
            singletask_mem_summary,
            singletask_read_summary,
            singletask_write_summary,
            singletask_power_summary,
        ) = extract_and_draw(
            singletask_mondir,
            time_bins,
            cpus,
            singletask_radar_path,
            singletask_radar_split_path,
            [software, "singletask", singletask_best_thread],
        )
        (
            singletask_time_linechart_path,
            singletask_cputime_linechart_path,
            singletask_threads,
            singletask_time,
            singletask_cputime,
        ) = make_singletask_linechart(software)
        singletask_threads_count[software] = len(singletask_threads)
        singletask_best_time[software] = min(singletask_time)
        singletask_best_cputime[software] = singletask_cputime[
            singletask_time.index(singletask_best_time[software])
        ]
        singletask_best_threads[software] = singletask_threads[
            singletask_time.index(singletask_best_time[software])
        ]
        singletask_speedup_table = cal_speedup(
            singletask_threads, singletask_time, singletask_cputime
        )

        if software != "GATK":
            (
                multitasks_time_linechart_path,
                best_mode,
                best_time,
                parallel,
                time,
                mode_with_err,
            ) = make_multitasks_linechart(software, multitasks_count, cpus, mem)
        else:
            best_mode, best_time, parallel, time, mode_with_err = (
                multitasks_best_parallelmode(software, multitasks_count, cpus, mem)
            )

        multitasks_mode_without_error_count[software] = len(parallel)
        multitasks_mode_with_error_count[software] = len(mode_with_err)
        multitasks_best_mode[software] = best_mode
        multitasks_best_time[software] = best_time
        if software == "GATK":
            parallel_count = int(cal_gatk_multitasks_scale(cpus, mem).split("_")[1])
            multitasks_best_cputime[software] = best_time * parallel_count
        else:
            multitasks_best_cputime[software] = best_time * cpus
        multitasks_score = cal_multi_best_score(software, multitasks_count, best_time)
        with open(
            f"log/SingleNodeMode/{software}/multitasks/{best_mode}/hostname", "r"
        ) as f:
            multitasks_hostname = f.read().strip()
        multitasks_start_timestamp, multitasks_end_timestamp = get_multitasks_timestamp(
            f"log/SingleNodeMode/{software}/multitasks/{best_mode}/"
        )
        multitasks_mondir = f"mon/SingleNodeMode/{software}/multitasks"
        if not os.path.exists(multitasks_mondir):
            os.makedirs(multitasks_mondir)
        time_bins = auto_extract(
            node_exporter_port,
            server_ip,
            server_port,
            multitasks_hostname,
            multitasks_start_timestamp,
            multitasks_end_timestamp,
            multitasks_mondir,
            11000,
        )
        multitasks_radar_path = f"image/SingleNodeMode/{software}/multitasks_radar.png"
        multitasks_radar_split_path = (
            f"image/SingleNodeMode/{software}/multitasks_radar_split.png"
        )
        (
            multitasks_cpu_summary,
            multitasks_mem_summary,
            multitasks_read_summary,
            multitasks_write_summary,
            multitasks_power_summary,
        ) = extract_and_draw(
            multitasks_mondir,
            time_bins,
            cpus,
            multitasks_radar_path,
            multitasks_radar_split_path,
            [software, "multitasks", multitasks_best_mode],
        )
        singletask_violin_path, singletask_stability_score = make_singletask_violin(
            software, repeats
        )

        stability_score = round(singletask_stability_score, 2)
        energy_score = cal_energy_score(
            software, multitasks_count, multitasks_power_summary["sum"]
        )
        score = round((multitasks_score + energy_score) / 2 * stability_score / 100, 2)
        scores.append(round(weights[software] * score, 2))

        content += '<div STYLE="page-break-after: always;"></div>\n\n'
        content += f"## {software}\n"
        content += f"软件版本：{version}  \n"
        content += f"数据集：{dataset[software]}  \n"
        content += f"测试环境：CPU型号{cpu_info['model']}，CPU核心数{cpu_info['cores']}，内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']}，文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。操作系统版本{os_version}，内核版本{kernel_version}，作业调度系统{slurm_version}。  \n\n"
        content += "### 运行特征分析：\n"
        content += f"![singletask_radar]({singletask_radar_path})  \n"
        content += '<div STYLE="page-break-after: always;"></div>\n\n'
        content += f"![singletask_radar_split]({singletask_radar_split_path})  \n"
        content += f"CPU使用率峰值：{singletask_cpu_summary['max']}%，中位数{singletask_cpu_summary['median']}%，平均值{singletask_cpu_summary['mean']}%  \n"
        content += f"内存使用峰值：{singletask_mem_summary['max']}GB，中位数{singletask_mem_summary['median']}GB，平均值{singletask_mem_summary['mean']}GB  \n"
        if singletask_read_summary["max"] < 1024:
            content += f"I/O读峰值速率：{round(singletask_read_summary['max'], 2)}B/s，"
        elif (
            singletask_read_summary["max"] >= 1024
            and singletask_read_summary["max"] < 1024 * 1024
        ):
            content += f"I/O读峰值速率：{round(singletask_read_summary['max'] / 1024, 2)}KB/s，"
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
            content += (
                f"中位数{round(singletask_read_summary['median'] / 1024, 2)}KB/s，"
            )
        elif (
            singletask_read_summary["median"] >= 1024 * 1024
            and singletask_read_summary["median"] < 1024 * 1024 * 1024
        ):
            content += f"中位数{round(singletask_read_summary['median'] / 1024 / 1024, 2)}MB/s，"
        elif singletask_read_summary["median"] >= 1024 * 1024 * 1024:
            content += f"中位数{round(singletask_read_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
        if singletask_read_summary["mean"] < 1024:
            content += f"平均值{round(singletask_read_summary['mean'], 2)}B/s  \n"
        elif (
            singletask_read_summary["mean"] >= 1024
            and singletask_read_summary["mean"] < 1024 * 1024
        ):
            content += (
                f"平均值{round(singletask_read_summary['mean'] / 1024, 2)}KB/s  \n"
            )
        elif (
            singletask_read_summary["mean"] >= 1024 * 1024
            and singletask_read_summary["mean"] < 1024 * 1024 * 1024
        ):
            content += f"平均值{round(singletask_read_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        elif singletask_read_summary["mean"] >= 1024 * 1024 * 1024:
            content += f"平均值{round(singletask_read_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
        if singletask_write_summary["max"] < 1024:
            content += (
                f"I/O写峰值速率：{round(singletask_write_summary['max'], 2)}B/s，"
            )
        elif (
            singletask_write_summary["max"] >= 1024
            and singletask_write_summary["max"] < 1024 * 1024
        ):
            content += f"I/O写峰值速率：{round(singletask_write_summary['max'] / 1024, 2)}KB/s，"
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
            content += (
                f"中位数{round(singletask_write_summary['median'] / 1024, 2)}KB/s，"
            )
        elif (
            singletask_write_summary["median"] >= 1024 * 1024
            and singletask_write_summary["median"] < 1024 * 1024 * 1024
        ):
            content += f"中位数{round(singletask_write_summary['median'] / 1024 / 1024, 2)}MB/s，"
        elif singletask_write_summary["median"] >= 1024 * 1024 * 1024:
            content += f"中位数{round(singletask_write_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
        if singletask_write_summary["mean"] < 1024:
            content += f"平均值{round(singletask_write_summary['mean'], 2)}B/s  \n"
        elif (
            singletask_write_summary["mean"] >= 1024
            and singletask_write_summary["mean"] < 1024 * 1024
        ):
            content += (
                f"平均值{round(singletask_write_summary['mean'] / 1024, 2)}KB/s  \n"
            )
        elif (
            singletask_write_summary["mean"] >= 1024 * 1024
            and singletask_write_summary["mean"] < 1024 * 1024 * 1024
        ):
            content += f"平均值{round(singletask_write_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        elif singletask_write_summary["mean"] >= 1024 * 1024 * 1024:
            content += f"平均值{round(singletask_write_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
        # content += f"I/O读峰值速率：{singletask_read_summary['max']}GB，中位数{singletask_read_summary['median']}GB，平均值{singletask_read_summary['mean']}GB  \n"
        # content += f"I/O写峰值速率：{singletask_write_summary['max']}GB，中位数{singletask_write_summary['median']}GB，平均值{singletask_write_summary['mean']}GB  \n"
        content += f"功耗峰值：{singletask_power_summary['max']}W，中位数{singletask_power_summary['median']}W，平均值{singletask_power_summary['mean']}W  \n"
        content += "&nbsp;\n"
        content += '<div STYLE="page-break-after: always;"></div>\n\n'
        content += "### 单任务不同线程加速比分析：\n"
        content += f"![singletask_time_linechart]({singletask_time_linechart_path})  \n"
        content += (
            f"![singletask_cputime_linechart]({singletask_cputime_linechart_path})  \n"
        )
        content += "&nbsp;\n"
        if software == "STAR":
            star_error_threads = []
            threads = sorted(cal_singletask_scale(cpus, software), reverse=True)
            for t in threads:
                log_file = f"log/SingleNodeMode/{software}/singletask/{t}.log"
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        file_content = f.read()
                        if "ERROR" not in file_content:
                            continue
                        else:
                            star_error_threads.append(t)
                            continue
            for i in sorted(star_error_threads):
                content += f"{i}线程任务失败，请检查任务输出日志log/SingleNodeMode/STAR/singletask/{i}.log。  \n"
        content += singletask_speedup_table
        content += "### 多任务并行模式分析：\n"
        if software == "GATK":
            gatk_parallel = cal_gatk_multitasks_scale(cpus, mem)
            if int(gatk_parallel.split("_")[1]) < cpus:
                content += f"节点内存受限，最多并行{gatk_parallel.split('_')[1]}个任务，得分为{multitasks_score}。  \n"
            else:
                content += f"节点内存充足，同时可并行{cpus}个任务，得分为{multitasks_score}。  \n"
        else:
            content += (
                f"![multitasks_time_linechart]({multitasks_time_linechart_path})  \n"
            )
            content += f"最佳并行模式为{best_mode.split('_')[0]}线程，得分为{multitasks_score}。  \n"
        content += "&nbsp;\n"
        content += '<div STYLE="page-break-after: always;"></div>\n\n'
        content += "\n### 多任务并行资源使用情况分析：\n"
        content += f"![multitasks_radar]({multitasks_radar_path})  \n"
        content += '<div STYLE="page-break-after: always;"></div>\n\n'
        content += f"![multitasks_radar_split]({multitasks_radar_split_path})  \n"
        content += f"CPU使用率峰值：{multitasks_cpu_summary['max']}%，中位数{multitasks_cpu_summary['median']}%，平均值{multitasks_cpu_summary['mean']}%  \n"
        content += f"内存使用峰值：{multitasks_mem_summary['max']}GB，中位数{multitasks_mem_summary['median']}GB，平均值{multitasks_mem_summary['mean']}GB  \n"
        if multitasks_read_summary["max"] < 1024:
            content += f"I/O读峰值速率：{round(multitasks_read_summary['max'], 2)}B/s，"
        elif (
            multitasks_read_summary["max"] >= 1024
            and multitasks_read_summary["max"] < 1024 * 1024
        ):
            content += f"I/O读峰值速率：{round(multitasks_read_summary['max'] / 1024, 2)}KB/s，"
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
            content += (
                f"中位数{round(multitasks_read_summary['median'] / 1024, 2)}KB/s，"
            )
        elif (
            multitasks_read_summary["median"] >= 1024 * 1024
            and multitasks_read_summary["median"] < 1024 * 1024 * 1024
        ):
            content += f"中位数{round(multitasks_read_summary['median'] / 1024 / 1024, 2)}MB/s，"
        elif multitasks_read_summary["median"] >= 1024 * 1024 * 1024:
            content += f"中位数{round(multitasks_read_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
        if multitasks_read_summary["mean"] < 1024:
            content += f"平均值{round(multitasks_read_summary['mean'], 2)}B/s  \n"
        elif (
            multitasks_read_summary["mean"] >= 1024
            and multitasks_read_summary["mean"] < 1024 * 1024
        ):
            content += (
                f"平均值{round(multitasks_read_summary['mean'] / 1024, 2)}KB/s  \n"
            )
        elif (
            multitasks_read_summary["mean"] >= 1024 * 1024
            and multitasks_read_summary["mean"] < 1024 * 1024 * 1024
        ):
            content += f"平均值{round(multitasks_read_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        elif multitasks_read_summary["mean"] >= 1024 * 1024 * 1024:
            content += f"平均值{round(multitasks_read_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
        if multitasks_write_summary["max"] < 1024:
            content += (
                f"I/O写峰值速率：{round(multitasks_write_summary['max'], 2)}B/s，"
            )
        elif (
            multitasks_write_summary["max"] >= 1024
            and multitasks_write_summary["max"] < 1024 * 1024
        ):
            content += f"I/O写峰值速率：{round(multitasks_write_summary['max'] / 1024, 2)}KB/s，"
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
            content += (
                f"中位数{round(multitasks_write_summary['median'] / 1024, 2)}KB/s，"
            )
        elif (
            multitasks_write_summary["median"] >= 1024 * 1024
            and multitasks_write_summary["median"] < 1024 * 1024 * 1024
        ):
            content += f"中位数{round(multitasks_write_summary['median'] / 1024 / 1024, 2)}MB/s，"
        elif multitasks_write_summary["median"] >= 1024 * 1024 * 1024:
            content += f"中位数{round(multitasks_write_summary['median'] / 1024 / 1024 / 1024, 2)}GB/s，"
        if multitasks_write_summary["mean"] < 1024:
            content += f"平均值{round(multitasks_write_summary['mean'], 2)}B/s  \n"
        elif (
            multitasks_write_summary["mean"] >= 1024
            and multitasks_write_summary["mean"] < 1024 * 1024
        ):
            content += (
                f"平均值{round(multitasks_write_summary['mean'] / 1024, 2)}KB/s  \n"
            )
        elif (
            multitasks_write_summary["mean"] >= 1024 * 1024
            and multitasks_write_summary["mean"] < 1024 * 1024 * 1024
        ):
            content += f"平均值{round(multitasks_write_summary['mean'] / 1024 / 1024, 2)}MB/s  \n"
        elif multitasks_write_summary["mean"] >= 1024 * 1024 * 1024:
            content += f"平均值{round(multitasks_write_summary['mean'] / 1024 / 1024 / 1024, 2)}GB/s  \n"
        # content += f"I/O读峰值速率：{multitasks_read_summary['max']}GB，中位数{multitasks_read_summary['median']}GB，平均值{multitasks_read_summary['mean']}GB  \n"
        # content += f"I/O写峰值速率：{multitasks_write_summary['max']}GB，中位数{multitasks_write_summary['median']}GB，平均值{multitasks_write_summary['mean']}GB  \n"
        content += f"功耗峰值：{multitasks_power_summary['max']}W，中位数{multitasks_power_summary['median']}W，平均值{multitasks_power_summary['mean']}W  \n"
        content += f"能耗得分：{energy_score}  \n"
        content += "&nbsp;\n"
        content += '<div STYLE="page-break-after: always;"></div>\n\n'
        content += "### 计算用时稳定性分析：\n"
        content += f"![singletask_violin]({singletask_violin_path})  \n"
        content += f"计算用时稳定性得分:{stability_score}  \n"
        content += f"#### {software}总得分:{score}  \n"
        content += "&nbsp;\n"
        content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"\n\n### 单节点测评总得分:{round(sum(scores), 2)}\n\n"
    score_bar_path = make_score_bar(softwares, scores)
    content += f"![score_bar]({score_bar_path})  \n"
    content += "&nbsp;\n"
    content += "#### 测试环境：\n"
    content += f"CPU型号{cpu_info['model']}，CPU核心数{cpu_info['cores']}，内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']},文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。操作系统版本{os_version}，内核版本{kernel_version}，作业调度系统{slurm_version}。  \n"
    content += "&nbsp;\n"
    for software in softwares:
        version = get_version(software)
        content += f"#### {software}\n"
        content += f"软件版本：{version}  \n"
        content += f"数据集：{dataset[software]}  \n"
        if software == "STAR":
            content += f"单任务测试{singletask_threads_count[software] + len(star_error_threads)}种不同线程数，失败{len(star_error_threads)}次。最佳线程数为{singletask_best_threads[software]}，用时{round(singletask_best_time[software], 2)}s，CPU核时{round(singletask_best_cputime[software], 2)}s。  \n"
        else:
            content += f"单任务测试{singletask_threads_count[software]}种不同线程数。最佳线程数为{singletask_best_threads[software]}，用时{round(singletask_best_time[software], 2)}s，CPU核时{round(singletask_best_cputime[software], 2)}s。  \n"
        content += f"多任务并行测试{multitasks_mode_without_error_count[software] + multitasks_mode_with_error_count[software]}种不同规模，失败{multitasks_mode_with_error_count[software]}次。最佳并行模式为{multitasks_best_mode[software]}，用时{round(multitasks_best_time[software], 2)}s，CPU核时{round(multitasks_best_cputime[software], 2)}s。  \n"
        content += "&nbsp;\n"
    content += "任务失败通常是任务资源不足导致的，如内存不足、文件描述符数量不足等，请查看相应软件的错误日志。  \n"
    with open("SingleNodeMode_report.md", "w", encoding="utf-8") as md_file:
        md_file.write(content)


if __name__ == "__main__":
    multitasks_count = 50
    repeats = 10
    cpus = 128
    mem = 1500000
    generate_report(multitasks_count, repeats, cpus, mem)
    # scores = [96.73, 122.43, 129.64, 111.72, 130.55, 86.77]
    # make_score_bar(softwares, scores)
