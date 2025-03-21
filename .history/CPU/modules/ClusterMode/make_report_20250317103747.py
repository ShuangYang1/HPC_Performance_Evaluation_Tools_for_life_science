import os
import re
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from modules.ClusterMode.extract_mon import auto_extract
from modules.ClusterMode.hpc_resource_radar import extract_and_draw

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


def cal_score(software, taskcount, sec):
    score = round((factor[software] * taskcount / sec), 2)
    return score


def cal_energy_score(software, taskcount, energy):
    score = round((energy_factor[software] * taskcount / energy), 2)
    return score


def make_score_bar(softwares, scores):
    bar_path = "image/ClusterMode/score_bar.png"
    y_pos = np.arange(len(softwares))
    plt.figure(figsize=(10, 5), dpi=1000)
    plt.barh(y_pos, scores, height=0.4, color="skyblue", align="center")
    plt.title("集群测评软件得分", fontproperties=my_font)
    plt.xlim(0, max(max(scores)/3*4, 110))
    plt.xlabel("Score")
    plt.yticks(y_pos, softwares)
    # plt.axvline(x=100, color="r", linestyle="--", linewidth=1.5)
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


def generate_report(
    multitasks_count,
    partition,
    nodelist,
    cpus,
    cpus_per_task_dic,
    node_exporter_port,
    server_ip,
    server_port,
):
    content = "# 集群测评报告\n"
    content += "针对生命科学计算的高性能集群性能分析及评测框架，使用代表性生物学计算软件集合，通过Qmon监控软件对运行时CPU、内存、IO等资源需求特征进行监控，使用计算用时、计算效率、CPU核时等关键评价指标对集群性能进行测评和打分，帮助指导生命科学计算集群的建设、提升生命科学计算生产效率。  \n"
    content += "在运行集群测评之前，建议先进行单节点深度测评，了解软件运行特征和最高效的运行模式，使用每款软件最高效的运行模式进行集群测评。集群测评模式对每款计算软件在集群大规模并行计算时的运行特征、计算效率进行分析，用于评估计算软件在目前集群配置下运行是否存在瓶颈以及计算软件在目前集群配置下的计算效率。  \n"
    content += "每款软件包括以下结果：  \n"
    content += "1. 集群资源使用情况分析。使用Qmon监控软件记录计算软件在集群大规模计算时的资源使用情况，分析CPU、内存、I/O读写带宽的使用情况，输出计算软件计算全程的资源使用情况雷达图。每款计算软件会输出两张雷达图，第一张图是将各个指标缩放到0-100并在同一比例尺下对比展示，第二张图将雷达图分成若干层级，从内向外不同层级分别展示一个指标的变化情况。两幅图的指标和数据都是相同的，都是从顶端顺时针方向为起始，只是展示形式不同。\n"
    content += "2. 集群大规模并行计算打分。  \n"
    content += "$$\n"
    content += "计算用时得分=系数×\\frac{任务数量}{计算用时}  \n"
    content += "$$\n"
    content += "若测试平台支持功耗统计，则  \n"
    content += "$$\n"
    content += "\\begin{align*}\n"
    content += "能耗得分&=系数×\\frac{任务数量}{平均每个节点的能耗}\\\\\n"
    content += "综合得分&=\\frac{计算用时得分+能耗得分}{2}  \n"
    content += "\\end{align*}\n"
    content += "$$\n"
    content += "若测试平台不支持功耗统计，则计算用时得分即软件最终得分。  \n"
    content += "系数以目前测试的主流配置为基准（100分）来确定，得分是一个相对值且没有上限。  \n\n"
    content += "报告最终会计算总得分，总得分为各软件得分之和。  \n"
    content += "&nbsp;\n"
    content += "&nbsp;\n"

    cpu_info = get_cpu_info(partition,nodelist[0])
    memory_info = get_memory_info(partition,nodelist[0])
    filesystem_info = get_filesystem_info()
    os_version = get_os_version(partition,nodelist[0])
    kernel_version = get_kernel_version(partition,nodelist[0])
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
    effs=[]

    for software in softwares:
        version = get_version(software)
        start_timestamp, end_timestamp = get_multitasks_timestamp(
            f"log/ClusterMode/{software}/multitasks/"
        )
        mondir = f"mon/ClusterMode/{software}/multitasks"
        if not os.path.exists(mondir):
            os.makedirs(mondir)
        time_bins = auto_extract(
            node_exporter_port,
            server_ip,
            server_port,
            nodelist,
            start_timestamp,
            end_timestamp,
            mondir,
            11000,
        )
        image_path = f"image/ClusterMode/{software}"
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        radar_path = f"image/ClusterMode/{software}/radar.png"
        radar_split_path = f"image/ClusterMode/{software}/radar_split.png"
        (
            multitasks_cpu_summary,
            multitasks_mem_summary,
            multitasks_read_summary,
            multitasks_write_summary,
            multitasks_power_summary,
        ) = extract_and_draw(
            mondir,
            time_bins,
            nodelist,
            cpus,
            radar_path,
            radar_split_path,
            [software, multitasks_count, 1],
        )
        threads = cpus_per_task_dic[software]
        sec = end_timestamp - start_timestamp
        multitasks_score = cal_score(software, multitasks_count, sec)
        energy_score = cal_energy_score(
            software, multitasks_count, multitasks_power_summary["sum"] / len(nodelist)
        )
        score = round((multitasks_score + energy_score) / 2, 2)
        scores.append(round(weights[software] * score, 2))
        eff=score/len(nodelist)

        content += '<div STYLE="page-break-after: always;"></div>\n\n'
        content += f"## {software}\n"
        content += f"软件版本：{version}  \n"
        content += f"数据集：{dataset[software]}  \n"
        content += f"集群测试环境：测试集群节点数量{len(nodelist)}，CPU型号{cpu_info['model']}，CPU核心数{cpu_info['cores']}，内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']}，文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。操作系统版本{os_version}，内核版本{kernel_version}，作业调度系统{slurm_version}。  \n"

        content += "### 集群资源使用情况分析：\n"
        content += f"测试软件：{software}  \n"
        if software == "Bismark":
            content += f"并行模式：{threads} parallel  \n"
        else:
            content += f"并行模式：{threads}线程  \n"
        content += f"任务量：{multitasks_count}  \n"
        content += f"![radar]({radar_path})  \n"
        content += f"![radar_split]({radar_split_path})  \n"
        content += f"平均CPU使用率峰值：{multitasks_cpu_summary['max']}%，中位数{multitasks_cpu_summary['median']}%，平均值{multitasks_cpu_summary['mean']}%  \n"
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
        content += "&nbsp;\n"
        content += f"计算用时得分：{multitasks_score}\n"
        content += f"功耗得分：{energy_score}\n"
        content += f"#### 集群大规模并行计算得分：{score}\n"
        

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"### 集群测评总得分:{round(sum(scores), 2)}"
    score_bar_path = make_score_bar(softwares, scores)
    content += f"![score_bar]({score_bar_path})  \n"
    content += "&nbsp;\n"
    content += "#### 集群测试环境：\n"
    content += f"测试集群节点数量{len(nodelist)}，CPU型号{cpu_info['model']}，CPU核心数{cpu_info['cores']}，内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']},文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。操作系统版本{os_version}，内核版本{kernel_version}，作业调度系统{slurm_version}。  \n"
    content += "&nbsp;\n"
    for software in softwares:
        version = get_version(software)
        content += f"#### {software}\n"
        content += f"软件版本：{version}  \n"
        content += f"数据集：{dataset[software]}  \n"
        content += "&nbsp;\n"
    with open("ClusterMode_report.md", "w", encoding="utf-8") as md_file:
        md_file.write(content)


if __name__ == "__main__":
    multitasks_count = 100
    nodelist = [
        "c02b06n02a",
        "c02b06n02c",
        "c02b06n02d",
        "c02b06n03b",
        "c02b06n03d",
        "dcu-1",
        "dcu-2",
    ]
    generate_report(multitasks_count, nodelist)
