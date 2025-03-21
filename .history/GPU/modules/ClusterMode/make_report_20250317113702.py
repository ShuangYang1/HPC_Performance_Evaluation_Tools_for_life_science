import os
import re
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from modules.ClusterMode.extract_mon import auto_extract
from modules.ClusterMode.hpc_resource_radar import extract_and_draw

# factor = {
#     "SPONGE": 2110.975316,
#     "GROMACS": 2056.074767,
#     "AMBER": 3157.360406,
#     "DSDP": 46358.43357,
# }


# energy_factor = {
#     "SPONGE": 14836737,
#     "GROMACS": 9638915.614,
#     "AMBER": 14423737.73,
#     "DSDP": 194411492,
# }

factor = {
    "SPONGE": 4318,
    "GROMACS": 748,
    "AMBER": 622,
    "DSDP": 95540,
}

energy_factor = {
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


def make_score_bar(softwares, scores):
    bar_path = "image/ClusterMode/score_bar.png"
    y_pos = np.arange(len(softwares))
    plt.figure(figsize=(10, 5), dpi=1000)
    plt.barh(y_pos, scores, height=0.4, color="skyblue", align="center")
    plt.title("集群测评软件得分", fontproperties=my_font)
    plt.xlim(0, max(max(scores) / 4 * 5, 110))
    plt.xlabel("Score")
    plt.yticks(y_pos, softwares)
    # plt.axvline(x=100, color="r", linestyle="--", linewidth=1.5)
    for i, v in enumerate(scores):
        plt.text(v + 1, i, str(v), color="black", va="center")
    plt.savefig(bar_path)
    plt.close()
    return bar_path


def cal_multitasks_time_score(software, taskcount, sec):
    score = round((factor[software] * taskcount / sec), 2)
    return score


def cal_multitasks_energy_score(software, taskcount, energy):
    score = round((energy_factor[software] * taskcount / energy), 2)
    return score


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


def make_singletask_linechart(modes, time_result, datatype, outpath):
    ylabel = {
        "time": "Time(s)",
        "cputime": "CPU Time(s)",
        "gputime": "GPU Time(s)",
        "performance": "Performance(ns/day)",
    }
    plt.figure(figsize=(10, 5), dpi=1000)
    plt.plot(modes, time_result, marker="o")
    plt.xlabel("节点数(GPU数)", fontproperties=my_font)
    plt.ylabel(ylabel[datatype])
    plt.savefig(outpath)
    plt.close()
    return True


def extract_gromacs_singletask_result(nodelist, gpus):
    modes = []
    time_result = []
    gputime_result = []
    performance_result = []
    for i in range(1, len(nodelist) + 1):
        total_gpus = i * gpus
        testmode = f"{i}({total_gpus})"
        modes.append(testmode)
        resultdir = f"result/ClusterMode/GROMACS/singletask/{i}_{total_gpus}"
        with open(f"{resultdir}/md.log", "r") as f:
            for line in f:
                if "Performance:" in line:
                    performance_result.append(float(line.split()[1]))
                elif "Time:" in line:
                    time_result.append(float(line.split()[2]))
                    gputime_result.append(float(line.split()[2]) * total_gpus)
                else:
                    continue
    return modes, time_result, gputime_result, performance_result


def extract_amber_singletask_result(nodelist, gpus):
    modes = []
    time_result = []
    gputime_result = []
    performance_result = []
    for i in range(1, len(nodelist) + 1):
        total_gpus = i * gpus
        testmode = f"{i}({total_gpus})"
        modes.append(testmode)
        resultdir = f"result/ClusterMode/AMBER/singletask/{i}_{total_gpus}"
        with open(f"{resultdir}/mdout", "r") as f:
            ready = False
            for line in f:
                if "Average timings for all steps:" in line:
                    ready = True
                elif "ns/day" in line and ready:
                    performance_result.append(float(line.split()[3]))
                elif "Elapsed(s)" in line and ready:
                    time_result.append(float(line.split()[3]))
                    gputime_result.append(float(line.split()[3]) * total_gpus)
                else:
                    continue
    return modes, time_result, gputime_result, performance_result



def make_eff_bar(softwares, effs):
    bar_path = "image/ClusterMode/eff_bar.png"
    y_pos = np.arange(len(softwares))
    plt.figure(figsize=(10, 5), dpi=1000)
    plt.barh(y_pos, scores, height=0.4, color="skyblue", align="center")
    plt.title("集群综合扩展效率", fontproperties=my_font)
    plt.xlim(0, 120)
    plt.xlabel("集群综合扩展效率（%）")
    plt.yticks(y_pos, softwares)
    plt.axvline(x=100, color="r", linestyle="--", linewidth=1.5)
    for i, v in enumerate(scores):
        plt.text(v + 1, i, str(v), color="black", va="center")
    plt.savefig(bar_path)
    plt.close()
    return bar_path



def sponge_report(
    content,
    multitasks_count,
    gpus,
    cpus,
    node_exporter_port,
    nvidia_gpu_exporter_port,
    server_ip,
    server_port,
    partition,
    nodelist,
    sponge_singlenode_score,
):
    software = "SPONGE"
    version = get_version(software)
    cpu_info = get_cpu_info(partition, nodelist[0])
    memory_info = get_memory_info(partition, nodelist[0])
    gpu_info = get_gpu_info(partition, nodelist[0])
    os_version = get_os_version(partition, nodelist[0])
    kernel_version = get_kernel_version(partition, nodelist[0])
    slurm_version = get_slurm_version()
    filesystem_info = get_filesystem_info()
    image_path = f"image/ClusterMode/{software}/"
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    multitasks_best_mode = 1
    multitasks_mondir = f"mon/ClusterMode/{software}/multitasks/"
    multitasks_start_timestamp, multitasks_end_timestamp = get_multitasks_timestamp(
        f"log/ClusterMode/{software}/multitasks/"
    )
    step = 11000
    time_bins = auto_extract(
        node_exporter_port,
        nvidia_gpu_exporter_port,
        server_ip,
        server_port,
        nodelist,
        multitasks_start_timestamp,
        multitasks_end_timestamp,
        multitasks_mondir,
        step,
    )
    multitasks_radar_path = f"image/ClusterMode/{software}/multitasks_radar.png"
    multitasks_radar_gpu_path = f"image/ClusterMode/{software}/multitasks_radar_gpu.png"
    multitasks_radar_split_path = (
        f"image/ClusterMode/{software}/multitasks_radar_split.png"
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
        nodelist,
        cpus,
        gpus,
        multitasks_radar_path,
        multitasks_radar_gpu_path,
        multitasks_radar_split_path,
        [software, multitasks_count, multitasks_best_mode],
    )
    multitasks_time_score = cal_multitasks_time_score(
        software,
        multitasks_count,
        multitasks_end_timestamp - multitasks_start_timestamp,
    )
    multitasks_energy_score = cal_multitasks_energy_score(
        software, multitasks_count, multitasks_power_summary["sum"] / len(nodelist)
    )
    total_score = round(
        np.mean(
            [
                multitasks_time_score,
                multitasks_energy_score,
            ]
        ),
        2,
    )
    eff=round(score/len(nodelist)/singlenode_scores[software],2)
        effs.append(eff)

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"## {software}\n"
    content += f"软件版本：{version}  \n"
    content += f"数据集：{dataset[software]}  \n"
    content += f"集群测试环境：测试集群节点数量{len(nodelist)}，GPU型号{gpu_info['gpu_name']}，单节点GPU卡数{gpu_info['gpu_count']}，单卡显存{gpu_info['memory']}GB，CPU型号{cpu_info['model']}，单节点CPU核心数{cpu_info['cores']}，单节点内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']}，文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。操作系统版本{os_version}，内核版本{kernel_version}，作业调度系统{slurm_version}。  \n"
    content += "&nbsp;\n"

    content += "测评内容：  \n"
    content += "1. 大批量任务并行计算集群资源使用情况分析。记录集群在大批量任务并行模式下的GPU、CPU、内存、I/O读写带宽等资源的使用情况和功耗，输出从任务运行到结束的资源使用情况雷达图，帮助用户了解软件在大批量任务并行模式下的集群资源需求特征、分析被测集群配置是否存在瓶颈。\n"
    content += "2. 集群大批量任务并行计算打分。  \n"
    content += "$$\n"
    content += "\\begin{align*}\n"
    content += "计算用时得分&=系数×\\frac{任务数量}{计算用时}\\\\\n"
    content += "计算能耗得分&=系数×\\frac{任务数量}{平均每个节点的能耗}\\\\\n"
    content += "综合得分&=avg(计算用时得分+计算能耗得分)  \n"
    content += "\\end{align*}\n"
    content += "$$\n"
    content += "系数以目前测试的主流配置为基准（100分）来确定，得分是一个相对值且没有上限。  \n\n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 大批量任务并行计算集群资源使用情况分析：\n"
    content += f"![multitasks_radar]({multitasks_radar_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间集群各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括CPU使用率、 内存使用量、I/O读带宽、I/O写带宽、功耗)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar_gpu]({multitasks_radar_gpu_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间集群各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括GPU使用率和全部显存使用量)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar_split]({multitasks_radar_split_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间集群各指标使用情况，从内向外不同层级分别展示一个指标的变化情况，每个层级内侧线为0，外侧线为该指标全程的最大值，指标范围见图例。  \n\n"
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
    content += f"计算用时得分：{multitasks_time_score}\n"
    content += f"计算能耗得分：{multitasks_energy_score}\n"
    content += f"#### 集群大规模并行计算综合得分：{total_score}\n"
    content += "&nbsp;\n"
    return content, total_score


def gromacs_report(
    content,
    multitasks_count,
    gpus,
    cpus,
    node_exporter_port,
    nvidia_gpu_exporter_port,
    server_ip,
    server_port,
    partition,
    nodelist,
):
    software = "GROMACS"
    version = get_version(software)
    cpu_info = get_cpu_info(partition, nodelist[0])
    memory_info = get_memory_info(partition, nodelist[0])
    gpu_info = get_gpu_info(partition, nodelist[0])
    filesystem_info = get_filesystem_info()
    os_version = get_os_version(partition, nodelist[0])
    kernel_version = get_kernel_version(partition, nodelist[0])
    slurm_version = get_slurm_version()
    image_path = f"image/ClusterMode/{software}/"
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    singletask_node_count = 2
    singletask_gpu_count = singletask_node_count * gpus
    singletask_best_mode = f"{singletask_node_count}_{singletask_gpu_count}"
    with open(
        f"log/ClusterMode/{software}/singletask/{singletask_best_mode}.start.timestamp",
        "r",
    ) as f:
        singletask_start_timestamp = int(f.read().strip())
    with open(
        f"log/ClusterMode/{software}/singletask/{singletask_best_mode}.end.timestamp",
        "r",
    ) as f:
        singletask_end_timestamp = int(f.read().strip())
    singletask_mondir = f"mon/ClusterMode/{software}/singletask/"
    step = 11000
    time_bins = auto_extract(
        node_exporter_port,
        nvidia_gpu_exporter_port,
        server_ip,
        server_port,
        nodelist[:singletask_node_count],
        singletask_start_timestamp,
        singletask_end_timestamp,
        singletask_mondir,
        step,
    )
    singletask_radar_path = f"image/ClusterMode/{software}/singletask_radar.png"
    singletask_radar_gpu_path = f"image/ClusterMode/{software}/singletask_radar_gpu.png"
    singletask_radar_split_path = (
        f"image/ClusterMode/{software}/singletask_radar_split.png"
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
        nodelist[:singletask_node_count],
        cpus,
        gpus,
        singletask_radar_path,
        singletask_radar_gpu_path,
        singletask_radar_split_path,
        [software, "singletask", singletask_gpu_count],
    )
    (
        singletask_modes,
        singletask_time_result,
        singletask_gputime_result,
        singletask_performance_result,
    ) = extract_gromacs_singletask_result(nodelist, gpus)
    singletask_time_linechart_path = (
        f"image/ClusterMode/{software}/singletask_time_linechart.png"
    )
    singletask_gputime_linechart_path = (
        f"image/ClusterMode/{software}/singletask_gputime_linechart.png"
    )
    singletask_performance_linechart_path = (
        f"image/ClusterMode/{software}/singletask_performance_linechart.png"
    )
    make_singletask_linechart(
        singletask_modes, singletask_time_result, "time", singletask_time_linechart_path
    )
    make_singletask_linechart(
        singletask_modes,
        singletask_gputime_result,
        "gputime",
        singletask_gputime_linechart_path,
    )
    make_singletask_linechart(
        singletask_modes,
        singletask_performance_result,
        "performance",
        singletask_performance_linechart_path,
    )
    singletask_best_time_node = int(
        singletask_modes[
            singletask_time_result.index(min(singletask_time_result))
        ].split("(")[0]
    )
    singletask_best_gputime_node = int(
        singletask_modes[
            singletask_gputime_result.index(min(singletask_gputime_result))
        ].split("(")[0]
    )
    singletask_best_performance_node = int(
        singletask_modes[
            singletask_performance_result.index(max(singletask_performance_result))
        ].split("(")[0]
    )

    multitasks_best_mode = 1
    multitasks_mondir = f"mon/ClusterMode/{software}/multitasks/"
    multitasks_start_timestamp, multitasks_end_timestamp = get_multitasks_timestamp(
        f"log/ClusterMode/{software}/multitasks/"
    )
    step = 11000
    time_bins = auto_extract(
        node_exporter_port,
        nvidia_gpu_exporter_port,
        server_ip,
        server_port,
        nodelist,
        multitasks_start_timestamp,
        multitasks_end_timestamp,
        multitasks_mondir,
        step,
    )
    multitasks_radar_path = f"image/ClusterMode/{software}/multitasks_radar.png"
    multitasks_radar_gpu_path = f"image/ClusterMode/{software}/multitasks_radar_gpu.png"
    multitasks_radar_split_path = (
        f"image/ClusterMode/{software}/multitasks_radar_split.png"
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
        nodelist,
        cpus,
        gpus,
        multitasks_radar_path,
        multitasks_radar_gpu_path,
        multitasks_radar_split_path,
        [software, multitasks_count, multitasks_best_mode],
    )
    multitasks_time_score = cal_multitasks_time_score(
        software,
        multitasks_count,
        multitasks_end_timestamp - multitasks_start_timestamp,
    )
    multitasks_energy_score = cal_multitasks_energy_score(
        software, multitasks_count, multitasks_power_summary["sum"] / len(nodelist)
    )
    total_score = round(
        np.mean(
            [
                multitasks_time_score,
                multitasks_energy_score,
            ]
        ),
        2,
    )

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"## {software}\n"
    content += f"软件版本：{version}  \n"
    content += f"数据集：{dataset[software]}  \n"
    content += f"集群测试环境：测试集群节点数量{len(nodelist)}，GPU型号{gpu_info['gpu_name']}，单节点GPU卡数{gpu_info['gpu_count']}，单卡显存{gpu_info['memory']}GB，CPU型号{cpu_info['model']}，单节点CPU核心数{cpu_info['cores']}，单节点内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']}，文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。操作系统版本{os_version}，内核版本{kernel_version}，作业调度系统{slurm_version}。  \n"
    content += "&nbsp;\n"

    content += "测评内容：  \n"
    content += "1. 单任务多机多卡并行运行特征分析。记录软件多机多卡并行计算时对GPU、CPU、内存、I/O读写带宽等资源的使用情况和功耗，输出从任务运行到结束的资源使用情况雷达图，帮助用户了解软件资源需求特征、分析被测集群配置是否存在瓶颈。  \n"
    content += "2. 单任务多机多卡并行效率分析。使用不同节点数量运行计算软件单任务多机多卡测试，统计计算用时、GPU卡时和分子动力学模拟性能。通过本指标的评测，可以得知计算用时最少的单任务并行策略以及计算卡时最少的单任务并行策略。  \n"
    content += "3. 大批量任务并行计算集群资源使用情况分析。记录集群在大批量任务并行模式下的GPU、CPU、内存、I/O读写带宽等资源的使用情况和功耗，输出从任务运行到结束的资源使用情况雷达图，帮助用户了解软件在大批量任务并行模式下的集群资源需求特征、分析被测集群配置是否存在瓶颈。\n"
    content += "4. 集群大批量任务并行计算打分。  \n"
    content += "$$\n"
    content += "\\begin{align*}\n"
    content += "计算用时得分&=系数×\\frac{任务数量}{计算用时}\\\\\n"
    content += "计算能耗得分&=系数×\\frac{任务数量}{平均每个节点的能耗}\\\\\n"
    content += "综合得分&=avg(计算用时得分+计算能耗得分)  \n"
    content += "\\end{align*}\n"
    content += "$$\n"
    content += "系数以目前测试的主流配置为基准（100分）来确定，得分是一个相对值且没有上限。  \n\n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 单任务多机多卡运行特征分析：\n"
    content += f"![singletask_radar]({singletask_radar_path})  \n"
    content += f"注：从顶端顺时针一周为软件单任务使用{singletask_node_count}个节点并行运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括CPU使用率、 内存使用量、I/O读带宽、I/O写带宽、功耗)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![singletask_radar_gpu]({singletask_radar_gpu_path})  \n"
    content += f"注：从顶端顺时针一周为软件单任务使用{singletask_node_count}个节点并行运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括GPU使用率和全部显存使用量)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![singletask_radar_split]({singletask_radar_split_path})  \n"
    content += f"注：从顶端顺时针一周为软件单任务使用{singletask_node_count}个节点并行运行期间各指标使用情况，从内向外不同层级分别展示一个指标的变化情况，每个层级内侧线为0，外侧线为该指标全程的最大值，指标范围见图例。  \n\n"
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
    content += "### 单任务多机多卡并行效率分析：\n"
    content += "计算用时：  \n"
    content += f"![singletask_time_linechart]({singletask_time_linechart_path})  \n"
    content += f"注：横坐标为不同的节点数量（括号内为对应的所用的GPU卡数），纵坐标为任务运行时间，用于对比发现{software}多机多卡并行时计算用时最少的运行模式。  \n"
    content += f"**使用{singletask_best_time_node}个节点{singletask_best_time_node * gpus}张GPU卡并行计算时计算用时最少。**  \n"
    content += "&nbsp;\n"
    content += "计算所用GPU卡时：  \n"
    content += (
        f"![singletask_gputime_linechart]({singletask_gputime_linechart_path})  \n"
    )
    content += f"注：横坐标为不同的节点数量（括号内为对应的所用的GPU卡数），纵坐标为计算所用的GPU卡时（GPU卡时=计算所用GPU数量×计算用时），用于对比发现{software}多机多卡并行时计算所用GPU卡时最少（GPU最经济）的运行模式。  \n"
    content += f"**使用{singletask_best_gputime_node}个节点{singletask_best_gputime_node * gpus}张GPU卡并行计算时GPU卡时最少。**  \n"
    content += "&nbsp;\n"
    content += "分子动力学模拟性能：  \n"
    content += f"![singletask_performance_linechart]({singletask_performance_linechart_path})  \n"
    content += f"注：横坐标为不同的节点数量（括号内为对应的所用的GPU卡数），纵坐标为分子动力学模拟性能，用于对比发现{software}多机多卡并行时分子动力学模拟性能最高的运行模式。分子动力学模拟性能为分子动力学模拟软件常用的性能指标，单位为ns/day，表示在当前系统配置和环境以及输入参数条件下每天能够模拟的纳秒数，该值越大代表单位时间内能完成更长时间的分子动力学模拟，代表整体系统性能更好。  \n"
    content += f"**使用{singletask_best_performance_node}个节点{singletask_best_performance_node * gpus}张GPU卡并行计算时分子动力学模拟性能最高。**  \n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 大批量任务并行计算集群资源使用情况分析：\n"
    content += f"![multitasks_radar]({multitasks_radar_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间集群各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括CPU使用率、 内存使用量、I/O读带宽、I/O写带宽、功耗)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar_gpu]({multitasks_radar_gpu_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间集群各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括GPU使用率和全部显存使用量)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar_split]({multitasks_radar_split_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间集群各指标使用情况，从内向外不同层级分别展示一个指标的变化情况，每个层级内侧线为0，外侧线为该指标全程的最大值，指标范围见图例。  \n\n"
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
    content += f"计算用时得分：{multitasks_time_score}\n"
    content += f"计算能耗得分：{multitasks_energy_score}\n"
    content += f"#### 集群大规模并行计算综合得分：{total_score}\n"
    content += "&nbsp;\n"
    return content, total_score


def amber_report(
    content,
    multitasks_count,
    gpus,
    cpus,
    node_exporter_port,
    nvidia_gpu_exporter_port,
    server_ip,
    server_port,
    partition,
    nodelist,
):
    software = "AMBER"
    version = get_version(software)
    cpu_info = get_cpu_info(partition, nodelist[0])
    memory_info = get_memory_info(partition, nodelist[0])
    gpu_info = get_gpu_info(partition, nodelist[0])
    filesystem_info = get_filesystem_info()
    os_version = get_os_version(partition, nodelist[0])
    kernel_version = get_kernel_version(partition, nodelist[0])
    slurm_version = get_slurm_version()
    image_path = f"image/ClusterMode/{software}/"
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    singletask_node_count = 2
    singletask_gpu_count = singletask_node_count * gpus
    singletask_best_mode = f"{singletask_node_count}_{singletask_gpu_count}"
    with open(
        f"log/ClusterMode/{software}/singletask/{singletask_best_mode}.start.timestamp",
        "r",
    ) as f:
        singletask_start_timestamp = int(f.read().strip())
    with open(
        f"log/ClusterMode/{software}/singletask/{singletask_best_mode}.end.timestamp",
        "r",
    ) as f:
        singletask_end_timestamp = int(f.read().strip())
    singletask_mondir = f"mon/ClusterMode/{software}/singletask/"
    step = 11000
    time_bins = auto_extract(
        node_exporter_port,
        nvidia_gpu_exporter_port,
        server_ip,
        server_port,
        nodelist[:singletask_node_count],
        singletask_start_timestamp,
        singletask_end_timestamp,
        singletask_mondir,
        step,
    )
    singletask_radar_path = f"image/ClusterMode/{software}/singletask_radar.png"
    singletask_radar_gpu_path = f"image/ClusterMode/{software}/singletask_radar_gpu.png"
    singletask_radar_split_path = (
        f"image/ClusterMode/{software}/singletask_radar_split.png"
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
        nodelist[:singletask_node_count],
        cpus,
        gpus,
        singletask_radar_path,
        singletask_radar_gpu_path,
        singletask_radar_split_path,
        [software, "singletask", singletask_gpu_count],
    )
    (
        singletask_modes,
        singletask_time_result,
        singletask_gputime_result,
        singletask_performance_result,
    ) = extract_amber_singletask_result(nodelist, gpus)
    singletask_time_linechart_path = (
        f"image/ClusterMode/{software}/singletask_time_linechart.png"
    )
    singletask_gputime_linechart_path = (
        f"image/ClusterMode/{software}/singletask_gputime_linechart.png"
    )
    singletask_performance_linechart_path = (
        f"image/ClusterMode/{software}/singletask_performance_linechart.png"
    )
    make_singletask_linechart(
        singletask_modes, singletask_time_result, "time", singletask_time_linechart_path
    )
    make_singletask_linechart(
        singletask_modes,
        singletask_gputime_result,
        "gputime",
        singletask_gputime_linechart_path,
    )
    make_singletask_linechart(
        singletask_modes,
        singletask_performance_result,
        "performance",
        singletask_performance_linechart_path,
    )
    singletask_best_time_node = int(
        singletask_modes[
            singletask_time_result.index(min(singletask_time_result))
        ].split("(")[0]
    )
    singletask_best_gputime_node = int(
        singletask_modes[
            singletask_gputime_result.index(min(singletask_gputime_result))
        ].split("(")[0]
    )
    singletask_best_performance_node = int(
        singletask_modes[
            singletask_performance_result.index(max(singletask_performance_result))
        ].split("(")[0]
    )

    multitasks_best_mode = 1
    multitasks_mondir = f"mon/ClusterMode/{software}/multitasks/"
    multitasks_start_timestamp, multitasks_end_timestamp = get_multitasks_timestamp(
        f"log/ClusterMode/{software}/multitasks/"
    )
    step = 11000
    time_bins = auto_extract(
        node_exporter_port,
        nvidia_gpu_exporter_port,
        server_ip,
        server_port,
        nodelist,
        multitasks_start_timestamp,
        multitasks_end_timestamp,
        multitasks_mondir,
        step,
    )
    multitasks_radar_path = f"image/ClusterMode/{software}/multitasks_radar.png"
    multitasks_radar_gpu_path = f"image/ClusterMode/{software}/multitasks_radar_gpu.png"
    multitasks_radar_split_path = (
        f"image/ClusterMode/{software}/multitasks_radar_split.png"
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
        nodelist,
        cpus,
        gpus,
        multitasks_radar_path,
        multitasks_radar_gpu_path,
        multitasks_radar_split_path,
        [software, multitasks_count, multitasks_best_mode],
    )
    multitasks_time_score = cal_multitasks_time_score(
        software,
        multitasks_count,
        multitasks_end_timestamp - multitasks_start_timestamp,
    )
    multitasks_energy_score = cal_multitasks_energy_score(
        software, multitasks_count, multitasks_power_summary["sum"] / len(nodelist)
    )
    total_score = round(
        np.mean(
            [
                multitasks_time_score,
                multitasks_energy_score,
            ]
        ),
        2,
    )

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"## {software}\n"
    content += f"软件版本：{version}  \n"
    content += f"数据集：{dataset[software]}  \n"
    content += f"集群测试环境：测试集群节点数量{len(nodelist)}，GPU型号{gpu_info['gpu_name']}，单节点GPU卡数{gpu_info['gpu_count']}，单卡显存{gpu_info['memory']}GB，CPU型号{cpu_info['model']}，单节点CPU核心数{cpu_info['cores']}，单节点内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']}，文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。操作系统版本{os_version}，内核版本{kernel_version}，作业调度系统{slurm_version}。  \n"
    content += "&nbsp;\n"

    content += "测评内容：  \n"
    content += "1. 单任务多机多卡并行运行特征分析。记录软件多机多卡并行计算时对GPU、CPU、内存、I/O读写带宽等资源的使用情况和功耗，输出从任务运行到结束的资源使用情况雷达图，帮助用户了解软件资源需求特征、分析被测集群配置是否存在瓶颈。  \n"
    content += "2. 单任务多机多卡并行效率分析。使用不同节点数量运行计算软件单任务多机多卡测试，统计计算用时、GPU卡时和分子动力学模拟性能。通过本指标的评测，可以得知计算用时最少的单任务并行策略以及计算卡时最少的单任务并行策略。  \n"
    content += "3. 大批量任务并行计算集群资源使用情况分析。记录集群在大批量任务并行模式下的GPU、CPU、内存、I/O读写带宽等资源的使用情况和功耗，输出从任务运行到结束的资源使用情况雷达图，帮助用户了解软件在大批量任务并行模式下的集群资源需求特征、分析被测集群配置是否存在瓶颈。\n"
    content += "4. 集群大批量任务并行计算打分。  \n"
    content += "$$\n"
    content += "\\begin{align*}\n"
    content += "计算用时得分&=系数×\\frac{任务数量}{计算用时}\\\\\n"
    content += "计算能耗得分&=系数×\\frac{任务数量}{平均每个节点的能耗}\\\\\n"
    content += "综合得分&=avg(计算用时得分+计算能耗得分)  \n"
    content += "\\end{align*}\n"
    content += "$$\n"
    content += "系数以目前测试的主流配置为基准（100分）来确定，得分是一个相对值且没有上限。  \n\n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 单任务多机多卡运行特征分析：\n"
    content += f"![singletask_radar]({singletask_radar_path})  \n"
    content += f"注：从顶端顺时针一周为软件单任务使用{singletask_node_count}个节点并行运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括CPU使用率、 内存使用量、I/O读带宽、I/O写带宽、功耗)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![singletask_radar_gpu]({singletask_radar_gpu_path})  \n"
    content += f"注：从顶端顺时针一周为软件单任务使用{singletask_node_count}个节点并行运行期间各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括GPU使用率和全部显存使用量)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![singletask_radar_split]({singletask_radar_split_path})  \n"
    content += f"注：从顶端顺时针一周为软件单任务使用{singletask_node_count}个节点并行运行期间各指标使用情况，从内向外不同层级分别展示一个指标的变化情况，每个层级内侧线为0，外侧线为该指标全程的最大值，指标范围见图例。  \n\n"
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
    content += "### 单任务多机多卡并行效率分析：\n"
    content += "计算用时：  \n"
    content += f"![singletask_time_linechart]({singletask_time_linechart_path})  \n"
    content += f"注：横坐标为不同的节点数量（括号内为对应的所用的GPU卡数），纵坐标为任务运行时间，用于对比发现{software}多机多卡并行时计算用时最少的运行模式。  \n"
    content += f"**使用{singletask_best_time_node}个节点{singletask_best_time_node * gpus}张GPU卡并行计算时计算用时最少。**  \n"
    content += "&nbsp;\n"
    content += "计算所用GPU卡时：  \n"
    content += (
        f"![singletask_gputime_linechart]({singletask_gputime_linechart_path})  \n"
    )
    content += f"注：横坐标为不同的节点数量（括号内为对应的所用的GPU卡数），纵坐标为计算所用的GPU卡时（GPU卡时=计算所用GPU数量×计算用时），用于对比发现{software}多机多卡并行时计算所用GPU卡时最少（GPU最经济）的运行模式。  \n"
    content += f"**使用{singletask_best_gputime_node}个节点{singletask_best_gputime_node * gpus}张GPU卡并行计算时GPU卡时最少。**  \n"
    content += "&nbsp;\n"
    content += "分子动力学模拟性能：  \n"
    content += f"![singletask_performance_linechart]({singletask_performance_linechart_path})  \n"
    content += f"注：横坐标为不同的节点数量（括号内为对应的所用的GPU卡数），纵坐标为分子动力学模拟性能，用于对比发现{software}多机多卡并行时分子动力学模拟性能最高的运行模式。分子动力学模拟性能为分子动力学模拟软件常用的性能指标，单位为ns/day，表示在当前系统配置和环境以及输入参数条件下每天能够模拟的纳秒数，该值越大代表单位时间内能完成更长时间的分子动力学模拟，代表整体系统性能更好。  \n"
    content += f"**使用{singletask_best_performance_node}个节点{singletask_best_performance_node * gpus}张GPU卡并行计算时分子动力学模拟性能最高。**  \n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 大批量任务并行计算集群资源使用情况分析：\n"
    content += f"![multitasks_radar]({multitasks_radar_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间集群各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括CPU使用率、 内存使用量、I/O读带宽、I/O写带宽、功耗)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar_gpu]({multitasks_radar_gpu_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间集群各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括GPU使用率和全部显存使用量)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar_split]({multitasks_radar_split_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间集群各指标使用情况，从内向外不同层级分别展示一个指标的变化情况，每个层级内侧线为0，外侧线为该指标全程的最大值，指标范围见图例。  \n\n"
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
    content += f"计算用时得分：{multitasks_time_score}\n"
    content += f"计算能耗得分：{multitasks_energy_score}\n"
    content += f"#### 集群大规模并行计算综合得分：{total_score}\n"
    content += "&nbsp;\n"
    return content, total_score


def dsdp_report(
    content,
    multitasks_count,
    gpus,
    cpus,
    node_exporter_port,
    nvidia_gpu_exporter_port,
    server_ip,
    server_port,
    partition,
    nodelist,
):
    software = "DSDP"
    version = get_version(software)
    cpu_info = get_cpu_info(partition, nodelist[0])
    memory_info = get_memory_info(partition, nodelist[0])
    gpu_info = get_gpu_info(partition, nodelist[0])
    filesystem_info = get_filesystem_info()
    os_version = get_os_version(partition, nodelist[0])
    kernel_version = get_kernel_version(partition, nodelist[0])
    slurm_version = get_slurm_version()
    image_path = f"image/ClusterMode/{software}/"
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    multitasks_best_mode = 1
    multitasks_mondir = f"mon/ClusterMode/{software}/multitasks/"
    multitasks_start_timestamp, multitasks_end_timestamp = get_multitasks_timestamp(
        f"log/ClusterMode/{software}/multitasks/"
    )
    step = 11000
    time_bins = auto_extract(
        node_exporter_port,
        nvidia_gpu_exporter_port,
        server_ip,
        server_port,
        nodelist,
        multitasks_start_timestamp,
        multitasks_end_timestamp,
        multitasks_mondir,
        step,
    )
    multitasks_radar_path = f"image/ClusterMode/{software}/multitasks_radar.png"
    multitasks_radar_gpu_path = f"image/ClusterMode/{software}/multitasks_radar_gpu.png"
    multitasks_radar_split_path = (
        f"image/ClusterMode/{software}/multitasks_radar_split.png"
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
        nodelist,
        cpus,
        gpus,
        multitasks_radar_path,
        multitasks_radar_gpu_path,
        multitasks_radar_split_path,
        [software, multitasks_count, multitasks_best_mode],
    )
    multitasks_time_score = cal_multitasks_time_score(
        software,
        multitasks_count,
        multitasks_end_timestamp - multitasks_start_timestamp,
    )
    multitasks_energy_score = cal_multitasks_energy_score(
        software, multitasks_count, multitasks_power_summary["sum"] / len(nodelist)
    )
    total_score = round(
        np.mean(
            [
                multitasks_time_score,
                multitasks_energy_score,
            ]
        ),
        2,
    )

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"## {software}\n"
    content += f"软件版本：{version}  \n"
    content += f"数据集：{dataset[software]}  \n"
    content += f"集群测试环境：测试集群节点数量{len(nodelist)}，GPU型号{gpu_info['gpu_name']}，单节点GPU卡数{gpu_info['gpu_count']}，单卡显存{gpu_info['memory']}GB，CPU型号{cpu_info['model']}，单节点CPU核心数{cpu_info['cores']}，单节点内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']}，文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。操作系统版本{os_version}，内核版本{kernel_version}，作业调度系统{slurm_version}。  \n"
    content += "&nbsp;\n"

    content += "测评内容：  \n"
    content += "1. 大批量任务并行计算集群资源使用情况分析。记录集群在大批量任务并行模式下的GPU、CPU、内存、I/O读写带宽等资源的使用情况和功耗，输出从任务运行到结束的资源使用情况雷达图，帮助用户了解软件在大批量任务并行模式下的集群资源需求特征、分析被测集群配置是否存在瓶颈。\n"
    content += "2. 集群大批量任务并行计算打分。  \n"
    content += "$$\n"
    content += "\\begin{align*}\n"
    content += "计算用时得分&=系数×\\frac{任务数量}{计算用时}\\\\\n"
    content += "计算能耗得分&=系数×\\frac{任务数量}{平均每个节点的能耗}\\\\\n"
    content += "综合得分&=avg(计算用时得分+计算能耗得分)  \n"
    content += "\\end{align*}\n"
    content += "$$\n"
    content += "系数以目前测试的主流配置为基准（100分）来确定，得分是一个相对值且没有上限。  \n\n"
    content += "&nbsp;\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += "### 大批量任务并行计算集群资源使用情况分析：\n"
    content += f"![multitasks_radar]({multitasks_radar_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间集群各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括CPU使用率、 内存使用量、I/O读带宽、I/O写带宽、功耗)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar_gpu]({multitasks_radar_gpu_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间集群各指标使用情况，将各个指标缩放到同一尺度并使用不同比例尺进行对比展示。(包括GPU使用率和全部显存使用量)  \n"
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"![multitasks_radar_split]({multitasks_radar_split_path})  \n"
    content += f"注：从顶端顺时针一周为软件{multitasks_count}个任务并行运行期间集群各指标使用情况，从内向外不同层级分别展示一个指标的变化情况，每个层级内侧线为0，外侧线为该指标全程的最大值，指标范围见图例。  \n\n"
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
    content += f"计算用时得分：{multitasks_time_score}\n"
    content += f"计算能耗得分：{multitasks_energy_score}\n"
    content += f"#### 集群大规模并行计算综合得分：{total_score}\n"
    content += "&nbsp;\n"
    return content, total_score


def generate_report(
    softwares,
    multitasks_count,
    gpus,
    cpus,
    node_exporter_port,
    nvidia_gpu_exporter_port,
    serverip,
    server_port,
    partition,
    nodelist,
):
    content = "# 集群测评报告\n"
    content += "针对生命科学计算的高性能集群性能分析及评测框架，使用代表性生物学计算软件集合，通过Prometheus监控软件对运行时CPU、GPU、内存、IO等资源需求特征进行监控，使用计算用时、计算效率、CPU核时、GPU卡时等关键评价指标对集群性能进行测评和打分，帮助指导生命科学计算集群的建设、提升生命科学计算生产效率。  \n"
    content += "在运行集群测评之前，建议先进行单节点深度测评，了解软件运行特征和最高效的运行模式，使用每款软件最高效的运行模式进行集群测评。集群测评模式对每款计算软件在集群大规模并行计算时的运行特征、计算效率进行分析，分析结果用于帮助用户评估相关计算软件在测试集群上运行是否存在瓶颈以及在测试集群配置下的计算效率。  \n\n"
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
            gpus,
            cpus,
            node_exporter_port,
            nvidia_gpu_exporter_port,
            serverip,
            server_port,
            partition,
            nodelist,
        )
        scores.append(round((sponge_score * weights["SPONGE"]), 2))
    if "GROMACS" in softwares:
        content, gromacs_score = gromacs_report(
            content,
            multitasks_count,
            gpus,
            cpus,
            node_exporter_port,
            nvidia_gpu_exporter_port,
            serverip,
            server_port,
            partition,
            nodelist,
        )
        scores.append(round((gromacs_score * weights["GROMACS"]), 2))
    if "AMBER" in softwares:
        content, amber_score = amber_report(
            content,
            multitasks_count,
            gpus,
            cpus,
            node_exporter_port,
            nvidia_gpu_exporter_port,
            serverip,
            server_port,
            partition,
            nodelist,
        )
        scores.append(round((amber_score * weights["AMBER"]), 2))
    if "DSDP" in softwares:
        content, dsdp_score = dsdp_report(
            content,
            multitasks_count,
            gpus,
            cpus,
            node_exporter_port,
            nvidia_gpu_exporter_port,
            serverip,
            server_port,
            partition,
            nodelist,
        )
        scores.append(round((dsdp_score * weights["DSDP"]), 2))

    cpu_info = get_cpu_info(partition, nodelist[0])
    memory_info = get_memory_info(partition, nodelist[0])
    filesystem_info = get_filesystem_info()
    gpu_info = get_gpu_info(partition, nodelist[0])
    os_version = get_os_version(partition, nodelist[0])
    kernel_version = get_kernel_version(partition, nodelist[0])
    slurm_version = get_slurm_version()
    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"### 集群测评总得分:{round(sum(scores), 2)}"
    score_bar_path = make_score_bar(softwares, scores)
    content += f"![score_bar]({score_bar_path})  \n"
    content += "&nbsp;\n"
    content += "#### 集群测试环境：\n"
    content += f"测试集群节点数量{len(nodelist)}，GPU型号{gpu_info['gpu_name']}，单节点GPU卡数{gpu_info['gpu_count']}，单卡显存{gpu_info['memory']}GB，CPU型号{cpu_info['model']}，单节点CPU核心数{cpu_info['cores']}，单节点内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']}，文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。操作系统版本{os_version}，内核版本{kernel_version}，作业调度系统{slurm_version}。  \n"
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
    softwares = ["BWA", "SPAdes", "Bismark", "STAR", "Cellranger", "GATK"]
    scores = [767.7, 674.59, 696.56, 626.02, 1069.29, 786.2]
    make_score_bar(softwares, scores)
