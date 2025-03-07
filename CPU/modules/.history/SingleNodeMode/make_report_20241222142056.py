import os
import re
import statistics
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from modules.SingleNodeMode.cal_multitasks_scale import (
    cal_gatk_multitasks_scale,
    cal_multitasks_scale,
)
from modules.SingleNodeMode.cal_singletask_scale import cal_singletask_scale

factor = {
    "BWA": 8994.596,
    "SPAdes": 18859.686,
    "Bismark": 29298.532,
    "STAR": 5647.574,
    "Cellranger": 23284.072,
    "GATK": 15304.418,
}

energy_factor = {
    "BWA": 5670354,
    "SPAdes": 10948712,
    "Bismark": 18726386,
    "STAR": 3680472,
    "Cellranger": 25010342,
    "GATK": 9717774,
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


def get_cpu_info():
    cpu_info = {}
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpu_data = f.read()
            model_match = re.search(r"model name\s+:\s+(.*)", cpu_data)
            if model_match:
                cpu_info["model"] = model_match.group(1)
        cpu_info["cores"] = subprocess.check_output(["nproc"]).decode("utf-8").strip()
    except Exception as e:
        print(f"获取CPU信息时出错: {e}")
    return cpu_info


def get_memory_info():
    memory_info = {}
    try:
        with open("/proc/meminfo", "r") as f:
            mem_data = f.read()
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


def reduce_vector_dim(vector):
    max_dim = 10000
    if len(vector) > max_dim:
        step = len(vector) // max_dim
        reduced_vector = vector[::step]
        return reduced_vector[:max_dim]
    else:
        return vector


def make_radar(monfile, runmode, best_mode):
    software = monfile.split("/")[2]
    filename = monfile.split("/")[-1]
    txtfile = monfile.replace(".mon", ".txt")
    os.system("qmon -i " + monfile + " > " + txtfile)
    x, cpu, ram, pfs_recv, pfs_send, power = [], [], [], [], [], []
    with open(txtfile, "r") as infile:
        for line in infile.readlines():
            if line[:6] == "Offset":
                time = line.strip("\n").split(" ")[-1]
                x.append(time)
            elif line[:3] == "ram":
                tot = line.split(" ")[2]
                free = line.split(" ")[4]
                used = (int(tot) - int(free)) * 4 * 1024 / 1024 / 1024 / 1024
                ram.append(used)
            elif line[:3] == "pfs":
                recv = int(line.split(" ")[3]) / 1024 / 1024 / 1024
                send = int(line.split(" ")[5]) / 1024 / 1024 / 1024
                pfs_recv.append(recv)
                pfs_send.append(send)
            elif line[:7] == "cpu_tot":
                total = (
                    float(line.split(" ")[2])
                    + float(line.split(" ")[4])
                    + float(line.split(" ")[6])
                )
                cpu.append(total)
            elif line[:5] == "Power":
                p = int(line.split()[1])
                power.append(p)

    resource_summary = {}
    resource_summary["cpu_max"] = round(max(cpu), 2)
    resource_summary["cpu_median"] = round(statistics.median(cpu), 2)
    resource_summary["cpu_mean"] = round(statistics.mean(cpu), 2)
    resource_summary["ram_max"] = round(max(ram), 2)
    resource_summary["ram_median"] = round(statistics.median(ram), 2)
    resource_summary["ram_mean"] = round(statistics.mean(ram), 2)
    resource_summary["pfs_recv_max"] = round(max(pfs_recv), 2)
    resource_summary["pfs_recv_median"] = round(statistics.median(pfs_recv), 2)
    resource_summary["pfs_recv_mean"] = round(statistics.mean(pfs_recv), 2)
    resource_summary["pfs_send_max"] = round(max(pfs_send), 2)
    resource_summary["pfs_send_median"] = round(statistics.median(pfs_send), 2)
    resource_summary["pfs_send_mean"] = round(statistics.mean(pfs_send), 2)
    if len(power) > 0:
        resource_summary["power_max"] = round(max(power), 2)
        resource_summary["power_median"] = round(statistics.median(power), 2)
        resource_summary["power_mean"] = round(statistics.mean(power), 2)
        resource_summary["energy"] = np.sum(power)
    else:
        resource_summary["power_max"] = 0
        resource_summary["power_median"] = 0
        resource_summary["power_mean"] = 0
        resource_summary["energy"] = 0

    x_reduce = reduce_vector_dim(x)
    ram_reduce = reduce_vector_dim(ram)
    cpu_reduce = reduce_vector_dim(cpu)
    pfs_recv_reduce = reduce_vector_dim(pfs_recv)
    pfs_send_reduce = reduce_vector_dim(pfs_send)
    if len(power) > 0:
        power_reduce = reduce_vector_dim(power)

    plt.rcParams["axes.facecolor"] = "white"
    angles = np.linspace(0.5 * np.pi, -1.5 * np.pi, len(x_reduce), endpoint=False)
    ram_reduce = np.concatenate((ram_reduce, [ram_reduce[0]]))
    cpu_reduce = np.concatenate((cpu_reduce, [cpu_reduce[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    pfs_recv_reduce = np.concatenate((pfs_recv_reduce, [pfs_recv_reduce[0]]))
    pfs_send_reduce = np.concatenate((pfs_send_reduce, [pfs_send_reduce[0]]))
    if len(power) > 0:
        power_reduce = np.concatenate((power_reduce, [power_reduce[0]]))

    fig = plt.figure(figsize=(7, 7), dpi=1000)
    ax = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
    if runmode == "singletask":
        if software == "Bismark":
            plt.title(
                f"{software} 单任务{best_mode//4}线程资源使用情况",
                fontproperties=my_font,
                fontsize=10,
                y=1.05,
            )
        else:
            plt.title(
                f"{software} 单任务{best_mode}线程资源使用情况",
                fontproperties=my_font,
                fontsize=10,
                y=1.05,
            )
    elif runmode == "multitasks":
        plt.title(
            f"{software} 多任务最佳并行模式{best_mode}资源使用情况",
            fontproperties=my_font,
            fontsize=10,
            y=1.05,
        )

    ax.plot(
        angles,
        list(map(lambda x: x * 100 / max(pfs_send), pfs_send_reduce)),
        color="deeppink",
        linestyle="-",
        label="I/O write",
        linewidth=1.0,
    )
    ax.plot(
        angles,
        list(map(lambda x: x * 100 / max(pfs_recv), pfs_recv_reduce)),
        "g-",
        label="I/O read",
        linewidth=1.0,
    )
    ax.plot(
        angles,
        list(map(lambda x: x * 100 / max(ram), ram_reduce)),
        "b-",
        label="RAM",
        linewidth=1.0,
    )
    ax.plot(
        angles, cpu_reduce, color="darkred", linestyle="-", label="CPU", linewidth=1.0
    )

    if len(power) > 0:
        ax.plot(
            angles,
            list(map(lambda x: x * 100 / max(power), power_reduce)),
            color="orange",
            linestyle="-",
            label="Power",
            linewidth=1.0,
        )

    if len(power) > 0:
        bin = 360 // 5
    else:
        bin = 360 // 4

    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax1 = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
    ax1.patch.set_visible(False)
    ax1.grid("off")
    ax1.xaxis.set_visible(False)
    ax1.set_rlabel_position(bin * 2)
    ax1.set_yticks([20, 40, 60, 80, 100])
    ax1.set_yticklabels(
        [
            str(round(max(pfs_recv) / 5, 2)) + "GB",
            str(round(max(pfs_recv) * 2 / 5, 2)) + "GB",
            str(round(max(pfs_recv) * 3 / 5, 2)) + "GB",
            str(round(max(pfs_recv) * 4 / 5, 2)) + "GB",
            str(round(max(pfs_recv), 2)) + "GB",
        ]
    )
    ax1.tick_params(axis="y", colors="g", labelsize=7)
    ax1.set_xticklabels([])

    ax2 = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
    ax2.patch.set_visible(False)
    ax2.grid("off")
    ax2.xaxis.set_visible(False)
    ax2.set_rlabel_position(bin * 3)
    ax2.set_yticks([20, 40, 60, 80, 100])
    ax2.set_yticklabels(
        [
            str(round(max(ram) / 5, 2)) + "GB",
            str(round(max(ram) * 2 / 5, 2)) + "GB",
            str(round(max(ram) * 3 / 5, 2)) + "GB",
            str(round(max(ram) * 4 / 5, 2)) + "GB",
            str(round(max(ram), 2)) + "GB",
        ]
    )
    ax2.tick_params(axis="y", colors="b", labelsize=7)
    ax2.set_xticklabels([])

    ax3 = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
    ax3.patch.set_visible(False)
    ax3.grid("off")
    ax3.xaxis.set_visible(False)
    ax3.set_rlabel_position(bin * 1)
    ax3.set_yticks([20, 40, 60, 80, 100])
    ax3.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
    ax3.tick_params(axis="y", colors="darkred", labelsize=7)
    ax3.set_xticklabels([])

    ax4 = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
    ax4.patch.set_visible(False)
    ax4.grid("off")
    ax4.xaxis.set_visible(False)
    ax4.set_rlabel_position(bin * 0)
    ax4.set_yticks([20, 40, 60, 80, 100])
    ax4.set_yticklabels(
        [
            str(round(max(pfs_send) / 5, 2)) + "GB",
            str(round(max(pfs_send) * 2 / 5, 2)) + "GB",
            str(round(max(pfs_send) * 3 / 5, 2)) + "GB",
            str(round(max(pfs_send) * 4 / 5, 2)) + "GB",
            str(round(max(pfs_send), 2)) + "GB",
        ]
    )
    ax4.tick_params(axis="y", colors="deeppink", labelsize=7)
    ax4.set_xticklabels([])

    if len(power) > 0:
        ax5 = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
        ax5.patch.set_visible(False)
        ax5.grid("off")
        ax5.xaxis.set_visible(False)
        ax5.set_rlabel_position(bin * 4)
        ax5.set_yticks([20, 40, 60, 80, 100])
        ax5.set_yticklabels(
            [
                str(round(max(power) / 5, 2)) + "W",
                str(round(max(power) * 2 / 5, 2)) + "W",
                str(round(max(power) * 3 / 5, 2)) + "W",
                str(round(max(power) * 4 / 5, 2)) + "W",
                str(round(max(power), 2)) + "W",
            ]
        )
        ax5.tick_params(axis="y", colors="orange", labelsize=7)
        ax5.set_xticklabels([])

    matplotlib.rcParams["agg.path.chunksize"] = 180000
    matplotlib.rcParams["path.simplify_threshold"] = 1
    # ax.legend(bbox_to_anchor=(1.1, 1.1))
    ax.legend(bbox_to_anchor=(1.1, 1.1), fontsize=8)
    path = "image/SingleNodeMode/" + software
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = path + "/" + filename.replace(".mon", ".png")
    fig.savefig(image_path)
    return image_path, resource_summary


def make_radar_split(monfile, runmode, best_mode):
    software = monfile.split("/")[2]
    filename = monfile.split("/")[-1]
    txtfile = monfile.replace(".mon", ".txt")
    x, cpu, ram, pfs_recv, pfs_send, power = [], [], [], [], [], []
    with open(txtfile, "r") as infile:
        for line in infile.readlines():
            if line[:6] == "Offset":
                time = line.strip("\n").split(" ")[-1]
                x.append(time)
            elif line[:3] == "ram":
                tot = line.split(" ")[2]
                free = line.split(" ")[4]
                used = (int(tot) - int(free)) * 4 * 1024 / 1024 / 1024 / 1024
                ram.append(used)
            elif line[:3] == "pfs":
                recv = int(line.split(" ")[3]) / 1024 / 1024 / 1024
                send = int(line.split(" ")[5]) / 1024 / 1024 / 1024
                pfs_recv.append(recv)
                pfs_send.append(send)
            elif line[:7] == "cpu_tot":
                total = (
                    float(line.split(" ")[2])
                    + float(line.split(" ")[4])
                    + float(line.split(" ")[6])
                )
                cpu.append(total)
            elif line[:5] == "Power":
                p = int(line.split()[1])
                power.append(p)

    x_reduce = reduce_vector_dim(x)
    ram_reduce = reduce_vector_dim(ram)
    cpu_reduce = reduce_vector_dim(cpu)
    pfs_recv_reduce = reduce_vector_dim(pfs_recv)
    pfs_send_reduce = reduce_vector_dim(pfs_send)
    if len(power) > 0:
        power_reduce = reduce_vector_dim(power)

    plt.rcParams["axes.facecolor"] = "white"
    angles = np.linspace(0.5 * np.pi, -1.5 * np.pi, len(x_reduce), endpoint=False)
    ram_reduce = np.concatenate((ram_reduce, [ram_reduce[0]]))
    cpu_reduce = np.concatenate((cpu_reduce, [cpu_reduce[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    pfs_recv_reduce = np.concatenate((pfs_recv_reduce, [pfs_recv_reduce[0]]))
    pfs_send_reduce = np.concatenate((pfs_send_reduce, [pfs_send_reduce[0]]))
    if len(power) > 0:
        power_reduce = np.concatenate((power_reduce, [power_reduce[0]]))

    fig = plt.figure(figsize=(7, 7), dpi=1000)
    ax = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")

    if len(power) > 0:
        bin = 360 // 5
    else:
        bin = 360 // 4

    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines["polar"].set_color("lightgray")
    ax.spines["polar"].set_linewidth(0.5)
    ax.grid(color="lightgrey", linewidth=0.3)

    ax1 = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
    ax1.set_ylim(0, 100)
    ax1.patch.set_visible(False)
    ax1.grid("off")
    ax1.xaxis.set_visible(False)
    ax1.set_rlabel_position(90 - bin * 0)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.yaxis.set_visible(False)
    ax1.tick_params(axis="y", colors="g", labelsize=7)
    ax1.yaxis.grid(color="lightgrey", linewidth=0.3)
    ax1.set_xticklabels([])
    ax1.spines["polar"].set_visible(False)

    ax2 = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
    ax2.set_ylim(0, 100)
    ax2.patch.set_visible(False)
    ax2.grid("off")
    ax2.xaxis.set_visible(False)
    ax2.set_rlabel_position(90 - bin * 2)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.yaxis.set_visible(False)
    ax2.tick_params(axis="y", colors="b", labelsize=7)
    ax2.yaxis.grid(color="lightgrey", linewidth=0.3)
    ax2.set_xticklabels([])
    ax2.spines["polar"].set_visible(False)

    ax3 = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
    ax3.set_ylim(0, 100)
    ax3.patch.set_visible(False)
    ax3.grid("off")
    ax3.xaxis.set_visible(False)
    ax3.set_rlabel_position(90 - bin * 3)
    ax3.set_yticks([0, 20, 40, 60, 80, 100])
    ax3.yaxis.set_visible(False)
    ax3.tick_params(axis="y", colors="firebrick", labelsize=7)
    ax3.yaxis.grid(color="lightgrey", linewidth=0.3)
    ax3.set_xticklabels([])
    ax3.spines["polar"].set_visible(False)

    ax4 = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
    ax4.set_ylim(0, 100)
    ax4.patch.set_visible(False)
    ax4.grid("off")
    ax4.xaxis.set_visible(False)
    ax4.set_rlabel_position(90 - bin * 1)
    ax4.set_yticks([0, 20, 40, 60, 80, 100])
    ax4.yaxis.set_visible(False)
    ax4.tick_params(axis="y", colors="deeppink", labelsize=7)
    ax4.yaxis.grid(color="lightgrey", linewidth=0.3)
    ax4.set_xticklabels([])
    ax4.spines["polar"].set_visible(False)

    if len(power) > 0:
        ax5 = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
        ax5.set_ylim(0, 100)
        ax5.patch.set_visible(False)
        ax5.grid("off")
        ax5.xaxis.set_visible(False)
        ax5.set_rlabel_position(90 - bin * 4)
        ax5.set_yticks([0, 20, 40, 60, 80, 100])
        ax5.yaxis.set_visible(False)
        ax5.tick_params(axis="y", colors="orange", labelsize=7)
        ax5.yaxis.grid(color="lightgrey", linewidth=0.3)
        ax5.set_xticklabels([])
        ax5.spines["polar"].set_visible(False)

    if len(power) > 0:
        axtop = ax5
    else:
        axtop = ax4

    axtop.plot(
        angles,
        list(map(lambda x: x * 20 / max(pfs_send) + 20, pfs_send_reduce)),
        color="deeppink",
        linestyle="-",
        label=f"I/O write(0-{round(max(pfs_send), 2)}GB)",
        linewidth=1.5,
    )
    axtop.plot(
        angles,
        list(map(lambda x: x * 20 / max(pfs_recv), pfs_recv_reduce)),
        "g-",
        label=f"I/O read(0-{round(max(pfs_recv), 2)}GB)",
        linewidth=1.5,
    )
    axtop.plot(
        angles,
        list(map(lambda x: x * 20 / max(ram) + 40, ram_reduce)),
        "b-",
        label=f"RAM(0-{round(max(ram), 2)}GB)",
        linewidth=1.5,
    )
    axtop.plot(
        angles,
        list(map(lambda x: x / 5 + 60, cpu_reduce)),
        color="firebrick",
        linestyle="-",
        label=f"CPU(0-{round(max(cpu), 2)}%)",
        linewidth=1.5,
    )

    if len(power) > 0:
        axtop.plot(
            angles,
            list(map(lambda x: x * 20 / max(power) + 80, power_reduce)),
            color="orange",
            linestyle="-",
            label=f"Power(0-{round(max(power), 2)}W)",
            linewidth=1.5,
        )
    if runmode == "singletask":
        if software == "Bismark":
            plt.title(
                f"{software} 单任务{best_mode//4}线程资源使用情况（分层级）",
                fontproperties=my_font,
                fontsize=10,
                y=1.05,
            )
        else:
            plt.title(
                f"{software} 单任务{best_mode}线程资源使用情况（分层级）",
                fontproperties=my_font,
                fontsize=10,
                y=1.05,
            )
    elif runmode == "multitasks":
        plt.title(
            f"{software} 多任务最佳并行模式{best_mode}资源使用情况（分层级）",
            fontproperties=my_font,
            fontsize=10,
            y=1.05,
        )

    matplotlib.rcParams["agg.path.chunksize"] = 180000
    matplotlib.rcParams["path.simplify_threshold"] = 1
    fig.legend(fontsize=8)
    path = "image/SingleNodeMode/" + software
    image_path = path + "/" + filename.replace(".mon", "_split.png")
    fig.savefig(image_path)
    return image_path


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
    logs = sorted(os.listdir(logdir))
    path = "image/SingleNodeMode/" + software
    if not os.path.exists(path):
        os.makedirs(path)
    time_dict = {}
    time, cputime = [], []
    for f in logs:
        if f.count("_") == 1:
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
        table += f"| {threads[i]} | {round(time[i],2)} | {round(time[0]/time[i],2)} | {round(time[0]/time[i]/threads[i]*100,2)}% | {round(cputime[i],2)} |\n"
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
    for root, _, files in os.walk(directory):
        for file in files:
            if len(file.split("_")) == 4:
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    if software == "GATK":
                        if "done" in f.read():
                            file_without_err.append(
                                file.split("_")[1] + "_" + file.split("_")[2]
                            )
                    elif software == "STAR":
                        file_content = f.read()
                        if (
                            error_keyword[software] not in file_content
                            and "ERROR" not in file_content
                        ):
                            file_without_err.append(
                                file.split("_")[1] + "_" + file.split("_")[2]
                            )
                    else:
                        if error_keyword[software] not in f.read():
                            file_without_err.append(
                                file.split("_")[1] + "_" + file.split("_")[2]
                            )
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


def multitasks_best_parallelmode(software, taskcount, cpus, mem):
    logdir = "log/SingleNodeMode/" + software + "/multitasks/"
    outputdir = "result/SingleNodeMode/" + software + "/multitasks/output/"
    mode_without_err, mode_with_err = find_mode_without_err(
        outputdir, software, taskcount, cpus, mem
    )
    logs = sorted(os.listdir(logdir))
    parallel, time = [], []
    for f in logs:
        if f.count("_") == 2:
            sec = cal_sec(logdir + f)
            parallelmode = f.split(".")[0].replace("test_", "")
            if parallelmode in mode_without_err:
                parallel.append(parallelmode)
                time.append(sec)
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
    logs = sorted(os.listdir(logdir))
    path = "image/SingleNodeMode/" + software
    if not os.path.exists(path):
        os.makedirs(path)
    time = []
    for f in logs:
        if f.count("_") == 2:
            sec = cal_sec(logdir + f)
            time.append(sec)
            thread = f.split("_")[1]
    time.append(cal_sec(f"{logdir}test_{thread}.log"))
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


def generate_report(multitasks_count, repeats, cpus, mem):
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

    cpu_info = get_cpu_info()
    memory_info = get_memory_info()
    filesystem_info = get_filesystem_info()

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
        singletask_monfile = f"mon/SingleNodeMode/{software}/singletask.mon"
        if software == "STAR":
            star_error_threads = []
            thread = cpus
            threads = sorted(cal_singletask_scale(cpus, software), reverse=True)
            for i in threads:
                log_file = f"log/SingleNodeMode/{software}/singletask/test_{i}.log"
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        file_content = f.read()
                        if "ERROR" not in file_content:
                            thread = i
                            break
                        else:
                            star_error_threads.append(i)
                            continue
            # print(f"STAR error threads: {star_error_threads}")
            singletask_radar_path, singletask_resource_summary = make_radar(
                singletask_monfile, "singletask", thread
            )
            singletask_radar_split_path = make_radar_split(
                singletask_monfile, "singletask", thread
            )
        else:
            singletask_radar_path, singletask_resource_summary = make_radar(
                singletask_monfile, "singletask", cpus
            )
            singletask_radar_split_path = make_radar_split(
                singletask_monfile, "singletask", cpus
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
        multitasks_monfile = (
            f"mon/SingleNodeMode/{software}/multitasks/test_{best_mode}.mon"
        )

        multitasks_radar_path, multitasks_resource_summary = make_radar(
            multitasks_monfile, "multitasks", best_mode
        )

        multitasks_radar_split_path = make_radar_split(
            multitasks_monfile, "multitasks", best_mode
        )

        singletask_violin_path, singletask_stability_score = make_singletask_violin(
            software, repeats
        )

        stability_score = round(singletask_stability_score, 2)
        if multitasks_resource_summary["energy"] != 0:
            energy_score = cal_energy_score(
                software, multitasks_count, multitasks_resource_summary["energy"]
            )
            score = round(
                (multitasks_score + energy_score) / 2 * stability_score / 100, 2
            )
        else:
            score = round(multitasks_score * stability_score / 100, 2)
        scores.append(round(weights[software] * score, 2))

        content += '<div STYLE="page-break-after: always;"></div>\n\n'
        content += f"## {software}\n"
        content += f"软件版本：{version}  \n"
        content += f"数据集：{dataset[software]}  \n"
        content += f"测试硬件配置：CPU型号{cpu_info['model']}，CPU核心数{cpu_info['cores']}，内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']}，文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。  \n\n"
        content += "### 运行特征分析：\n"
        content += f"![singletask_radar]({singletask_radar_path})  \n"
        content += '<div STYLE="page-break-after: always;"></div>\n\n'
        content += f"![singletask_radar_split]({singletask_radar_split_path})  \n"
        content += f"CPU使用率峰值：{singletask_resource_summary['cpu_max']}%，中位数{singletask_resource_summary['cpu_median']}%，平均值{singletask_resource_summary['cpu_mean']}%  \n"
        content += f"内存使用峰值：{singletask_resource_summary['ram_max']}GB，中位数{singletask_resource_summary['ram_median']}GB，平均值{singletask_resource_summary['ram_mean']}GB  \n"
        content += f"I/O读峰值速率：{singletask_resource_summary['pfs_recv_max']}GB，中位数{singletask_resource_summary['pfs_recv_median']}GB，平均值{singletask_resource_summary['pfs_recv_mean']}GB  \n"
        content += f"I/O写峰值速率：{singletask_resource_summary['pfs_send_max']}GB，中位数{singletask_resource_summary['pfs_send_median']}GB，平均值{singletask_resource_summary['pfs_send_mean']}GB  \n"
        if singletask_resource_summary["power_max"] != 0:
            content += f"功耗峰值：{singletask_resource_summary['power_max']}W，中位数{singletask_resource_summary['power_median']}W，平均值{singletask_resource_summary['power_mean']}W  \n"
        content += "&nbsp;\n"
        content += '<div STYLE="page-break-after: always;"></div>\n\n'
        content += "### 单任务不同线程加速比分析：\n"
        content += f"![singletask_time_linechart]({singletask_time_linechart_path})  \n"
        content += (
            f"![singletask_cputime_linechart]({singletask_cputime_linechart_path})  \n"
        )
        content += "&nbsp;\n"
        if software == "STAR":
            for i in sorted(star_error_threads):
                content += f"{i}线程任务失败，请检查任务输出日志log/SingleNodeMode/STAR/test_{i}.log。  \n"
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
        content += f"CPU使用率峰值：{multitasks_resource_summary['cpu_max']}%，中位数{multitasks_resource_summary['cpu_median']}%，平均值{multitasks_resource_summary['cpu_mean']}%  \n"
        content += f"内存使用峰值：{multitasks_resource_summary['ram_max']}GB，中位数{multitasks_resource_summary['ram_median']}GB，平均值{multitasks_resource_summary['ram_mean']}GB  \n"
        content += f"I/O读峰值速率：{multitasks_resource_summary['pfs_recv_max']}GB，中位数{multitasks_resource_summary['pfs_recv_median']}GB，平均值{multitasks_resource_summary['pfs_recv_mean']}GB  \n"
        content += f"I/O写峰值速率：{multitasks_resource_summary['pfs_send_max']}GB，中位数{multitasks_resource_summary['pfs_send_median']}GB，平均值{multitasks_resource_summary['pfs_send_mean']}GB  \n"
        if multitasks_resource_summary["power_max"] != 0:
            content += f"功耗峰值：{multitasks_resource_summary['power_max']}W，中位数{multitasks_resource_summary['power_median']}W，平均值{multitasks_resource_summary['power_mean']}W  \n"
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
    content += f"\n\n### 单节点测评总得分:{round(sum(scores),2)}\n\n"
    score_bar_path = make_score_bar(softwares, scores)
    content += f"![score_bar]({score_bar_path})  \n"
    content += "&nbsp;\n"
    content += "#### 测试硬件配置：\n"
    content += f"CPU型号{cpu_info['model']}，CPU核心数{cpu_info['cores']}，内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']},文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。  \n"
    content += "&nbsp;\n"
    for software in softwares:
        content += f"#### {software}\n"
        content += f"软件版本：{version}  \n"
        content += f"数据集：{dataset[software]}  \n"
        if software == "STAR":
            content += f"单任务测试{singletask_threads_count[software]+len(star_error_threads)}种不同线程数，失败{len(star_error_threads)}次。最佳线程数为{singletask_best_threads[software]}，用时{round(singletask_best_time[software],2)}s，CPU核时{round(singletask_best_cputime[software],2)}s。  \n"
        else:
            content += f"单任务测试{singletask_threads_count[software]}种不同线程数。最佳线程数为{singletask_best_threads[software]}，用时{round(singletask_best_time[software],2)}s，CPU核时{round(singletask_best_cputime[software],2)}s。  \n"
        content += f"多任务并行测试{multitasks_mode_without_error_count[software]+multitasks_mode_with_error_count[software]}种不同规模，失败{multitasks_mode_with_error_count[software]}次。最佳并行模式为{multitasks_best_mode[software]}，用时{round(multitasks_best_time[software],2)}s，CPU核时{round(multitasks_best_cputime[software],2)}s。  \n"
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
