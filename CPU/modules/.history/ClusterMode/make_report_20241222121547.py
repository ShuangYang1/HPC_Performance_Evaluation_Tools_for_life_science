import os
import re
import statistics
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

factor = {
    "BWA": 1364.49218,
    "SPAdes": 3598.489983,
    "Bismark": 4372.523655,
    "STAR": 1751.022851,
    "Cellranger": 4214.99828,
    "GATK": 4349.452355,
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


def reduce_vector_dim(vector, max_dim=10000):
    if len(vector) > max_dim:
        step = len(vector) // max_dim
        reduced_vector = vector[::step]
        return reduced_vector[:max_dim]
    else:
        return vector


def make_radar(monfile, nodelist):
    software = monfile.split("/")[2]
    filename = monfile.split("/")[-1]
    all_x = [[] for _ in range(len(nodelist))]
    all_cpu = [[] for _ in range(len(nodelist))]
    all_ram = [[] for _ in range(len(nodelist))]
    all_pfs_recv = [[] for _ in range(len(nodelist))]
    all_pfs_send = [[] for _ in range(len(nodelist))]
    all_power = [[] for _ in range(len(nodelist))]
    for i in range(len(nodelist)):
        node = nodelist[i]
        txtfile = monfile.replace("cluster.mon", f"{node}.txt")
        os.system(f"qmon -i {monfile} -h {node} > {txtfile}")
        with open(txtfile, "r") as infile:
            for line in infile.readlines():
                if line[:6] == "Offset":
                    time = line.strip("\n").split(" ")[-1]
                    all_x[i].append(time)
                elif line[:3] == "ram":
                    tot = line.split(" ")[2]
                    free = line.split(" ")[4]
                    used = (int(tot) - int(free)) * 4 * 1024 / 1024 / 1024 / 1024
                    all_ram[i].append(used)
                elif line[:3] == "pfs":
                    recv = int(line.split(" ")[3]) / 1024 / 1024 / 1024
                    send = int(line.split(" ")[5]) / 1024 / 1024 / 1024
                    all_pfs_recv[i].append(recv)
                    all_pfs_send[i].append(send)
                elif line[:7] == "cpu_tot":
                    total = (
                        float(line.split(" ")[2])
                        + float(line.split(" ")[4])
                        + float(line.split(" ")[6])
                    )
                    all_cpu[i].append(total)
                elif line[:5] == "Power":
                    p = int(line.split()[1])
                    all_power[i].append(p)

    min_len = min(len(row) for row in all_x)
    all_cpu_array = np.array([row[:min_len] for row in all_cpu])
    all_ram_array = np.array([row[:min_len] for row in all_ram])
    all_pfs_recv_array = np.array([row[:min_len] for row in all_pfs_recv])
    all_pfs_send_array = np.array([row[:min_len] for row in all_pfs_send])
    if len(all_power[0]) > 0:
        all_power_array = np.array([row[:min_len] for row in all_power])
    else:
        all_power_array = np.array(all_power)

    cpu = np.mean(all_cpu_array, axis=0)
    ram = np.sum(all_ram_array, axis=0)
    pfs_recv = np.sum(all_pfs_recv_array, axis=0)
    pfs_send = np.sum(all_pfs_send_array, axis=0)
    if len(all_power_array[0]) > 0:
        power = np.sum(all_power_array, axis=0)
    else:
        power = np.array([])

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

    ram_reduce = reduce_vector_dim(ram, 10000)
    cpu_reduce = reduce_vector_dim(cpu, 10000)
    pfs_recv_reduce = reduce_vector_dim(pfs_recv, 10000)
    pfs_send_reduce = reduce_vector_dim(pfs_send, 10000)
    if len(power) > 0:
        power_reduce = reduce_vector_dim(power, 10000)

    plt.rcParams["axes.facecolor"] = "whitesmoke"
    angles = np.linspace(0.5 * np.pi, -1.5 * np.pi, len(ram_reduce), endpoint=False)
    ram_reduce = np.concatenate((ram_reduce, [ram_reduce[0]]))
    cpu_reduce = np.concatenate((cpu_reduce, [cpu_reduce[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    pfs_recv_reduce = np.concatenate((pfs_recv_reduce, [pfs_recv_reduce[0]]))
    pfs_send_reduce = np.concatenate((pfs_send_reduce, [pfs_send_reduce[0]]))
    if len(power) > 0:
        power_reduce = np.concatenate((power_reduce, [power_reduce[0]]))

    fig = plt.figure(figsize=(7, 7), dpi=1000)
    ax = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
    ax.plot(
        angles, cpu_reduce, color="darkred", linestyle="-", label="CPU", linewidth=1.0
    )
    ax.plot(
        angles,
        list(map(lambda x: x * 100 / max(ram), ram_reduce)),
        "b-",
        label="RAM",
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
        list(map(lambda x: x * 100 / max(pfs_send), pfs_send_reduce)),
        color="deeppink",
        linestyle="-",
        label="I/O write",
        linewidth=1.0,
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
    ax.legend(bbox_to_anchor=(1.1, 1.1))
    path = f"image/ClusterMode/{software}"
    if not os.path.exists(path):
        os.makedirs(path)
    pngname = filename.replace(".mon", "_radar.png")
    image_path = f"{path}/{pngname}"
    fig.savefig(image_path)
    return image_path, resource_summary


def make_radar_split(monfile, nodelist):
    software = monfile.split("/")[2]
    filename = monfile.split("/")[-1]
    all_x = [[] for _ in range(len(nodelist))]
    all_cpu = [[] for _ in range(len(nodelist))]
    all_ram = [[] for _ in range(len(nodelist))]
    all_pfs_recv = [[] for _ in range(len(nodelist))]
    all_pfs_send = [[] for _ in range(len(nodelist))]
    all_power = [[] for _ in range(len(nodelist))]
    for i in range(len(nodelist)):
        node = nodelist[i]
        txtfile = monfile.replace("cluster.mon", f"{node}.txt")
        os.system(f"qmon -i {monfile} -h {node} > {txtfile}")
        with open(txtfile, "r") as infile:
            for line in infile.readlines():
                if line[:6] == "Offset":
                    time = line.strip("\n").split(" ")[-1]
                    all_x[i].append(time)
                elif line[:3] == "ram":
                    tot = line.split(" ")[2]
                    free = line.split(" ")[4]
                    used = (int(tot) - int(free)) * 4 * 1024 / 1024 / 1024 / 1024
                    all_ram[i].append(used)
                elif line[:3] == "pfs":
                    recv = int(line.split(" ")[3]) / 1024 / 1024 / 1024
                    send = int(line.split(" ")[5]) / 1024 / 1024 / 1024
                    all_pfs_recv[i].append(recv)
                    all_pfs_send[i].append(send)
                elif line[:7] == "cpu_tot":
                    total = (
                        float(line.split(" ")[2])
                        + float(line.split(" ")[4])
                        + float(line.split(" ")[6])
                    )
                    all_cpu[i].append(total)
                elif line[:5] == "Power":
                    p = int(line.split()[1])
                    all_power[i].append(p)

    min_len = min(len(row) for row in all_x)
    all_cpu_array = np.array([row[:min_len] for row in all_cpu])
    all_ram_array = np.array([row[:min_len] for row in all_ram])
    all_pfs_recv_array = np.array([row[:min_len] for row in all_pfs_recv])
    all_pfs_send_array = np.array([row[:min_len] for row in all_pfs_send])
    if len(all_power[0]) > 0:
        all_power_array = np.array([row[:min_len] for row in all_power])
    else:
        all_power_array = np.array(all_power)

    cpu = np.mean(all_cpu_array, axis=0)
    ram = np.sum(all_ram_array, axis=0)
    pfs_recv = np.sum(all_pfs_recv_array, axis=0)
    pfs_send = np.sum(all_pfs_send_array, axis=0)
    if len(all_power_array[0]) > 0:
        power = np.sum(all_power_array, axis=0)
    else:
        power = np.array([])

    ram_reduce = reduce_vector_dim(ram, 10000)
    cpu_reduce = reduce_vector_dim(cpu, 10000)
    pfs_recv_reduce = reduce_vector_dim(pfs_recv, 10000)
    pfs_send_reduce = reduce_vector_dim(pfs_send, 10000)
    if len(power) > 0:
        power_reduce = reduce_vector_dim(power, 10000)

    plt.rcParams["axes.facecolor"] = "whitesmoke"
    angles = np.linspace(0.5 * np.pi, -1.5 * np.pi, len(ram_reduce), endpoint=False)
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

    matplotlib.rcParams["agg.path.chunksize"] = 180000
    matplotlib.rcParams["path.simplify_threshold"] = 1
    fig.legend(bbox_to_anchor=(1.1, 1.1), fontsize=8)
    path = f"image/ClusterMode/{software}"
    if not os.path.exists(path):
        os.makedirs(path)
    pngname = filename.replace(".mon", "_radar_split.png")
    image_path = f"{path}/{pngname}"
    fig.savefig(image_path)
    return image_path


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
    plt.xlim(0, max(max(scores) + 20, 110))
    plt.xlabel("Score")
    plt.yticks(y_pos, softwares)
    plt.axvline(x=100, color="r", linestyle="--", linewidth=1.5)
    for i, v in enumerate(scores):
        plt.text(v + 1, i, str(v), color="black", va="center")
    plt.savefig(bar_path)
    plt.close()
    return bar_path


def generate_report(multitasks_count, nodelist):
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
    content += "能耗得分&=系数×\\frac{任务数量}{集群总能耗}\\\\\n"
    content += "综合得分&=\\frac{计算用时得分+能耗得分}{2}  \n"
    content += "\\end{align*}\n"
    content += "$$\n"
    content += "若测试平台不支持功耗统计，则计算用时得分即软件最终得分。  \n"
    content += "系数以目前测试的主流配置为基准（100分）来确定，得分是一个相对值且没有上限。  \n\n"
    content += "报告最终会计算总得分，总得分为各软件得分之和。  \n"
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

    for software in softwares:
        version = get_version(software)
        monfile = f"mon/ClusterMode/{software}/cluster.mon"
        radar_path, resource_summary = make_radar(monfile, nodelist)
        radar_split_path = make_radar_split(monfile, nodelist)
        with open(f"log/ClusterMode/{software}/threads/0.log", "r") as f:
            threads = int(f.read().strip())
        with open(f"log/ClusterMode/{software}/time.log", "r") as f:
            sec = int(f.read().strip())
        multitasks_score = cal_score(software, multitasks_count, sec)
        if resource_summary["power_max"] != 0:
            energy_score = cal_energy_score(
                software, multitasks_count, resource_summary["energy"]
            )
            score = round((multitasks_score + energy_score) / 2, 2)
        else:
            score = multitasks_score
        scores.append(round(weights[software] * score, 2))

        content += '<div STYLE="page-break-after: always;"></div>\n\n'
        content += f"## {software}\n"
        content += f"软件版本：{version}  \n"
        content += f"数据集：{dataset[software]}  \n"
        content += f"测试集群硬件配置：测试集群节点数量{len(nodelist)}，CPU型号{cpu_info['model']}，CPU核心数{cpu_info['cores']}，内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']}，文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。  \n"

        content += "### 集群资源使用情况分析：\n"
        content += f"测试软件：{software}  \n"
        content += f"并行模式：{threads}线程  \n"
        content += f"任务量：{multitasks_count}  \n"
        content += f"![radar]({radar_path})  \n"
        content += f"![radar_split]({radar_split_path})  \n"
        content += f"平均CPU使用率峰值：{resource_summary['cpu_max']}%，中位数{resource_summary['cpu_median']}%，平均值{resource_summary['cpu_mean']}%  \n"
        content += f"内存使用峰值：{resource_summary['ram_max']}GB，中位数{resource_summary['ram_median']}GB，平均值{resource_summary['ram_mean']}GB  \n"
        content += f"I/O读峰值速率：{resource_summary['pfs_recv_max']}GB，中位数{resource_summary['pfs_recv_median']}GB，平均值{resource_summary['pfs_recv_mean']}GB  \n"
        content += f"I/O写峰值速率：{resource_summary['pfs_send_max']}GB，中位数{resource_summary['pfs_send_median']}GB，平均值{resource_summary['pfs_send_mean']}GB  \n"
        if resource_summary["power_max"] != 0:
            content += f"功耗峰值：{resource_summary['power_max']}W，中位数{resource_summary['power_median']}W，平均值{resource_summary['power_mean']}W  \n"
            content += "&nbsp;\n"
            content += f"计算用时得分：{multitasks_score}\n"
            content += f"功耗得分：{energy_score}\n"
        content += f"#### 集群大规模并行计算得分：{score}\n"

    content += '<div STYLE="page-break-after: always;"></div>\n\n'
    content += f"### 集群测评总得分:{round(sum(scores),2)}"
    score_bar_path = make_score_bar(softwares, scores)
    content += f"![score_bar]({score_bar_path})  \n"
    content += "&nbsp;\n"
    content += "#### 测试硬件配置：\n"
    content += f"测试集群节点数量{len(nodelist)}，CPU型号{cpu_info['model']}，CPU核心数{cpu_info['cores']}，内存容量{memory_info['total']}GB。文件系统挂载点{filesystem_info['mount_point']}，文件系统类型{filesystem_info['fs_type']},文件系统可用容量{filesystem_info['available']}GB，文件系统使用率{filesystem_info['use_percentage']}。  \n"
    content += "&nbsp;\n"
    for software in softwares:
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
