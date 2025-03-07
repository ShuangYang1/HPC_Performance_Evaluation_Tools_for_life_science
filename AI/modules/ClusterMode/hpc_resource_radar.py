import json
import statistics

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

translate = {"singletask": "单任务", "multitasks": "大批量任务"}
font_path = "modules/SingleNodeMode/SimHei.ttf"
my_font = matplotlib.font_manager.FontProperties(fname=font_path)


def reduce_vector_dim(vector):
    max_dim = 10000
    if len(vector) > max_dim:
        step = len(vector) // max_dim
        reduced_vector = vector[::step]
        return reduced_vector[:max_dim]
    else:
        return vector


def hpc_resource_radar(cpu, ram, pfs_recv, pfs_send, power, image_path, datatype):
    time = len(cpu)
    cpu = reduce_vector_dim(cpu)
    ram = reduce_vector_dim(ram)
    pfs_recv = reduce_vector_dim(pfs_recv)
    pfs_send = reduce_vector_dim(pfs_send)
    power = reduce_vector_dim(power)
    plt.rcParams["axes.facecolor"] = "white"
    angles = np.linspace(0.5 * np.pi, -1.5 * np.pi, len(ram), endpoint=False)
    ram = np.concatenate((ram, [ram[0]]))
    cpu = np.concatenate((cpu, [cpu[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    pfs_recv = np.concatenate((pfs_recv, [pfs_recv[0]]))
    pfs_send = np.concatenate((pfs_send, [pfs_send[0]]))
    power = np.concatenate((power, [power[0]]))

    fig = plt.figure(figsize=(9, 7), dpi=1000)
    ax = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
    ax.plot(
        angles,
        cpu,
        color="darkred",
        linestyle="-",
        label="CPU",
        linewidth=1.5,
        clip_on=False,
    )
    ax.grid(color="lightgrey", linewidth=0.3)
    ax.spines["polar"].set_color("lightgray")
    ax.spines["polar"].set_linewidth(0.5)
    ax.plot(
        angles,
        list(map(lambda x: x * 100 / max(ram), ram)),
        "b-",
        label="RAM",
        linewidth=1.5,
        clip_on=False,
    )
    ax.plot(
        angles,
        list(map(lambda x: x * 100 / max(pfs_recv), pfs_recv)),
        "g-",
        label="I/O read",
        linewidth=1.5,
        clip_on=False,
    )
    ax.plot(
        angles,
        list(map(lambda x: x * 100 / max(pfs_send), pfs_send)),
        color="deeppink",
        linestyle="-",
        label="I/O write",
        linewidth=1.5,
        clip_on=False,
    )
    ax.plot(
        angles,
        list(map(lambda x: x * 100 / max(power), power)),
        color="orange",
        linestyle="-",
        label="Power",
        linewidth=1.5,
        clip_on=False,
    )

    bin = 360 // 5

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
    if max(pfs_recv) <= 1024:
        ax1.set_yticklabels(
            [
                str(round(max(pfs_recv) / 5, 2)) + "B/s",
                str(round(max(pfs_recv) * 2 / 5, 2)) + "B/s",
                str(round(max(pfs_recv) * 3 / 5, 2)) + "B/s",
                str(round(max(pfs_recv) * 4 / 5, 2)) + "B/s",
                str(round(max(pfs_recv), 2)) + "B/s",
            ]
        )
    elif max(pfs_recv) > 1024 and max(pfs_recv) <= 1024 * 1024:
        ax1.set_yticklabels(
            [
                str(round(max(pfs_recv) / 5 / 1024, 2)) + "KB/s",
                str(round(max(pfs_recv) * 2 / 5 / 1024, 2)) + "KB/s",
                str(round(max(pfs_recv) * 3 / 5 / 1024, 2)) + "KB/s",
                str(round(max(pfs_recv) * 4 / 5 / 1024, 2)) + "KB/s",
                str(round(max(pfs_recv) / 1024, 2)) + "KB/s",
            ]
        )
    elif max(pfs_recv) > 1024 * 1024 and max(pfs_recv) <= 1024 * 1024 * 1024:
        ax1.set_yticklabels(
            [
                str(round(max(pfs_recv) / 5 / 1024 / 1024, 2)) + "MB/s",
                str(round(max(pfs_recv) * 2 / 5 / 1024 / 1024, 2)) + "MB/s",
                str(round(max(pfs_recv) * 3 / 5 / 1024 / 1024, 2)) + "MB/s",
                str(round(max(pfs_recv) * 4 / 5 / 1024 / 1024, 2)) + "MB/s",
                str(round(max(pfs_recv) / 1024 / 1024, 2)) + "MB/s",
            ]
        )
    elif max(pfs_recv) > 1024 * 1024 * 1024:
        ax1.set_yticklabels(
            [
                str(round(max(pfs_recv) / 5 / 1024 / 1024 / 1024, 2)) + "GB/s",
                str(round(max(pfs_recv) * 2 / 5 / 1024 / 1024 / 1024, 2)) + "GB/s",
                str(round(max(pfs_recv) * 3 / 5 / 1024 / 1024 / 1024, 2)) + "GB/s",
                str(round(max(pfs_recv) * 4 / 5 / 1024 / 1024 / 1024, 2)) + "GB/s",
                str(round(max(pfs_recv) / 1024 / 1024 / 1024, 2)) + "GB/s",
            ]
        )
    ax1.tick_params(axis="y", colors="g", labelsize=7)
    ax1.grid(color="lightgrey", linewidth=0.3)
    ax1.spines["polar"].set_color("lightgray")
    ax1.spines["polar"].set_linewidth(0.5)
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
    ax2.grid(color="lightgrey", linewidth=0.3)
    ax2.spines["polar"].set_color("lightgray")
    ax2.spines["polar"].set_linewidth(0.5)
    ax2.set_xticklabels([])

    ax3 = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
    ax3.patch.set_visible(False)
    ax3.grid("off")
    ax3.xaxis.set_visible(False)
    ax3.set_rlabel_position(bin * 1)
    ax3.set_yticks([20, 40, 60, 80, 100])
    ax3.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
    ax3.tick_params(axis="y", colors="darkred", labelsize=7)
    ax3.grid(color="lightgrey", linewidth=0.3)
    ax3.spines["polar"].set_color("lightgray")
    ax3.spines["polar"].set_linewidth(0.5)
    ax3.set_xticklabels([])

    ax4 = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
    ax4.patch.set_visible(False)
    ax4.grid("off")
    ax4.xaxis.set_visible(False)
    ax4.set_rlabel_position(bin * 0)
    ax4.set_yticks([20, 40, 60, 80, 100])
    if max(pfs_send) < 1024:
        ax4.set_yticklabels(
            [
                str(round(max(pfs_send) / 5, 2)) + "B/s",
                str(round(max(pfs_send) * 2 / 5, 2)) + "B/s",
                str(round(max(pfs_send) * 3 / 5, 2)) + "B/s",
                str(round(max(pfs_send) * 4 / 5, 2)) + "B/s",
                str(round(max(pfs_send), 2)) + "B/s",
            ]
        )
    elif max(pfs_send) >= 1024 and max(pfs_send) < 1024 * 1024:
        ax4.set_yticklabels(
            [
                str(round(max(pfs_send) / 5 / 1024, 2)) + "KB/s",
                str(round(max(pfs_send) * 2 / 5 / 1024, 2)) + "KB/s",
                str(round(max(pfs_send) * 3 / 5 / 1024, 2)) + "KB/s",
                str(round(max(pfs_send) * 4 / 5 / 1024, 2)) + "KB/s",
                str(round(max(pfs_send) / 1024, 2)) + "KB/s",
            ]
        )
    elif max(pfs_send) >= 1024 * 1024 and max(pfs_send) < 1024 * 1024 * 1024:
        ax4.set_yticklabels(
            [
                str(round(max(pfs_send) / 5 / 1024 / 1024, 2)) + "MB/s",
                str(round(max(pfs_send) * 2 / 5 / 1024 / 1024, 2)) + "MB/s",
                str(round(max(pfs_send) * 3 / 5 / 1024 / 1024, 2)) + "MB/s",
                str(round(max(pfs_send) * 4 / 5 / 1024 / 1024, 2)) + "MB/s",
                str(round(max(pfs_send) / 1024 / 1024, 2)) + "MB/s",
            ]
        )
    elif max(pfs_send) >= 1024 * 1024 * 1024:
        ax4.set_yticklabels(
            [
                str(round(max(pfs_send) / 5 / 1024 / 1024 / 1024, 2)) + "GB/s",
                str(round(max(pfs_send) * 2 / 5 / 1024 / 1024 / 1024, 2)) + "GB/s",
                str(round(max(pfs_send) * 3 / 5 / 1024 / 1024 / 1024, 2)) + "GB/s",
                str(round(max(pfs_send) * 4 / 5 / 1024 / 1024 / 1024, 2)) + "GB/s",
                str(round(max(pfs_send) / 1024 / 1024 / 1024, 2)) + "GB/s",
            ]
        )
    ax4.tick_params(axis="y", colors="deeppink", labelsize=7)
    ax4.grid(color="lightgrey", linewidth=0.3)
    ax4.spines["polar"].set_color("lightgray")
    ax4.spines["polar"].set_linewidth(0.5)
    ax4.set_xticklabels([])

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
    ax5.grid(color="lightgrey", linewidth=0.3)
    ax5.spines["polar"].set_color("lightgray")
    ax5.spines["polar"].set_linewidth(0.5)
    ax5.set_xticklabels([])

    software = datatype[0]
    best_mode = datatype[2]
    if datatype[1] == "singletask":
        testmode = translate[datatype[1]]
        plt.title(
            f"{software} {testmode}{best_mode}卡计算资源使用情况（全程{time}s）",
            fontproperties=my_font,
            fontsize=10,
            y=1.05,
        )
    else:
        multitasks_count = datatype[1]
        plt.title(
            f"{software} {multitasks_count}个任务并行资源使用情况（全程{time}s）",
            fontproperties=my_font,
            fontsize=10,
            y=1.05,
        )

    matplotlib.rcParams["agg.path.chunksize"] = 180000
    matplotlib.rcParams["path.simplify_threshold"] = 1
    fig.legend(loc="upper right")
    # ax.legend(bbox_to_anchor=(1.1, 1.1))
    fig.savefig(image_path)
    return True


def hpc_resource_radar_gpu(gpu, gpu_mem, image_path, datatype):
    time = len(gpu)
    gpu = reduce_vector_dim(gpu)
    gpu_mem = reduce_vector_dim(gpu_mem)
    plt.rcParams["axes.facecolor"] = "white"
    angles = np.linspace(0.5 * np.pi, -1.5 * np.pi, len(gpu), endpoint=False)
    gpu = np.concatenate((gpu, [gpu[0]]))
    gpu_mem = np.concatenate((gpu_mem, [gpu_mem[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(9, 7), dpi=1000)
    ax = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")

    bin = 360 // 2

    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="lightgrey", linewidth=0.3)
    ax.spines["polar"].set_color("lightgray")
    ax.spines["polar"].set_linewidth(0.5)

    ax1 = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
    ax1.patch.set_visible(False)
    ax1.grid("off")
    ax1.xaxis.set_visible(False)
    ax1.set_rlabel_position(bin * 0)
    ax1.set_yticks([20, 40, 60, 80, 100])
    ax1.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
    ax1.tick_params(axis="y", colors="red", labelsize=7)
    ax1.set_xticklabels([])
    ax1.grid(color="lightgrey", linewidth=0.3)
    ax1.spines["polar"].set_color("lightgray")
    ax1.spines["polar"].set_linewidth(0.5)

    ax2 = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")
    ax2.patch.set_visible(False)
    ax2.grid("off")
    ax2.xaxis.set_visible(False)
    ax2.set_rlabel_position(bin * 1)
    ax2.set_yticks([20, 40, 60, 80, 100])
    ax2.set_yticklabels(
        [
            str(round(max(gpu_mem) / 5, 2)) + "GB",
            str(round(max(gpu_mem) * 2 / 5, 2)) + "GB",
            str(round(max(gpu_mem) * 3 / 5, 2)) + "GB",
            str(round(max(gpu_mem) * 4 / 5, 2)) + "GB",
            str(round(max(gpu_mem), 2)) + "GB",
        ]
    )
    ax2.tick_params(axis="y", colors="dodgerblue", labelsize=7)
    ax2.set_xticklabels([])
    ax2.grid(color="lightgrey", linewidth=0.3)
    ax2.spines["polar"].set_color("lightgray")
    ax2.spines["polar"].set_linewidth(0.5)
    ax.plot(
        angles,
        gpu,
        color="red",
        linestyle="-",
        label="GPU",
        linewidth=1.5,
        clip_on=False,
    )
    ax.plot(
        angles,
        list(map(lambda x: x * 100 / max(gpu_mem), gpu_mem)),
        color="dodgerblue",
        linestyle="-",
        label="GPU Memory",
        linewidth=1.5,
        clip_on=False,
    )

    software = datatype[0]
    best_mode = datatype[2]
    if datatype[1] == "singletask":
        testmode = translate[datatype[1]]
        plt.title(
            f"{software} {testmode}{best_mode}卡计算资源使用情况（GPU）（全程{time}s）",
            fontproperties=my_font,
            fontsize=10,
            y=1.05,
        )
    else:
        multitasks_count = datatype[1]
        plt.title(
            f"{software} {multitasks_count}个任务并行资源使用情况（GPU）（全程{time}s）",
            fontproperties=my_font,
            fontsize=10,
            y=1.05,
        )

    matplotlib.rcParams["agg.path.chunksize"] = 180000
    matplotlib.rcParams["path.simplify_threshold"] = 1
    fig.legend(loc="upper right")
    # ax.legend(bbox_to_anchor=(0.8, 0.8))
    fig.savefig(image_path)
    return True


def hpc_resource_radar_split(
    cpu, ram, pfs_recv, pfs_send, power, gpu, gpu_mem, image_path, datatype
):
    time = len(cpu)
    cpu = reduce_vector_dim(cpu)
    ram = reduce_vector_dim(ram)
    pfs_recv = reduce_vector_dim(pfs_recv)
    pfs_send = reduce_vector_dim(pfs_send)
    power = reduce_vector_dim(power)
    gpu = reduce_vector_dim(gpu)
    gpu_mem = reduce_vector_dim(gpu_mem)
    plt.rcParams["axes.facecolor"] = "white"
    angles = np.linspace(0.5 * np.pi, -1.5 * np.pi, len(ram), endpoint=False)
    ram = np.concatenate((ram, [ram[0]]))
    cpu = np.concatenate((cpu, [cpu[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    pfs_recv = np.concatenate((pfs_recv, [pfs_recv[0]]))
    pfs_send = np.concatenate((pfs_send, [pfs_send[0]]))
    power = np.concatenate((power, [power[0]]))
    gpu = np.concatenate((gpu, [gpu[0]]))
    gpu_mem = np.concatenate((gpu_mem, [gpu_mem[0]]))

    fig = plt.figure(figsize=(9, 7), dpi=1000)
    ax = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8], projection="polar")

    ax.set_ylim(0, 140)
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines["polar"].set_color("lightgray")
    ax.spines["polar"].set_linewidth(0.5)
    ax.grid(color="lightgrey", linewidth=0.3)

    axtop = ax
    if max(pfs_send) < 1024:
        io_write_label = f"I/O write(0-{round(max(pfs_send), 2)}B/s)"
    elif max(pfs_send) >= 1024 and max(pfs_send) < 1024 * 1024:
        io_write_label = f"I/O write(0-{round(max(pfs_send) / 1024, 2)}KB/s)"
    elif max(pfs_send) >= 1024 * 1024 and max(pfs_send) < 1024 * 1024 * 1024:
        io_write_label = f"I/O write(0-{round(max(pfs_send) / 1024 / 1024, 2)}MB/s)"
    else:
        io_write_label = (
            f"I/O write(0-{round(max(pfs_send) / 1024 / 1024 / 1024, 2)}GB/s)"
        )
    if max(pfs_recv) < 1024:
        io_read_label = f"I/O read(0-{round(max(pfs_recv), 2)}B/s)"
    elif max(pfs_recv) >= 1024 and max(pfs_recv) < 1024 * 1024:
        io_read_label = f"I/O read(0-{round(max(pfs_recv) / 1024, 2)}KB/s)"
    elif max(pfs_recv) >= 1024 * 1024 and max(pfs_recv) < 1024 * 1024 * 1024:
        io_read_label = f"I/O read(0-{round(max(pfs_recv) / 1024 / 1024, 2)}MB/s)"
    else:
        io_read_label = (
            f"I/O read(0-{round(max(pfs_recv) / 1024 / 1024 / 1024, 2)}GB/s)"
        )
    axtop.plot(
        angles,
        list(map(lambda x: x * 20 / max(pfs_recv), pfs_recv)),
        "g-",
        label=io_read_label,
        linewidth=1.5,
        clip_on=False,
    )
    axtop.plot(
        angles,
        list(map(lambda x: x * 20 / max(pfs_send) + 20, pfs_send)),
        color="deeppink",
        linestyle="-",
        label=io_write_label,
        linewidth=1.5,
        clip_on=False,
    )
    axtop.plot(
        angles,
        list(map(lambda x: x / 5 + 40, cpu)),
        color="firebrick",
        linestyle="-",
        label=f"CPU(0-{round(max(cpu), 2)}%)",
        linewidth=1.5,
        clip_on=False,
    )
    axtop.plot(
        angles,
        list(map(lambda x: x / 5 + 60, gpu)),
        color="red",
        linestyle="-",
        label=f"GPU(0-{round(max(gpu), 2)}%)",
        linewidth=1.5,
        clip_on=False,
    )
    axtop.plot(
        angles,
        list(map(lambda x: x * 20 / max(gpu_mem) + 80, gpu_mem)),
        color="dodgerblue",
        linestyle="-",
        label=f"GPU Memory(0-{round(max(gpu_mem), 2)}GB)",
        linewidth=1.5,
        clip_on=False,
    )
    axtop.plot(
        angles,
        list(map(lambda x: x * 20 / max(ram) + 100, ram)),
        "b-",
        label=f"RAM(0-{round(max(ram), 2)}GB)",
        linewidth=1.5,
        clip_on=False,
    )
    axtop.plot(
        angles,
        list(map(lambda x: x * 20 / max(power) + 120, power)),
        color="orange",
        linestyle="-",
        label=f"Power(0-{round(max(power), 2)}W)",
        linewidth=1.5,
        clip_on=False,
    )

    software = datatype[0]
    best_mode = datatype[2]
    if datatype[1] == "singletask":
        testmode = translate[datatype[1]]
        plt.title(
            f"{software} {testmode}{best_mode}卡计算资源使用情况（分层级）（全程{time}s）",
            fontproperties=my_font,
            fontsize=10,
            y=1.05,
        )
    else:
        multitasks_count = datatype[1]
        plt.title(
            f"{software} {multitasks_count}个任务并行资源使用情况（分层级）（全程{time}s）",
            fontproperties=my_font,
            fontsize=10,
            y=1.05,
        )

    matplotlib.rcParams["agg.path.chunksize"] = 180000
    matplotlib.rcParams["path.simplify_threshold"] = 1
    fig.legend(fontsize=8)
    fig.savefig(image_path)
    return True


def extract_cpu(jsonfile, cpus):
    cpu_total = [[] for _ in range(cpus)]
    cpu_total_rate = []
    with open(jsonfile, "r") as infile:
        data = json.load(infile)
        i = 0
        for singlecpu in data["data"]["result"]:
            for value in singlecpu["values"]:
                cpu_total[i].append(float(value[1]))
            i += 1
        for singlecpu_total in cpu_total:
            singlecpu_total_rate = [0]
            for i in range(1, len(singlecpu_total)):
                if singlecpu_total[i] - singlecpu_total[i - 1] > 1:
                    singlecpu_total_rate.append(1)
                else:
                    singlecpu_total_rate.append(
                        singlecpu_total[i] - singlecpu_total[i - 1]
                    )
            cpu_total_rate.append(singlecpu_total_rate)
    cpu = np.mean(1 - np.array(cpu_total_rate), axis=0) * 100
    return cpu


def extract_gpu(jsonfile, gpus, start, end):
    gpu_total = [[] for _ in range(gpus)]
    with open(jsonfile, "r") as infile:
        data = json.load(infile)
        i = 0
        for singlegpu in data["data"]["result"]:
            values = singlegpu["values"]
            values_dict = {int(row[0]): int(row[1]) for row in values}
            value = None
            for timestamp in range(start, end + 1):
                if timestamp in values_dict:
                    value = values_dict[timestamp]
                gpu_total[i].append(value)
            i += 1
    gpu = np.mean(np.array(gpu_total), axis=0)
    return gpu


def extract_power(jsonfile):
    power = []
    with open(jsonfile, "r") as infile:
        data = json.load(infile)
        for singleresult in data["data"]["result"]:
            for value in singleresult["values"]:
                power.append(float(value[1]))
    return np.array(power)


def extract_mem(free_jsonfile, cached_jsonfile, buffers_jsonfile, total_jsonfile):
    with open(total_jsonfile, "r") as infile:
        data = json.load(infile)
        mem_total = int(data["data"]["result"][0]["values"][0][1])
    mem_free, mem_cached, mem_buffers = [], [], []
    with open(free_jsonfile, "r") as infile:
        data = json.load(infile)
        for singleresult in data["data"]["result"]:
            for value in singleresult["values"]:
                mem_free.append(float(value[1]))
    with open(cached_jsonfile, "r") as infile:
        data = json.load(infile)
        for singleresult in data["data"]["result"]:
            for value in singleresult["values"]:
                mem_cached.append(float(value[1]))
    with open(buffers_jsonfile, "r") as infile:
        data = json.load(infile)
        for singleresult in data["data"]["result"]:
            for value in singleresult["values"]:
                mem_buffers.append(float(value[1]))
    mem = (
        (mem_total - np.array(mem_free) - np.array(mem_cached) - np.array(mem_buffers))
        / 1024
        / 1024
        / 1024
    )
    return mem


def extract_gpu_mem(jsonfile, gpus, start, end):
    gpu_mem_total = [[] for _ in range(gpus)]
    with open(jsonfile, "r") as infile:
        data = json.load(infile)
        i = 0
        for singlegpu in data["data"]["result"]:
            values = singlegpu["values"]
            values_dict = {int(row[0]): int(row[1]) for row in values}
            value = None
            for timestamp in range(start, end + 1):
                if timestamp in values_dict:
                    value = values_dict[timestamp]
                gpu_mem_total[i].append(value)
            i += 1
    gpu_mem = np.sum(np.array(gpu_mem_total), axis=0)
    return gpu_mem / 1024 / 1024 / 1024


def extract_read(jsonfile):
    read_total_rate = []
    with open(jsonfile, "r") as infile:
        data = json.load(infile)
        result = data["data"]["result"]
        read_total = [[] for _ in range(len(result))]
        i = 0
        for singledevice in result:
            for value in singledevice["values"]:
                read_total[i].append(float(value[1]))
            i += 1
        for singledevice_total in read_total:
            singledevice_total_rate = [0]
            for i in range(1, len(singledevice_total)):
                singledevice_total_rate.append(
                    singledevice_total[i] - singledevice_total[i - 1]
                )
            read_total_rate.append(singledevice_total_rate)
    read_rate = np.sum(np.array(read_total_rate), axis=0)
    return np.array(read_rate)


def extract_write(jsonfile):
    write_total_rate = []
    with open(jsonfile, "r") as infile:
        data = json.load(infile)
        result = data["data"]["result"]
        write_total = [[] for _ in range(len(result))]
        i = 0
        for singledevice in result:
            for value in singledevice["values"]:
                write_total[i].append(float(value[1]))
            i += 1
        for singledevice_total in write_total:
            singledevice_total_rate = [0]
            for i in range(1, len(singledevice_total)):
                singledevice_total_rate.append(
                    singledevice_total[i] - singledevice_total[i - 1]
                )
            write_total_rate.append(singledevice_total_rate)
    write_rate = np.sum(np.array(write_total_rate), axis=0)
    return np.array(write_rate)


def summary_resource(resource_data):
    resource_summary = {}
    resource_summary["max"] = round(max(resource_data), 2)
    resource_summary["median"] = round(statistics.median(resource_data), 2)
    resource_summary["mean"] = round(statistics.mean(resource_data), 2)
    resource_summary["sum"] = round(sum(resource_data), 2)
    return resource_summary


def extract_and_draw(
    mondir,
    bins,
    nodelist,
    cpus,
    gpus,
    outputfile,
    gpu_outputfile,
    split_outputfile,
    datatype,
):
    (
        cpu_array,
        power_array,
        mem_array,
        read_array,
        write_array,
        gpu_array,
        gpu_mem_array,
    ) = [], [], [], [], [], [], []
    for node in nodelist:
        (
            cpu_data_node,
            power_data_node,
            mem_data_node,
            read_data_node,
            write_data_node,
            gpu_data_node,
            gpu_mem_data_node,
        ) = (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
        for i in range(len(bins)):
            start = bins[i][0]
            end = bins[i][1]
            cpu_data_node = np.append(
                cpu_data_node, extract_cpu(f"{mondir}/{node}_cpu_{i}.json", cpus)
            )
            power_data_node = np.append(
                power_data_node, extract_power(f"{mondir}/{node}_power_{i}.json")
            )
            mem_data_node = np.append(
                mem_data_node,
                extract_mem(
                    f"{mondir}/{node}_memFree_{i}.json",
                    f"{mondir}/{node}_memCached_{i}.json",
                    f"{mondir}/{node}_memBuffers_{i}.json",
                    f"{mondir}/memTotal.json",
                ),
            )
            read_data_node = np.append(
                read_data_node, extract_read(f"{mondir}/{node}_read_{i}.json")
            )
            write_data_node = np.append(
                write_data_node, extract_write(f"{mondir}/{node}_write_{i}.json")
            )
            gpu_data_node = np.append(
                gpu_data_node,
                extract_gpu(f"{mondir}/{node}_gpu_{i}.json", gpus, start, end),
            )
            gpu_mem_data_node = np.append(
                gpu_mem_data_node,
                extract_gpu_mem(f"{mondir}/{node}_gpuMem_{i}.json", gpus, start, end),
            )
        cpu_array.append(cpu_data_node)
        power_array.append(power_data_node)
        mem_array.append(mem_data_node)
        read_array.append(read_data_node)
        write_array.append(write_data_node)
        gpu_array.append(gpu_data_node)
        gpu_mem_array.append(gpu_mem_data_node)

    cpu_data = np.mean(cpu_array, axis=0)
    power_data = np.sum(power_array, axis=0)
    mem_data = np.sum(mem_array, axis=0)
    read_data = np.sum(read_array, axis=0)
    write_data = np.sum(write_array, axis=0)
    gpu_data = np.mean(gpu_array, axis=0)
    gpu_mem_data = np.sum(gpu_mem_array, axis=0)

    hpc_resource_radar(
        cpu_data, mem_data, read_data, write_data, power_data, outputfile, datatype
    )
    hpc_resource_radar_gpu(gpu_data, gpu_mem_data, gpu_outputfile, datatype)
    hpc_resource_radar_split(
        cpu_data,
        mem_data,
        read_data,
        write_data,
        power_data,
        gpu_data,
        gpu_mem_data,
        split_outputfile,
        datatype,
    )
    cpu_summary = summary_resource(cpu_data)
    mem_summary = summary_resource(mem_data)
    read_summary = summary_resource(read_data)
    write_summary = summary_resource(write_data)
    power_summary = summary_resource(power_data)
    gpu_summary = summary_resource(gpu_data)
    gpu_mem_summary = summary_resource(gpu_mem_data)
    return (
        cpu_summary,
        mem_summary,
        read_summary,
        write_summary,
        power_summary,
        gpu_summary,
        gpu_mem_summary,
    )


if __name__ == "__main__":
    nodelist = ["c05b26n04"]
    extract_and_draw(
        "tmp/GROMACS_1_8/",
        [(1736836352, 1736836386)],
        nodelist,
        64,
        8,
        "GROMACS_1_8.png",
        "GROMACS_1_8_gpu.png",
        "GROMACS_1_8_split.png",
    )
