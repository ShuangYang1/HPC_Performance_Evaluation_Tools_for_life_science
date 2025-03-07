import os
import re

import numpy as np
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import adfuller


def adf(data):
    data_series = np.array(data)
    adf_result = adfuller(data_series)
    pvalue = adf_result[1]
    if pvalue < 0.05:
        return True
    else:
        return False


def check(data1, data2):
    start_index = len(data2) - len(data2) // 5
    data1_b = data1[start_index:]
    data2_b = data2[start_index:]
    mean1 = np.mean(data1_b)
    mean2 = np.mean(data2_b)
    std1 = np.std(data1_b, ddof=1)
    std2 = np.std(data2_b, ddof=1)
    sem1 = std1 / np.sqrt(len(data1_b))
    sem2 = std2 / np.sqrt(len(data2_b))
    if abs(mean1 - mean2) <= (sem1 + sem2):
        return True
    else:
        return False


def check_result(software):
    if software == "SPONGE":
        std_file = "dataset/std_result/sponge.mdout"
        result_file = f"result/TestMode/{software}/mdout.txt"
        energy1, energy2 = [], []
        temp1, temp2 = [], []
        with open(std_file, "r") as infile:
            for line in infile:
                t = line.strip().split()[2]
                if t != "temperature":
                    temp1.append(float(t))
                e = line.strip().split()[3]
                if e != "potential":
                    energy1.append(float(e))
        with open(result_file, "r") as infile:
            for line in infile:
                t = line.strip().split()[2]
                if t != "temperature":
                    temp2.append(float(t))
                e = line.strip().split()[3]
                if e != "potential":
                    energy2.append(float(e))
        if check(temp1, temp2) and check(energy1, energy2) and adf(energy2):
            return True
        else:
            print("There is something wrong with the SPONGE result.")
            return False
    elif software == "GROMACS":
        std_energy_file = "dataset/std_result/gromacs_energy.xvg"
        std_rmsd_file = "dataset/std_result/gromacs_rmsd.xvg"
        result_energy = f"result/TestMode/{software}/ener.edr"
        energy_file = f"result/TestMode/{software}/energy.xvg"
        os.system(
            f'echo -e "Temperature\nPotential" |gmx energy -f {result_energy} -o {energy_file}'
        )
        energy1, energy2 = [], []
        temp1, temp2 = [], []
        with open(std_energy_file, "r") as infile:
            for line in infile:
                if line.startswith("#") or line.startswith("@"):
                    continue
                else:
                    e = line.strip().split()[1]
                    t = line.strip().split()[2]
                    energy1.append(float(e))
                    temp1.append(float(t))
        with open(energy_file, "r") as infile:
            for line in infile:
                if line.startswith("#") or line.startswith("@"):
                    continue
                else:
                    e = line.strip().split()[1]
                    t = line.strip().split()[2]
                    energy2.append(float(e))
                    temp2.append(float(t))
        result_traj = f"result/TestMode/{software}/traj.trr"
        rmsd_file = f"result/TestMode/{software}/rmsd.xvg"
        os.system(
            f'echo -e "4\n4" | gmx rms -s dataset/GROMACS/61k-atoms/benchmark.tpr -f {result_traj} -o {rmsd_file}'
        )
        rmsd1, rmsd2 = [], []
        with open(std_rmsd_file, "r") as infile:
            for line in infile:
                if line.startswith("#") or line.startswith("@"):
                    continue
                else:
                    rmsd = line.strip().split()[1]
                    rmsd1.append(float(rmsd))
        with open(rmsd_file, "r") as infile:
            for line in infile:
                if line.startswith("#") or line.startswith("@"):
                    continue
                else:
                    rmsd = line.strip().split()[1]
                    rmsd2.append(float(rmsd))
        if (
            adf(energy2)
            and adf(rmsd2)
            and check(energy1, energy2)
            and check(rmsd1, rmsd2)
            and check(temp1, temp2)
        ):
            return True
        else:
            print("There is something wrong with the GROMACS result.")
            return False
    elif software == "AMBER":
        std_energy_file = "dataset/std_result/amber_energy.mdout"
        std_rmsd_file = "dataset/std_result/amber_rmsd.dat"
        energy_file = f"result/TestMode/{software}/mdout"
        energy1, energy2, temp1, temp2 = [], [], [], []
        with open(std_energy_file, "r") as infile:
            for line in infile:
                match_e = re.search(r"EPtot\s+=\s+([-+]?\d*\.\d+|\d+)", line)
                if match_e:
                    energy1.append(float(match_e.group(1)))
                match_t = re.search(r"TEMP\(K\)\s*=\s*([\d\.]+)", line)
                if match_t:
                    temp1.append(float(match_t.group(1)))
        energy1.pop()
        energy1.pop()
        temp1.pop()
        temp1.pop()
        with open(energy_file, "r") as infile:
            for line in infile:
                match_e = re.search(r"EPtot\s+=\s+([-+]?\d*\.\d+|\d+)", line)
                if match_e:
                    energy2.append(float(match_e.group(1)))
                match_t = re.search(r"TEMP\(K\)\s*=\s*([\d\.]+)", line)
                if match_t:
                    temp2.append(float(match_t.group(1)))
        energy2.pop()
        energy2.pop()
        temp2.pop()
        temp2.pop()
        os.system("cpptraj -i dataset/std_result/amber_rmsd.in")
        rmsd_file = f"result/TestMode/{software}/rmsd.dat"
        rmsd1, rmsd2 = [], []
        with open(std_rmsd_file, "r") as infile:
            for line in infile:
                rmsd = line.strip().split()[1]
                if "RMSD" not in rmsd:
                    rmsd1.append(float(rmsd))
        with open(rmsd_file, "r") as infile:
            for line in infile:
                rmsd = line.strip().split()[1]
                if "RMSD" not in rmsd:
                    rmsd2.append(float(rmsd))
        if (
            adf(energy2)
            and adf(rmsd2)
            and check(energy1, energy2)
            and check(rmsd1, rmsd2)
            and check(temp1, temp2)
        ):
            return True
        else:
            print("There is something wrong with the AMBER result.")
            return False
    elif software == "DSDP":
        std_file = "dataset/std_result/dsdp.out"
        data1, data2 = [], []
        with open(std_file) as f:
            for line in f:
                data1.append(float(line.strip()))
        listdir = os.listdir(f"result/TestMode/{software}/test_output")
        for filename in listdir:
            with open(f"result/TestMode/{software}/test_output/{filename}") as f:
                line = f.readline()
                data2.append(float(line.strip().split()[1]))
        statistic, p_value = ks_2samp(data1, data2)
        if p_value > 0.05:
            return True
        else:
            print("There is something wrong with the DSDP result.")
            return False
