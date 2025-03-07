import numpy as np
from statsmodels.tsa.stattools import adfuller
from tmtools import tm_align
from tmtools.io import get_residue_data, get_structure


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
    if software == "MindSPONGE":
        std_file = "dataset/std_result/mindsponge.log"
        result_file = f"log/TestMode/{software}/test.log"
        energy1, energy2 = [], []
        temp1, temp2 = [], []
        with open(std_file, "r") as infile:
            for line in infile:
                if "Temperature" in line:
                    e = line.split(":")[2].strip().split(",")[0]
                    energy1.append(float(e))
                    t = line.split(":")[5].strip().split(",")[0]
                    temp1.append(float(t))
        with open(result_file, "r") as infile:
            for line in infile:
                if "Temperature" in line:
                    e = line.split(":")[2].strip().split(",")[0]
                    energy2.append(float(e))
                    t = line.split(":")[5].strip().split(",")[0]
                    temp2.append(float(t))
        if check(temp1, temp2) and check(energy1, energy2) and adf(energy2):
            return True
        else:
            print("There is something wrong with the MindSPONGE result.")
            return False
    elif software == "Alphafold3":
        std_file = "dataset/std_result/alphafold3.cif"
        result_file = f"log/TestMode/{software}/2pv7/2pv7_model.cif"
        s1 = get_structure(std_file, format="mmcif")
        chain1 = next(s1.get_chains())
        coords1, seq1 = get_residue_data(chain1)
        s2 = get_structure(result_file, format="mmcif")
        chain2 = next(s2.get_chains())
        coords2, seq2 = get_residue_data(chain2)
        res = tm_align(coords1, coords2, seq1, seq2)
        if res.tm_norm_chain1 == 1.0 or res.tm_norm_chain2 == 1.0:
            return True
        else:
            print("There is something wrong with the Alphafold3 result.")
            return False
