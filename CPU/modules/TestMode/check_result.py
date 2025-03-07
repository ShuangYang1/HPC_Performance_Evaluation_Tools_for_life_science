import filecmp
import os
import re
import subprocess


def check_result(software):
    if software == "BWA":
        result_file = f"result/TestMode/{software}/result.sam"
        command = ["samtools", "flagstat", result_file]
        output = subprocess.run(command, capture_output=True, text=True).stdout
        mapped_match = re.search(
            r"^\d+\s+\+\s+\d+\s+mapped\s+\(([\d\.]+)%", output, re.MULTILINE
        )
        if mapped_match:
            mapped_percentage = float(mapped_match.group(1))
            if mapped_percentage != 98.77:
                print("There is something wrong with the BWA result.")
        else:
            print("There is something wrong with the BWA result.")
        primary_mapped_match = re.search(
            r"^\d+\s+\+\s+\d+\s+primary mapped\s+\(([\d\.]+)%", output, re.MULTILINE
        )
        if primary_mapped_match:
            primary_mapped_percentage = float(primary_mapped_match.group(1))
            if primary_mapped_percentage != 98.77:
                print("There is something wrong with the BWA result.")
        else:
            print("There is something wrong with the BWA result.")
    elif software == "SPAdes":
        result_file = f"result/TestMode/{software}/scaffolds.fasta"
        std_file = "dataset/std_result/spades.fasta"
        if not filecmp.cmp(std_file, result_file, shallow=False):
            print("There is something wrong with the SPAdes result.")
    elif software == "Bismark":
        skip_start = 3
        skip_end = 1
        result_file = f"result/TestMode/{software}/SRR020138_bismark_bt2_SE_report.txt"
        std_file = "dataset/std_result/bismark_report.txt"
        with open(std_file, "r") as f:
            lines1 = f.read().splitlines()
        with open(result_file, "r") as f:
            lines2 = f.read().splitlines()
        content1 = lines1[skip_start - 1 : len(lines1) - skip_end]
        content2 = lines2[skip_start : len(lines2) - skip_end]
        if content1 != content2:
            print("There is something wrong with the Bismark result.")
    elif software == "STAR":
        skip_start = 4
        result_file = f"result/TestMode/{software}/Log.final.out"
        std_file = "dataset/std_result/star.out"
        with open(std_file, "r") as f:
            lines1 = f.read().splitlines()
        with open(result_file, "r") as f:
            lines2 = f.read().splitlines()
        content1 = lines1[skip_start:]
        content2 = lines2[skip_start:]
        if content1 != content2:
            print("There is something wrong with the STAR result.")
    elif software == "Cellranger":
        result_file = f"result/TestMode/{software}/outs/metrics_summary.csv"
        std_file = "dataset/std_result/cellranger_report.csv"
        if not filecmp.cmp(std_file, result_file, shallow=False):
            print("There is something wrong with the Cellranger result.")
    elif software == "GATK":
        result_file = f"result/TestMode/{software}/test.vcf"
        gz = f"{result_file}.gz"
        os.system(f"bgzip {result_file}")
        os.system(f"tabix -p vcf {gz}")
        command = ["vcf-compare", "dataset/std_result/gatk.vcf.gz", gz]
        output = subprocess.run(command, capture_output=True, text=True).stdout
        pattern1 = r"result/TestMode/GATK/test\.vcf\.gz\s+\(([\d\.]+)%\)"
        match1 = re.search(pattern1, output)
        if match1:
            percent = float(match1.group(1))
            if percent != 100.0:
                print("There is something wrong with the GATK result.")
        else:
            print("There is something wrong with the GATK result.")
        pattern2 = r"Number of REF mismatches:\s+(\d+)"
        match2 = re.search(pattern2, output)
        if match2:
            ref_mismatches = int(match2.group(1))
            if ref_mismatches != 0:
                print("There is something wrong with the GATK result.")
        else:
            print("There is something wrong with the GATK result.")
            ref_mismatches = None
