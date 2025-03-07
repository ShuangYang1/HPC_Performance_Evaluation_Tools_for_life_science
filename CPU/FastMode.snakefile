mode="FastMode"
configfile: f"config/{mode}/config.yaml"

softwares = []
with open("SoftwaresWeight.txt","r") as infile:
    for line in infile:
        if line.strip().split()[1]!="0":
            softwares.append(line.strip().split()[0])
import os
from modules.FastMode.make_sbatch import make_sbatch
from modules.FastMode.cal_score import cal_score
from modules.submit import run_slurm_job
pwd=os.getcwd()
cpus=config.get("cpus")
partition=config.get("partition")

rule all:
    input:
        expand("{pwd}/log/{mode}/{software}/run.done",mode=mode,software=softwares,pwd=pwd)
    run:
        with open(mode+"_result.txt","w") as outfile:
            scores=[]
            weights={}
            if os.path.exists("softwaresWeights.txt"):
                with open("softwaresWeights.txt", "r") as weightsfile:
                    for line in weightsfile:
                        line = line.strip().split()
                        weights[line[0]] = float(line[1])
                for software in softwares:
                    if software not in weights:
                        weights[software] = 1.0
            else:
                for software in softwares:
                    weights[software] = 1.0
            for software in softwares:
                f=f"{pwd}/log/{mode}/{software}/test.log"
                score=cal_score(software,f)
                scores.append(score*weights[software])
                outfile.write(software+" score: "+str(round(score,2))+"\n")
            outfile.write("Total score: "+str(round(sum(scores),2))+"\n")

rule bwa:
    input:
        genome="{pwd}/dataset/BWA/hg18/hg18.fa.gz",
        fastq1="{pwd}/dataset/BWA/ERR000589/ERR000589_1.fastq",
        fastq2="{pwd}/dataset/BWA/ERR000589/ERR000589_2.fastq"
    output:
        flag=touch("{pwd}/log/{mode}/BWA/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/BWA/script"):
            os.makedirs(f"{pwd}/result/{mode}/BWA/script")
        script_path=f"{pwd}/result/{mode}/BWA/script/test.sh"
        make_sbatch(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,software="BWA",pwd=pwd,script_path=script_path)
        run_slurm_job(f"sbatch {script_path}")

rule spades:
    input:
        fastq1="{pwd}/dataset/SPAdes/EAS20_8/s_6_1.fastq",
        fastq2="{pwd}/dataset/SPAdes/EAS20_8/s_6_2.fastq"
    output:
        flag=touch("{pwd}/log/{mode}/SPAdes/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/SPAdes/script"):
            os.makedirs(f"{pwd}/result/{mode}/SPAdes/script")
        script_path=f"{pwd}/result/{mode}/SPAdes/script/test.sh"
        make_sbatch(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,software="SPAdes",pwd=pwd,script_path=script_path)
        run_slurm_job(f"sbatch {script_path}")

rule bismark:
    input:
        genome="{pwd}/dataset/Bismark/hg18/",
        fastq="{pwd}/dataset/Bismark/SRR020138.fastq"
    output:
        flag=touch("{pwd}/log/{mode}/Bismark/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/Bismark/script"):
            os.makedirs(f"{pwd}/result/{mode}/Bismark/script")
        script_path=f"{pwd}/result/{mode}/Bismark/script/test.sh"
        make_sbatch(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,software="Bismark",pwd=pwd,script_path=script_path)
        run_slurm_job(f"sbatch {script_path}")

rule star:
    input:
        genome="{pwd}/dataset/STAR/mm39StarIndex",
        fastq="{pwd}/dataset/STAR/SRR6821753.fastq"
    output:
        flag=touch("{pwd}/log/{mode}/STAR/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/STAR/script"):
            os.makedirs(f"{pwd}/result/{mode}/STAR/script")
        script_path=f"{pwd}/result/{mode}/STAR/script/test.sh"
        make_sbatch(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,software="STAR",pwd=pwd,script_path=script_path)
        run_slurm_job(f"sbatch {script_path}")

rule cellranger:
    input:
        genome="{pwd}/dataset/Cellranger/refdata-gex-GRCh38-2024-A/",
        fastq="{pwd}/dataset/Cellranger/pbmc_1k_v3_fastqs"
    output:
        flag=touch("{pwd}/log/{mode}/Cellranger/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/Cellranger/script"):
            os.makedirs(f"{pwd}/result/{mode}/Cellranger/script")
        script_path=f"{pwd}/result/{mode}/Cellranger/script/test.sh"
        make_sbatch(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,software="Cellranger",pwd=pwd,script_path=script_path)
        run_slurm_job(f"sbatch {script_path}")

rule gatk:
    input:
        genome="{pwd}/dataset/GATK/GRCh38_full_analysis_set_plus_decoy_hla.fa",
        bam="{pwd}/dataset/GATK/NA12878_24RG_small.hg38.bam"
    output:
        flag=touch("{pwd}/log/{mode}/GATK/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/GATK/script"):
            os.makedirs(f"{pwd}/result/{mode}/GATK/script")
        script_path=f"{pwd}/result/{mode}/GATK/script/test.sh"
        make_sbatch(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,software="GATK",pwd=pwd,script_path=script_path)
        run_slurm_job(f"sbatch {script_path}")
