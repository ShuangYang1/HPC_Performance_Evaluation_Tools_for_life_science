mode="TestMode"
configfile: f"config/{mode}/config.yaml"

softwares = []
with open("SoftwaresWeight.txt","r") as infile:
    for line in infile:
        if line.strip().split()[1]!="0":
            softwares.append(line.strip().split()[0])
import os
from pathlib import Path
from modules.TestMode.make_sbatch import make_sbatch
from modules.TestMode.check_result import check_result
from modules.submit import run_slurm_job
pwd=os.getcwd()
cpus=config.get("cpus")
partition=config.get("partition")

rule all:
    input:
        expand("{pwd}/log/{mode}/{software}/run.done",mode=mode,software=softwares,pwd=pwd)
    run:
        failed=[]
        success=[]
        for software in softwares:
            if os.path.exists(f"{pwd}/log/{mode}/{software}/run.err"):
                failed.append(software)
                os.system(f"rm {pwd}/log/{mode}/{software}/run.err")
                os.system(f"rm {pwd}/log/{mode}/{software}/run.done")
            else:
                success.append(software)
        if len(failed)==0:
            print("Your softwares are fully prepared.")
        else:
            print(",".join(failed),"test failed.")
        for software in success:
            check_result(software)

rule bwa_test:
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
        job_status=run_slurm_job(f"sbatch {script_path}")
        if not job_status:
            Path(f"{pwd}/log/{mode}/BWA/run.err").touch()

rule spades_test:
    output:
        flag=touch("{pwd}/log/{mode}/SPAdes/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/SPAdes/script"):
            os.makedirs(f"{pwd}/result/{mode}/SPAdes/script")
        script_path=f"{pwd}/result/{mode}/SPAdes/script/test.sh"
        make_sbatch(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,software="SPAdes",pwd=pwd,script_path=script_path)
        job_status=run_slurm_job(f"sbatch {script_path}")
        if not job_status:
            Path(f"{pwd}/log/{mode}/SPAdes/run.err").touch()

rule bismark_test:
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
        job_status=run_slurm_job(f"sbatch {script_path}")
        if not job_status:
            Path(f"{pwd}/log/{mode}/Bismark/run.err").touch()

rule star_test:
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
        job_status=run_slurm_job(f"sbatch {script_path}")
        if not job_status:
            Path(f"{pwd}/log/{mode}/STAR/run.err").touch()

rule cellranger_test:
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
        job_status=run_slurm_job(f"sbatch {script_path}")
        if not job_status:
            Path(f"{pwd}/log/{mode}/Cellranger/run.err").touch()

rule gatk_test:
    output:
        flag=touch("{pwd}/log/{mode}/GATK/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/GATK/script"):
            os.makedirs(f"{pwd}/result/{mode}/GATK/script")
        script_path=f"{pwd}/result/{mode}/GATK/script/test.sh"
        make_sbatch(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,software="GATK",pwd=pwd,script_path=script_path)
        job_status=run_slurm_job(f"sbatch {script_path}")
        if not job_status:
            Path(f"{pwd}/log/{mode}/GATK/run.err").touch()
