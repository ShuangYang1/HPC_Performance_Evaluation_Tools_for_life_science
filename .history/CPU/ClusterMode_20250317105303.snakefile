mode="ClusterMode"
configfile: f"config/{mode}/config.yaml"

softwares = []
with open("SoftwaresWeight.txt","r") as infile:
    for line in infile:
        if line.strip().split()[1]!="0":
            softwares.append(line.strip().split()[0])
import os
from modules.ClusterMode.make_sbatch import make_multitasks
from modules.ClusterMode.make_report import generate_report
from modules.submit import run_slurm_job
pwd=os.getcwd()
cpus=config.get("cpus_per_node")
partition=config.get("partition")
nodes=config.get("nodelist")
nodelist=nodes.split(",")
multitasks_count=config.get("multitasks_count",50*len(nodelist))
bwa_cpus_per_task=config.get("bwa_cpus_per_task",1)
spades_cpus_per_task=config.get("spades_cpus_per_task",1)
bismark_parallel=config.get("bismark_parallel",1)
star_cpus_per_task=config.get("star_cpus_per_task",1)
cellranger_cpus_per_task=config.get("cellranger_cpus_per_task",1)
gatk_cpus_per_task=config.get("gatk_cpus_per_task",1)
node_exporter_port=config.get("node_exporter_port")
server_ip=config.get("server_ip")
server_port=config.get("server_port")
delete=config.get("delete-intermediate-files",False)


cpus_per_task_dic={}
cpus_per_task_dic["BWA"]=bwa_cpus_per_task
cpus_per_task_dic["SPAdes"]=spades_cpus_per_task
cpus_per_task_dic["Bismark"]=bismark_parallel
cpus_per_task_dic["STAR"]=star_cpus_per_task
cpus_per_task_dic["Cellranger"]=cellranger_cpus_per_task
cpus_per_task_dic["GATK"]=gatk_cpus_per_task

rule all:
    input:
        expand("{pwd}/log/{mode}/{software}/run.done",mode=mode,software=softwares,pwd=pwd),
    run:
        generate_report(multitasks_count,partition,nodelist,cpus,cpus_per_task_dic,node_exporter_port,server_ip,server_port,singlenode_scores)

rule bwa:
    input:
        genome="{pwd}/dataset/BWA/hg18/hg18.fa.gz",
        fastq="{pwd}/dataset/BWA/ERR000589"
    output:
        flag=touch("{pwd}/log/{mode}/BWA/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/BWA/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for count in range(1,multitasks_count+1):
            outdir=f"{pwd}/dataset/BWA_multitasks/{count}/"
            if not os.path.exists(outdir):
                os.system(f"mkdir -p {outdir}")
                os.system(f"cp -r {pwd}/dataset/BWA/ERR000589 {outdir}")
        multitasks_script_path=f"{script_path}/multitasks.sh"
        make_multitasks(partition=partition,nodes=1,ntasks=1,cpus_per_task=bwa_cpus_per_task,software="BWA",pwd=pwd,nodelist=nodes,multitasks_count=multitasks_count,script_path=multitasks_script_path)
        run_slurm_job(f"sbatch {multitasks_script_path}")
        if delete:
            os.system(f"rm -rf {pwd}/result/{mode}/BWA/multitasks/*")
        else:
            pass

rule spades:
    input:
        fastq="{pwd}/dataset/SPAdes/EAS20_8",
        start_flag="{pwd}/log/{mode}/BWA/run.done"
    output:
        flag=touch("{pwd}/log/{mode}/SPAdes/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/SPAdes/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for count in range(1,multitasks_count+1):
            outdir=f"{pwd}/dataset/SPAdes_multitasks/{count}/"
            if not os.path.exists(outdir):
                os.system(f"mkdir -p {outdir}")
                os.system(f"cp -r {pwd}/dataset/SPAdes/EAS20_8 {outdir}")
        multitasks_script_path=f"{script_path}/multitasks.sh"
        make_multitasks(partition=partition,nodes=1,ntasks=1,cpus_per_task=spades_cpus_per_task,software="SPAdes",pwd=pwd,nodelist=nodes,multitasks_count=multitasks_count,script_path=multitasks_script_path)
        run_slurm_job(f"sbatch {multitasks_script_path}")
        if delete:
            os.system(f"rm -rf {pwd}/result/{mode}/SPAdes/multitasks/*")
        else:
            pass

rule bismark:
    input:
        start_flag="{pwd}/log/{mode}/SPAdes/run.done",
        genome="{pwd}/dataset/Bismark/hg18/",
        fastq="{pwd}/dataset/Bismark/SRR020138.fastq"
    output:
        flag=touch("{pwd}/log/{mode}/Bismark/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/Bismark/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for count in range(1,multitasks_count+1):
            outdir=f"{pwd}/dataset/Bismark_multitasks/{count}/"
            if not os.path.exists(outdir):
                os.system(f"mkdir -p {outdir}")
                os.system(f"cp {pwd}/dataset/Bismark/SRR020138.fastq {outdir}")
        multitasks_script_path=f"{script_path}/multitasks.sh"
        bismark_cpus_per_task=bismark_parallel*4
        make_multitasks(partition=partition,nodes=1,ntasks=1,cpus_per_task=bismark_cpus_per_task,software="Bismark",pwd=pwd,nodelist=nodes,multitasks_count=multitasks_count,script_path=multitasks_script_path)
        run_slurm_job(f"sbatch {multitasks_script_path}")
        if delete:
            os.system(f"rm -rf {pwd}/result/{mode}/Bismark/multitasks/*")
        else:
            pass

rule star:
    input:
        start_flag="{pwd}/log/{mode}/Bismark/run.done",
        genome="{pwd}/dataset/STAR/mm39StarIndex",
        fastq="{pwd}/dataset/STAR/SRR6821753.fastq"
    output:
        flag=touch("{pwd}/log/{mode}/STAR/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/STAR/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for count in range(1,multitasks_count+1):
            outdir=f"{pwd}/dataset/STAR_multitasks/{count}/"
            if not os.path.exists(outdir):
                os.system(f"mkdir -p {outdir}")
                os.system(f"cp {pwd}/dataset/STAR/SRR6821753.fastq {outdir}")
        multitasks_script_path=f"{script_path}/multitasks.sh"
        make_multitasks(partition=partition,nodes=1,ntasks=1,cpus_per_task=star_cpus_per_task,software="STAR",pwd=pwd,nodelist=nodes,multitasks_count=multitasks_count,script_path=multitasks_script_path)
        run_slurm_job(f"sbatch {multitasks_script_path}")
        if delete:
            os.system(f"rm -rf {pwd}/result/{mode}/STAR/multitasks/*")
        else:
            pass

rule cellranger:
   input:
       start_flag="{pwd}/log/{mode}/STAR/run.done",
       genome="{pwd}/dataset/Cellranger/refdata-gex-GRCh38-2024-A/",
       fastq="{pwd}/dataset/Cellranger/pbmc_1k_v3_fastqs"
   output:
       flag=touch("{pwd}/log/{mode}/Cellranger/run.done")
   run:
        script_path=f"{pwd}/result/{mode}/Cellranger/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for count in range(1,multitasks_count+1):
            outdir=f"{pwd}/dataset/Cellranger_multitasks/{count}/"
            if not os.path.exists(outdir):
                os.system(f"mkdir -p {outdir}")
                os.system(f"cp -r {pwd}/dataset/Cellranger/pbmc_1k_v3_fastqs {outdir}")
        multitasks_script_path=f"{script_path}/multitasks.sh"
        make_multitasks(partition=partition,nodes=1,ntasks=1,cpus_per_task=cellranger_cpus_per_task,software="Cellranger",pwd=pwd,nodelist=nodes,multitasks_count=multitasks_count,script_path=multitasks_script_path)
        run_slurm_job(f"sbatch {multitasks_script_path}")
        if delete:
            os.system(f"rm -rf {pwd}/result/{mode}/Cellranger/multitasks/*")
        else:
            pass

rule gatk:
   input:
       start_flag="{pwd}/log/{mode}/Cellranger/run.done",
       genome="{pwd}/dataset/GATK/GRCh38_full_analysis_set_plus_decoy_hla.fa",
       bam="{pwd}/dataset/GATK/NA12878_24RG_small.hg38.bam"
   output:
       flag=touch("{pwd}/log/{mode}/GATK/run.done")
   run:
        script_path=f"{pwd}/result/{mode}/GATK/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for count in range(1,multitasks_count+1):
            outdir=f"{pwd}/dataset/GATK_multitasks/{count}/"
            if not os.path.exists(outdir):
                os.system(f"mkdir -p {outdir}")
                os.system(f"cp {pwd}/dataset/GATK/NA12878_24RG_small.hg38.bam* {outdir}")
        multitasks_script_path=f"{script_path}/multitasks.sh"
        make_multitasks(partition=partition,nodes=1,ntasks=1,cpus_per_task=gatk_cpus_per_task,software="GATK",pwd=pwd,nodelist=nodes,multitasks_count=multitasks_count,script_path=multitasks_script_path)
        run_slurm_job(f"sbatch {multitasks_script_path}")
        if delete:
            os.system(f"rm -rf {pwd}/result/{mode}/GATK/multitasks/*")
        else:
            pass
