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
gpus=config.get("gpus")
cpus=config.get("cpus")
partition=config.get("partition")
conda_path=config.get("conda_path",None)
dsdp_env=config.get("dsdp_env",None)

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
                score=cal_score(software)
                scores.append(score*weights[software])
                outfile.write(software+" score: "+str(round(score,2))+"\n")
            outfile.write("Total score: "+str(round(sum(scores),2))+"\n")

rule sponge:
    input:
        mdin="{pwd}/dataset/SPONGE/data/sponge_nvt.in",
        parm7="{pwd}/dataset/SPONGE/data/Q.parm7",
        rst7="{pwd}/dataset/SPONGE/data/Q.rst7"
    output:
        flag=touch("{pwd}/log/{mode}/SPONGE/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/SPONGE/script"):
            os.makedirs(f"{pwd}/result/{mode}/SPONGE/script")
        script_path=f"{pwd}/result/{mode}/SPONGE/script/test.sh"
        make_sbatch(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,gpus=1,software="SPONGE",pwd=pwd,script_path=script_path)
        run_slurm_job(f"sbatch {script_path}")

rule gromacs:
    input:
        tpr="{pwd}/dataset/GROMACS/61k-atoms/benchmark.tpr"
    output:
        flag=touch("{pwd}/log/{mode}/GROMACS/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/GROMACS/script"):
            os.makedirs(f"{pwd}/result/{mode}/GROMACS/script")
        script_path=f"{pwd}/result/{mode}/GROMACS/script/test.sh"
        cpus_per_task=cpus//gpus
        make_sbatch(partition=partition,nodes=1,ntasks=gpus,cpus_per_task=cpus_per_task,gpus=gpus,software="GROMACS",pwd=pwd,script_path=script_path)
        run_slurm_job(f"sbatch {script_path}")

rule amber:
    input:
        mdin="{pwd}/dataset/AMBER/61k-atoms/benchmark.in",
        top="{pwd}/dataset/AMBER/61k-atoms/benchmark.top",
        rst="{pwd}/dataset/AMBER/61k-atoms/benchmark.rst"
    output:
        flag=touch("{pwd}/log/{mode}/AMBER/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/AMBER/script"):
            os.makedirs(f"{pwd}/result/{mode}/AMBER/script")
        script_path=f"{pwd}/result/{mode}/AMBER/script/test.sh"
        cpus_per_task=cpus//gpus
        make_sbatch(partition=partition,nodes=1,ntasks=gpus,cpus_per_task=cpus_per_task,gpus=gpus,software="AMBER",pwd=pwd,script_path=script_path)
        run_slurm_job(f"sbatch {script_path}")

rule dsdp:
    input:
        dataset="{pwd}/dataset/DSDP/DSDP_dataset"
    output:
        flag=touch("{pwd}/log/{mode}/DSDP/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/DSDP/script"):
            os.makedirs(f"{pwd}/result/{mode}/DSDP/script")
        script_path=f"{pwd}/result/{mode}/DSDP/script/test.sh"
        make_sbatch(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,gpus=1,software="DSDP",pwd=pwd,script_path=script_path,conda_path=conda_path,dsdp_env=dsdp_env)
        run_slurm_job(f"sbatch {script_path}")