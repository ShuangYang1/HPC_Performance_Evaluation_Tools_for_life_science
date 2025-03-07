mode="TestMode"
configfile: f"config/{mode}/config.yaml"

softwares = []
with open("SoftwaresWeight.txt","r") as infile:
    for line in infile:
        if line.strip().split()[1]!="0":
            softwares.append(line.strip().split()[0])
import os
from modules.TestMode.make_sbatch import make_sbatch
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
        for software in softwares:
            if os.path.exists(f"{pwd}/log/{mode}/{software}/run.err"):
                print(f"{software} failed")
                os.system(f"rm {pwd}/log/{mode}/{software}/run.err")
                os.system(f"rm {pwd}/log/{mode}/{software}/run.done")

rule sponge_test:
    input:
        mdin="{pwd}/dataset/SPONGE/sponge_nvt.in",
        parm7="{pwd}/dataset/SPONGE/Q.parm7",
        rst7="{pwd}/dataset/SPONGE/Q.rst7"
    output:
        flag=touch("{pwd}/log/{mode}/SPONGE/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/SPONGE/script"):
            os.makedirs(f"{pwd}/result/{mode}/SPONGE/script")
        script_path=f"{pwd}/result/{mode}/SPONGE/script/test.sh"
        make_sbatch(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,gpus=1,software="SPONGE",pwd=pwd,script_path=script_path)
        job_status = run_slurm_job(f"sbatch {script_path}")
        if not job_status:
            Path(f"{pwd}/log/{mode}/SPONGE/run.err").touch()

rule gromacs_test:
    input:
        tpr="{pwd}/dataset/GROMACS/20k-atoms/benchmark.tpr"
    output:
        flag=touch("{pwd}/log/{mode}/GROMACS/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/GROMACS/script"):
            os.makedirs(f"{pwd}/result/{mode}/GROMACS/script")
        script_path=f"{pwd}/result/{mode}/GROMACS/script/test.sh"
        cpus_per_task=cpus//gpus
        make_sbatch(partition=partition,nodes=1,ntasks=gpus,cpus_per_task=cpus_per_task,gpus=gpus,software="GROMACS",pwd=pwd,script_path=script_path)
        job_status = run_slurm_job(f"sbatch {script_path}")
        if not job_status:
            Path(f"{pwd}/log/{mode}/GROMACS/run.err").touch()

rule amber_test:
    input:
        mdin="{pwd}/dataset/AMBER/20k-atoms/benchmark.in",
        top="{pwd}/dataset/AMBER/20k-atoms/benchmark.top",
        rst="{pwd}/dataset/AMBER/20k-atoms/benchmark.rst"
    output:
        flag=touch("{pwd}/log/{mode}/AMBER/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/AMBER/script"):
            os.makedirs(f"{pwd}/result/{mode}/AMBER/script")
        script_path=f"{pwd}/result/{mode}/AMBER/script/test.sh"
        cpus_per_task=cpus//gpus
        make_sbatch(partition=partition,nodes=1,ntasks=gpus,cpus_per_task=cpus_per_task,gpus=gpus,software="AMBER",pwd=pwd,script_path=script_path)
        job_status = run_slurm_job(f"sbatch {script_path}")
        if not job_status:
            Path(f"{pwd}/log/{mode}/AMBER/run.err").touch()

rule dsdp_test:
    input:
        dataset="{pwd}/dataset/DSDP/testset"
    output:
        flag=touch("{pwd}/log/{mode}/DSDP/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/DSDP/script"):
            os.makedirs(f"{pwd}/result/{mode}/DSDP/script")
        script_path=f"{pwd}/result/{mode}/DSDP/script/test.sh"
        make_sbatch(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,gpus=1,software="DSDP",pwd=pwd,script_path=script_path,conda_path=conda_path,dsdp_env=dsdp_env)
        job_status = run_slurm_job(f"sbatch {script_path}")
        if not job_status:
            Path(f"{pwd}/log/{mode}/DSDP/run.err").touch()
