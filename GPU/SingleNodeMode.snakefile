mode="SingleNodeMode"
configfile: f"config/{mode}/config.yaml"

softwares = []
with open("SoftwaresWeight.txt","r") as infile:
    for line in infile:
        if line.strip().split()[1]!="0":
            softwares.append(line.strip().split()[0])
import os
from modules.SingleNodeMode.make_sbatch import make_singletask_repeat,make_multitasks
from modules.SingleNodeMode.make_report import generate_report
from modules.submit import run_slurm_job
pwd=os.getcwd()
gpus=config.get("gpus")
cpus=config.get("cpus")
partition=config.get("partition")
repeat=config.get("repeat")
multitasks_count=config.get("multitasks_count")
node=config.get("node")
node_exporter_port=config.get("node_exporter_port")
nvidia_gpu_exporter_port=config.get("gpu_exporter_port")
server_ip=config.get("server_ip")
server_port=config.get("server_port")
conda_path=config.get("conda_path",None)
dsdp_env=config.get("dsdp_env",None)

rule all:
    input:
        expand("{pwd}/log/{mode}/{software}/run.done",mode=mode,software=softwares,pwd=pwd)
    run:
        generate_report(softwares,multitasks_count,repeat,gpus,cpus,node_exporter_port,nvidia_gpu_exporter_port,server_ip,server_port,partition,node)

rule sponge:
    input:
        mdin="{pwd}/dataset/SPONGE/sponge_nvt.in",
        parm7="{pwd}/dataset/SPONGE/Q.parm7",
        rst7="{pwd}/dataset/SPONGE/Q.rst7"
    output:
        flag=touch("{pwd}/log/{mode}/SPONGE/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/SPONGE/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        singletask_repeat_script_path=f"{script_path}/singletask_repeat.sh"
        make_singletask_repeat(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,gpus=1,software="SPONGE",pwd=pwd,nodelist=node,repeat=repeat,script_path=singletask_repeat_script_path)
        run_slurm_job(f"sbatch {singletask_repeat_script_path}")
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/SPONGE_multitasks/{count}")
            os.system(f"cp {pwd}/dataset/SPONGE/* {pwd}/dataset/SPONGE_multitasks/{count}/")
        multitasks_script_path=f"{script_path}/multitasks.sh"
        cpus_per_task=cpus//gpus
        make_multitasks(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus_per_task,gpus=1,software="SPONGE",pwd=pwd,nodelist=node,parallel=gpus,multitasks_count=multitasks_count,script_path=multitasks_script_path)
        run_slurm_job(f"sbatch {multitasks_script_path}")

rule gromacs:
    input:
        start_flag="{pwd}/log/{mode}/SPONGE/run.done",
        tpr="{pwd}/dataset/GROMACS/61k-atoms/benchmark.tpr"
    output:
        flag=touch("{pwd}/log/{mode}/GROMACS/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/GROMACS/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for i in range(1,gpus+1):
            if cpus%i==0:
                cpus_per_task=cpus//i
                singletask_repeat_script_path=f"{script_path}/singletask_repeat_{i}_{cpus_per_task}.sh"
                make_singletask_repeat(partition=partition,nodes=1,ntasks=i,cpus_per_task=cpus_per_task,gpus=i,software="GROMACS",pwd=pwd,nodelist=node,repeat=repeat,script_path=singletask_repeat_script_path)
                run_slurm_job(f"sbatch {singletask_repeat_script_path}")
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/GROMACS_multitasks/{count}")
            os.system(f"cp -r {pwd}/dataset/GROMACS/61k-atoms {pwd}/dataset/GROMACS_multitasks/{count}/")
        for i in range(1,gpus+1):
            if gpus%i==0:
                parallel=gpus//i
                cpus_per_task=cpus//gpus
                multitasks_script_path=f"{script_path}/multitasks_{i}_{cpus_per_task}_{parallel}.sh"
                make_multitasks(partition=partition,nodes=1,ntasks=i,cpus_per_task=cpus_per_task,gpus=i,software="GROMACS",pwd=pwd,nodelist=node,parallel=parallel,multitasks_count=multitasks_count,script_path=multitasks_script_path)
                run_slurm_job(f"sbatch {multitasks_script_path}")


rule amber:
    input:
        start_flag="{pwd}/log/{mode}/GROMACS/run.done",
        mdin="{pwd}/dataset/AMBER/61k-atoms/benchmark.in",
        top="{pwd}/dataset/AMBER/61k-atoms/benchmark.top",
        rst="{pwd}/dataset/AMBER/61k-atoms/benchmark.rst"
    output:
        flag=touch("{pwd}/log/{mode}/AMBER/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/AMBER/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for i in range(1,gpus+1):
            cpus_per_task=cpus//i
            singletask_repeat_script_path=f"{script_path}/singletask_repeat_{i}_{cpus_per_task}.sh"
            make_singletask_repeat(partition=partition,nodes=1,ntasks=i,cpus_per_task=cpus_per_task,gpus=i,software="AMBER",pwd=pwd,nodelist=node,repeat=repeat,script_path=singletask_repeat_script_path)
            run_slurm_job(f"sbatch {singletask_repeat_script_path}")
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/AMBER_multitasks/{count}")
            os.system(f"cp -r {pwd}/dataset/AMBER/61k-atoms {pwd}/dataset/AMBER_multitasks/{count}/")
        for i in range(1,gpus+1):
            if gpus%i==0:
                parallel=gpus//i
                cpus_per_task=cpus//gpus
                multitasks_script_path=f"{script_path}/multitasks_{i}_{cpus_per_task}_{parallel}.sh"
                make_multitasks(partition=partition,nodes=1,ntasks=i,cpus_per_task=cpus_per_task,gpus=i,software="AMBER",pwd=pwd,nodelist=node,parallel=parallel,multitasks_count=multitasks_count,script_path=multitasks_script_path)
                run_slurm_job(f"sbatch {multitasks_script_path}")

rule dsdp:
    input:
        start_flag="{pwd}/log/{mode}/AMBER/run.done",
        dataset="{pwd}/dataset/DSDP/DSDP_dataset"
    output:
        flag=touch("{pwd}/log/{mode}/DSDP/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/DSDP/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        singletask_repeat_script_path=f"{script_path}/singletask_repeat.sh"
        make_singletask_repeat(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,gpus=1,software="DSDP",pwd=pwd,nodelist=node,repeat=repeat,script_path=singletask_repeat_script_path,conda_path=conda_path,dsdp_env=dsdp_env)
        run_slurm_job(f"sbatch {singletask_repeat_script_path}")
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/DSDP_multitasks/{count}")
            os.system(f"cp -r {pwd}/dataset/DSDP/DSDP_dataset {pwd}/dataset/DSDP_multitasks/{count}/")
        multitasks_script_path=f"{script_path}/multitasks.sh"
        cpus_per_task=cpus//gpus
        make_multitasks(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus_per_task,gpus=1,software="DSDP",pwd=pwd,nodelist=node,parallel=gpus,multitasks_count=multitasks_count,script_path=multitasks_script_path,conda_path=conda_path,dsdp_env=dsdp_env)
        run_slurm_job(f"sbatch {multitasks_script_path}")