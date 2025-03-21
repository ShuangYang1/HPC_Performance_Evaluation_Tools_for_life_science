mode="ClusterMode"
configfile: f"config/{mode}/config.yaml"

softwares = []
with open("SoftwaresWeight.txt","r") as infile:
    for line in infile:
        if line.strip().split()[1]!="0":
            softwares.append(line.strip().split()[0])
import os
from modules.ClusterMode.make_sbatch import make_singletask,make_multitasks
from modules.ClusterMode.make_report import generate_report
from modules.submit import run_slurm_job
pwd=os.getcwd()
gpus=config.get("gpus_per_node")
cpus=config.get("cpus_per_node")
partition=config.get("partition")
repeat=config.get("repeat")
nodes=config.get("nodelist")
nodelist=nodes.split(",")
multitasks_count=config.get("multitasks_count",50*len(nodelist))
sponge_gpus_per_task=config.get("sponge_gpus_per_task",1)
gromacs_gpus_per_task=config.get("gromacs_gpus_per_task",1)
amber_gpus_per_task=config.get("amber_gpus_per_task",1)
dsdp_gpus_per_task=config.get("dsdp_gpus_per_task",1)
node_exporter_port=config.get("node_exporter_port")
nvidia_gpu_exporter_port=config.get("gpu_exporter_port")
server_ip=config.get("server_ip")
server_port=config.get("server_port")
conda_path=config.get("conda_path",None)
dsdp_env=config.get("dsdp_env",None)

sponge_singlenode_score=

rule all:
    input:
        expand("{pwd}/log/{mode}/{software}/run.done",mode=mode,software=softwares,pwd=pwd)
    run:
        generate_report(softwares,multitasks_count,gpus,cpus,node_exporter_port,nvidia_gpu_exporter_port,server_ip,server_port,partition,nodelist,singlenode_scores)

rule sponge:
    input:
        mdin="{pwd}/dataset/SPONGE/data/sponge_nvt.in",
        parm7="{pwd}/dataset/SPONGE/data/Q.parm7",
        rst7="{pwd}/dataset/SPONGE/data/Q.rst7"
    output:
        flag=touch("{pwd}/log/{mode}/SPONGE/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/SPONGE/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/SPONGE_multitasks/{count}")
            os.system(f"cp {pwd}/dataset/SPONGE/data/* {pwd}/dataset/SPONGE_multitasks/{count}/")
        multitasks_script_path=f"{script_path}/multitasks.sh"
        cpus_per_task=cpus//gpus
        make_multitasks(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus_per_task,gpus=1,software="SPONGE",pwd=pwd,nodelist=nodes,multitasks_count=multitasks_count,script_path=multitasks_script_path)
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
        for i in range(1,len(nodelist)+1):
            cpus_per_task=cpus//gpus
            ntasks=i*gpus
            singletask_script_path=f"{script_path}/singletask_{i}_{ntasks}.sh"
            make_singletask(partition=partition,nodes=i,ntasks=ntasks,cpus_per_task=cpus_per_task,gpus=gpus,software="GROMACS",pwd=pwd,nodelist=",".join(nodelist[:i]),script_path=singletask_script_path)
            run_slurm_job(f"sbatch {singletask_script_path}")
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/GROMACS_multitasks/{count}")
            os.system(f"cp -r {pwd}/dataset/GROMACS/61k-atoms {pwd}/dataset/GROMACS_multitasks/{count}/")
        cpus_per_task=cpus//gpus
        multitasks_script_path=f"{script_path}/multitasks.sh"
        make_multitasks(partition=partition,nodes=1,ntasks=gromacs_gpus_per_task,cpus_per_task=cpus_per_task,gpus=gromacs_gpus_per_task,software="GROMACS",pwd=pwd,nodelist=nodes,multitasks_count=multitasks_count,script_path=multitasks_script_path)
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
        for i in range(1,len(nodelist)+1):
            cpus_per_task=cpus//gpus
            ntasks=i*gpus
            singletask_script_path=f"{script_path}/singletask_{i}_{ntasks}.sh"
            make_singletask(partition=partition,nodes=i,ntasks=ntasks,cpus_per_task=cpus_per_task,gpus=gpus,software="AMBER",pwd=pwd,nodelist=",".join(nodelist[:i]),script_path=singletask_script_path)
            run_slurm_job(f"sbatch {singletask_script_path}")
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/AMBER_multitasks/{count}")
            os.system(f"cp -r {pwd}/dataset/AMBER/61k-atoms {pwd}/dataset/AMBER_multitasks/{count}/")
        cpus_per_task=cpus//gpus
        multitasks_script_path=f"{script_path}/multitasks.sh"
        make_multitasks(partition=partition,nodes=1,ntasks=amber_gpus_per_task,cpus_per_task=cpus_per_task,gpus=amber_gpus_per_task,software="AMBER",pwd=pwd,nodelist=nodes,multitasks_count=multitasks_count,script_path=multitasks_script_path)
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
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/DSDP_multitasks/{count}")
            os.system(f"cp -r {pwd}/dataset/DSDP/DSDP_dataset {pwd}/dataset/DSDP_multitasks/{count}/")
        multitasks_script_path=f"{script_path}/multitasks.sh"
        cpus_per_task=cpus//gpus
        make_multitasks(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus_per_task,gpus=1,software="DSDP",pwd=pwd,nodelist=nodes,multitasks_count=multitasks_count,script_path=multitasks_script_path,conda_path=conda_path,dsdp_env=dsdp_env)
        run_slurm_job(f"sbatch {multitasks_script_path}")