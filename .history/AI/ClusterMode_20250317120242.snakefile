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
gpus=config.get("gpus_per_node")
cpus=config.get("cpus_per_node")
partition=config.get("partition")
repeat=config.get("repeat")
nodes=config.get("nodelist")
nodelist=nodes.split(",")
multitasks_count=config.get("multitasks_count",50*len(nodelist))
mindsponge_gpus_per_task=config.get("mindsponge_gpus_per_task",1)
alphafold_gpus_per_task=config.get("alphafold_gpus_per_task",1)
node_exporter_port=config.get("node_exporter_port")
nvidia_gpu_exporter_port=config.get("gpu_exporter_port")
server_ip=config.get("server_ip")
server_port=config.get("server_port")
mindsponge_cuda_home=config.get("mindsponge_cuda_home")
alphafold_cuda_home=config.get("alphafold_cuda_home")
mindsponge_env=config.get("mindsponge_env")
alphafold_env=config.get("alphafold_env")
alphafold_path=config.get("alphafold_path")
alphafold_param_path=config.get("alphafold_param_path")
alphafold_db_path=config.get("alphafold_db_path")
conda_path=config.get("conda_path")

mindsponge_singlenode_score=config.get("mindsponge_singlenode_score",False)
alphafold_singlenode_score=config.get("alphafold_singlenode_score",False)

singlenode_scores={}


rule all:
    input:
        expand("{pwd}/log/{mode}/{software}/run.done",mode=mode,software=softwares,pwd=pwd)
    run:
        generate_report(softwares,multitasks_count,gpus,cpus,node_exporter_port,nvidia_gpu_exporter_port,server_ip,server_port,partition,nodelist,mindsponge_env,alphafold_path)

rule mindsponge:
    input:
        pdb="{pwd}/dataset/MindSPONGE/case2.pdb"
    output:
        flag=touch("{pwd}/log/{mode}/MindSPONGE/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/MindSPONGE/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/MindSPONGE_multitasks/{count}")
            os.system(f"cp {pwd}/dataset/MindSPONGE/* {pwd}/dataset/MindSPONGE_multitasks/{count}/")
        multitasks_script_path=f"{script_path}/multitasks.sh"
        cpus_per_task=cpus//gpus
        make_multitasks(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus_per_task,gpus=1,software="MindSPONGE",cuda_home=mindsponge_cuda_home,conda_path=conda_path,env=mindsponge_env,pwd=pwd,nodelist=nodes,multitasks_count=multitasks_count,script_path=multitasks_script_path)
        run_slurm_job(f"sbatch {multitasks_script_path}")

rule alphafold:
    input:
        seq="{pwd}/dataset/Alphafold3/2PV7_inference_only.json",
        flag="{pwd}/log/{mode}/MindSPONGE/run.done"
    output:
        flag=touch("{pwd}/log/{mode}/Alphafold3/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/Alphafold3/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/Alphafold3_multitasks/{count}")
            os.system(f"cp {pwd}/dataset/Alphafold3/* {pwd}/dataset/Alphafold3_multitasks/{count}/")
        multitasks_script_path=f"{script_path}/multitasks.sh"
        cpus_per_task=cpus//gpus
        make_multitasks(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus_per_task,gpus=1,software="Alphafold3",cuda_home=alphafold_cuda_home,conda_path=conda_path,env=alphafold_env,pwd=pwd,nodelist=nodes,multitasks_count=multitasks_count,script_path=multitasks_script_path,alphafold_path=alphafold_path,alphafold_param_path=alphafold_param_path,alphafold_db_path=alphafold_db_path)
        run_slurm_job(f"sbatch {multitasks_script_path}")