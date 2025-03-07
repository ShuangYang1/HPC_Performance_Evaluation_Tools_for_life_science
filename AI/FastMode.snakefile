mode="FastMode"
configfile: f"config/{mode}/config.yaml"

softwares = []
with open("SoftwaresWeight.txt","r") as infile:
    for line in infile:
        if line.strip().split()[1]!="0":
            softwares.append(line.strip().split()[0])
import os
from modules.FastMode.make_sbatch import make_sbatch
from modules.submit import run_slurm_job
pwd=os.getcwd()
gpus=config.get("gpus")
cpus=config.get("cpus")
partition=config.get("partition")
mindsponge_cuda_home=config.get("mindsponge_cuda_home")
alphafold_cuda_home=config.get("alphafold_cuda_home")
mindsponge_env=config.get("mindsponge_env")
alphafold_env=config.get("alphafold_env")
alphafold_path=config.get("alphafold_path")
alphafold_param_path=config.get("alphafold_param_path")
alphafold_db_path=config.get("alphafold_db_path")
conda_path=config.get("conda_path")

rule all:
    input:
        expand("{pwd}/log/{mode}/{software}/run.done",mode=mode,software=softwares,pwd=pwd)
    shell:"echo 'Your softwares are fully prepared.'"

rule mindsponge_test:
    input:
        pdb="{pwd}/dataset/MindSPONGE/case2.pdb"
    output:
        flag=touch("{pwd}/log/{mode}/MindSPONGE/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/MindSPONGE/script"):
            os.makedirs(f"{pwd}/result/{mode}/MindSPONGE/script")
        script_path=f"{pwd}/result/{mode}/MindSPONGE/script/test.sh"
        make_sbatch(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,gpus=1,software="MindSPONGE",cuda_home=mindsponge_cuda_home,conda_path=conda_path,env=mindsponge_env,pwd=pwd,script_path=script_path)
        run_slurm_job(f"sbatch {script_path}")

rule alphafold_test:
    input:
        seq="{pwd}/dataset/Alphafold3/2PV7_inference_only.json"
    output:
        flag=touch("{pwd}/log/{mode}/Alphafold3/run.done")
    run:
        if not os.path.exists(f"{pwd}/result/{mode}/Alphafold3/script"):
            os.makedirs(f"{pwd}/result/{mode}/Alphafold3/script")
        script_path=f"{pwd}/result/{mode}/Alphafold3/script/test.sh"
        make_sbatch(partition=partition,nodes=1,ntasks=1,cpus_per_task=cpus,gpus=1,software="Alphafold3",cuda_home=alphafold_cuda_home,conda_path=conda_path,env=alphafold_env,pwd=pwd,script_path=script_path,alphafold_path=alphafold_path,alphafold_param_path=alphafold_param_path,alphafold_db_path=alphafold_db_path)
        run_slurm_job(f"sbatch {script_path}")