import os


def make_sbatch(
    partition,
    nodes,
    ntasks,
    cpus_per_task,
    gpus,
    software,
    cuda_home,
    conda_path,
    env,
    pwd,
    script_path,
    alphafold_path=None,
    alphafold_param_path=None,
    alphafold_db_path=None,
):
    if not os.path.exists(f"{pwd}/log/FastMode/{software}"):
        os.makedirs(f"{pwd}/log/FastMode/{software}")
    output_path = f"{pwd}/log/FastMode/{software}/test.out"
    error_path = f"{pwd}/log/FastMode/{software}/test.err"
    log_path = f"{pwd}/log/FastMode/{software}/test.log"
    resultdir = f"{pwd}/result/FastMode/{software}/"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --partition={}\n".format(partition))
        f.write("#SBATCH --nodes={}\n".format(nodes))
        f.write("#SBATCH --ntasks={}\n".format(ntasks))
        f.write("#SBATCH --cpus-per-task={}\n".format(cpus_per_task))
        f.write("#SBATCH --gres=gpu:{}\n".format(gpus))
        f.write("#SBATCH --job-name={}\n".format(software))
        f.write("#SBATCH --output={}\n".format(output_path))
        f.write("#SBATCH --error={}\n".format(error_path))
        f.write("#SBATCH --time=7-00:00:00\n")
        f.write("\n")
        f.write(f"mkdir -p {resultdir}\n")
        f.write(f"cd {resultdir}\n")
        f.write(f"source {conda_path}/bin/activate {env}\n")
        f.write("export CUDA_HOME={}\n".format(cuda_home))
        f.write("export PATH=$CUDA_HOME/bin:$PATH\n")
        f.write("export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH\n")
        if software == "MindSPONGE":
            pdb = f"{pwd}/dataset/MindSPONGE/case2.pdb"
            md = f"{pwd}/modules/md.py"
            f.write(f"python {md} {pdb} test.h5md > {log_path}\n")
        elif software == "Alphafold3":
            seq = f"{pwd}/dataset/Alphafold3/2PV7_inference_only.json"
            f.write(
                f"python {alphafold_path}/run_alphafold.py --gpu_device=0 --norun_data_pipeline --json_path={seq} --model_dir={alphafold_param_path} --db_dir={alphafold_db_path} --output_dir=./ &> {log_path}\n"
            )
    return True
