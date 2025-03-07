import os


def make_singletask_repeat(
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
    nodelist,
    repeat,
    script_path,
    alphafold_path=None,
    alphafold_param_path=None,
    alphafold_db_path=None,
):
    logdir = f"{pwd}/log/SingleNodeMode/{software}/singletask"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    output_path = f"{logdir}/{gpus}_{cpus_per_task}_%a.out"
    error_path = f"{logdir}/{gpus}_{cpus_per_task}_%a.err"
    log_path = f"{logdir}/{gpus}_{cpus_per_task}_${{SLURM_ARRAY_TASK_ID}}.log"
    start = f"{logdir}/{gpus}_{cpus_per_task}_${{SLURM_ARRAY_TASK_ID}}.start.timestamp"
    end = f"{logdir}/{gpus}_{cpus_per_task}_${{SLURM_ARRAY_TASK_ID}}.end.timestamp"
    resultdir = f"{pwd}/result/SingleNodeMode/{software}/singletask/"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --partition={}\n".format(partition))
        f.write("#SBATCH --nodes={}\n".format(nodes))
        f.write("#SBATCH --ntasks={}\n".format(ntasks))
        f.write("#SBATCH --cpus-per-task={}\n".format(cpus_per_task))
        f.write("#SBATCH --gres=gpu:{}\n".format(gpus))
        f.write("#SBATCH --job-name={}_repeat\n".format(software))
        f.write("#SBATCH --output={}\n".format(output_path))
        f.write("#SBATCH --error={}\n".format(error_path))
        f.write(f"#SBATCH --array=1-{repeat}\n")
        f.write(f"#SBATCH --nodelist={nodelist}\n")
        f.write("#SBATCH --time=7-00:00:00\n")
        f.write("\n")
        f.write(
            f'WORKDIR="{resultdir}/{gpus}_{cpus_per_task}_${{SLURM_ARRAY_TASK_ID}}"\n'
        )
        f.write("mkdir -p $WORKDIR\n")
        f.write("cd $WORKDIR\n")
        f.write(f"source {conda_path}/bin/activate {env}\n")
        f.write("export CUDA_HOME={}\n".format(cuda_home))
        f.write("export PATH=$CUDA_HOME/bin:$PATH\n")
        f.write("export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH\n")
        f.write(f"date +%s > {start}\n")
        if software == "MindSPONGE":
            pdb = f"{pwd}/dataset/MindSPONGE/case2.pdb"
            md = f"{pwd}/modules/md.py"
            f.write(f"python {md} {pdb} test.h5md > {log_path}\n")
        elif software == "Alphafold3":
            seq = f"{pwd}/dataset/Alphafold3/2PV7_inference_only.json"
            f.write(
                f"python {alphafold_path}/run_alphafold.py --gpu_device=0 --norun_data_pipeline --json_path={seq} --model_dir={alphafold_param_path} --db_dir={alphafold_db_path} --output_dir=./ &> {log_path}\n"
            )
        f.write(f"date +%s > {end}\n")
    return True


def make_multitasks(
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
    nodelist,
    parallel,
    multitasks_count,
    script_path,
    alphafold_path=None,
    alphafold_param_path=None,
    alphafold_db_path=None,
):
    logdir = f"{pwd}/log/SingleNodeMode/{software}/multitasks/{gpus}_{cpus_per_task}_{parallel}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    output_path = f"{logdir}/%a.out"
    error_path = f"{logdir}/%a.err"
    log_path = f"{logdir}/${{SLURM_ARRAY_TASK_ID}}.log"
    start = f"{logdir}/${{SLURM_ARRAY_TASK_ID}}.start.timestamp"
    end = f"{logdir}/${{SLURM_ARRAY_TASK_ID}}.end.timestamp"
    resultdir = f"{pwd}/result/SingleNodeMode/{software}/multitasks/"
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
        f.write(f"#SBATCH --array=1-{multitasks_count}\n")
        f.write(f"#SBATCH --nodelist={nodelist}\n")
        f.write("#SBATCH --time=7-00:00:00\n")
        f.write("\n")
        f.write(
            f'WORKDIR="{resultdir}/{gpus}_{cpus_per_task}_{parallel}/${{SLURM_ARRAY_TASK_ID}}"\n'
        )
        f.write("mkdir -p $WORKDIR\n")
        f.write("cd $WORKDIR\n")
        f.write(f"source {conda_path}/bin/activate {env}\n")
        f.write("export CUDA_HOME={}\n".format(cuda_home))
        f.write("export PATH=$CUDA_HOME/bin:$PATH\n")
        f.write("export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH\n")
        f.write(f"date +%s > {start}\n")
        if software == "MindSPONGE":
            pdb = f"{pwd}/dataset/MindSPONGE_multitasks/${{SLURM_ARRAY_TASK_ID}}/case2.pdb"
            md = f"{pwd}/modules/md.py"
            f.write(f"python {md} {pdb} test.h5md > {log_path}\n")
        elif software == "Alphafold3":
            seq = f"{pwd}/dataset/Alphafold3_multitasks/${{SLURM_ARRAY_TASK_ID}}/2PV7_inference_only.json"
            f.write(
                f"python {alphafold_path}/run_alphafold.py --gpu_device=0 --norun_data_pipeline --json_path={seq} --model_dir={alphafold_param_path} --db_dir={alphafold_db_path} --output_dir=./ &> {log_path}\n"
            )
        f.write(f"date +%s > {end}\n")
    return True
