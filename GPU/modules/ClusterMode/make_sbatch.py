import os


def make_singletask(
    partition,
    nodes,
    ntasks,
    cpus_per_task,
    gpus,
    software,
    pwd,
    nodelist,
    script_path,
):
    logdir = f"{pwd}/log/ClusterMode/{software}/singletask"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    output_path = f"{logdir}/{nodes}_{ntasks}.out"
    error_path = f"{logdir}/{nodes}_{ntasks}_%a.err"
    log_path = f"{logdir}/{nodes}_{ntasks}.log"
    start = f"{logdir}/{nodes}_{ntasks}.start.timestamp"
    end = f"{logdir}/{nodes}_{ntasks}.end.timestamp"
    resultdir = f"{pwd}/result/ClusterMode/{software}/singletask/"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --partition={}\n".format(partition))
        f.write("#SBATCH --nodes={}\n".format(nodes))
        f.write("#SBATCH --ntasks-per-node={}\n".format(gpus))
        f.write("#SBATCH --cpus-per-task={}\n".format(cpus_per_task))
        f.write("#SBATCH --gres=gpu:{}\n".format(gpus))
        f.write("#SBATCH --job-name={}\n".format(software))
        f.write("#SBATCH --output={}\n".format(output_path))
        f.write("#SBATCH --error={}\n".format(error_path))
        f.write(f"#SBATCH --nodelist={nodelist}\n")
        f.write("#SBATCH --time=7-00:00:00\n")
        f.write("\n")
        f.write(f'WORKDIR="{resultdir}/{nodes}_{ntasks}"\n')
        f.write("mkdir -p $WORKDIR\n")
        f.write("cd $WORKDIR\n")
        f.write(f"date +%s > {start}\n")
        if software == "AMBER":
            mdin = f"{pwd}/dataset/AMBER/61k-atoms/benchmark.in"
            top = f"{pwd}/dataset/AMBER/61k-atoms/benchmark.top"
            rst = f"{pwd}/dataset/AMBER/61k-atoms/benchmark.rst"
            f.write(
                f"mpirun -np {ntasks} pmemd.cuda.MPI -O -o ./mdout -i {mdin} -p {top} -c {rst} >& {log_path}\n"
            )
        elif software == "GROMACS":
            tpr = f"{pwd}/dataset/GROMACS/61k-atoms/benchmark.tpr"
            f.write(
                f"gmx mdrun -ntmpi {ntasks} -ntomp {cpus_per_task} -s {tpr} -noconfout -pin on -nsteps 50000 -nb gpu -bonded gpu -pme gpu -npme 1 -o ./traj.trr -cpo ./state.cpt -e ./ener.edr -g ./md.log >& {log_path}\n"
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
    pwd,
    nodelist,
    multitasks_count,
    script_path,
    conda_path=None,
    dsdp_env=None,
):
    logdir = f"{pwd}/log/ClusterMode/{software}/multitasks/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    output_path = f"{logdir}/%a.out"
    error_path = f"{logdir}/%a.err"
    log_path = f"{logdir}/${{SLURM_ARRAY_TASK_ID}}.log"
    start = f"{logdir}/${{SLURM_ARRAY_TASK_ID}}.start.timestamp"
    end = f"{logdir}/${{SLURM_ARRAY_TASK_ID}}.end.timestamp"
    resultdir = f"{pwd}/result/ClusterMode/{software}/multitasks/"
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
        f.write("#SBATCH --cpu-bind=cores\n")
        f.write("#SBATCH --mem-bind=local\n")
        f.write("\n")
        f.write(f'WORKDIR="{resultdir}/${{SLURM_ARRAY_TASK_ID}}"\n')
        f.write("mkdir -p $WORKDIR\n")
        f.write("cd $WORKDIR\n")
        f.write(f"date +%s > {start}\n")
        if software == "AMBER":
            mdin = f"{pwd}/dataset/AMBER_multitasks/${{SLURM_ARRAY_TASK_ID}}/61k-atoms/benchmark.in"
            top = f"{pwd}/dataset/AMBER_multitasks/${{SLURM_ARRAY_TASK_ID}}/61k-atoms/benchmark.top"
            rst = f"{pwd}/dataset/AMBER_multitasks/${{SLURM_ARRAY_TASK_ID}}/61k-atoms/benchmark.rst"
            if gpus == 1:
                f.write(
                    f"pmemd.cuda -O -o ./mdout -i {mdin} -p {top} -c {rst} >& {log_path}\n"
                )
            else:
                f.write(
                    f"mpirun -np {gpus} pmemd.cuda.MPI -O -o ./mdout -i {mdin} -p {top} -c {rst} >& {log_path}\n"
                )
        elif software == "GROMACS":
            tpr = f"{pwd}/dataset/GROMACS_multitasks/${{SLURM_ARRAY_TASK_ID}}/61k-atoms/benchmark.tpr"
            if gpus == 1:
                f.write(
                    f"gmx mdrun -ntmpi {gpus} -ntomp {cpus_per_task} -s {tpr} -noconfout -pin on -nsteps 50000 -nb gpu -bonded gpu -pme gpu -o ./traj.trr -cpo ./state.cpt -e ./ener.edr -g ./md.log >& {log_path}\n"
                )
            else:
                f.write(
                    f"gmx mdrun -ntmpi {gpus} -ntomp {cpus_per_task} -s {tpr} -noconfout -pin on -nsteps 50000 -nb gpu -bonded gpu -pme gpu -npme 1 -o ./traj.trr -cpo ./state.cpt -e ./ener.edr -g ./md.log >& {log_path}\n"
                )
        elif software == "SPONGE":
            mdin = f"{pwd}/dataset/SPONGE_multitasks/${{SLURM_ARRAY_TASK_ID}}/sponge_nvt.in"
            parm7 = f"{pwd}/dataset/SPONGE_multitasks/${{SLURM_ARRAY_TASK_ID}}/Q.parm7"
            rst7 = f"{pwd}/dataset/SPONGE_multitasks/${{SLURM_ARRAY_TASK_ID}}/Q.rst7"
            f.write(
                f"SPONGE -mdin {mdin} -amber_parm7 {parm7} -amber_rst7 {rst7} >& {log_path}\n"
            )
        elif software == "DSDP":
            dataset = (
                f"{pwd}/dataset/DSDP_multitasks/${{SLURM_ARRAY_TASK_ID}}/DSDP_dataset"
            )
            if conda_path is not None and dsdp_env is not None:
                f.write(f"source {conda_path}/bin/activate {dsdp_env}\n")
            f.write(
                f"DSDP blind -i {dataset} -o ./test_output --exhaustiveness 384 --search_depth 40 --top_n 1 >& {log_path}\n"
            )
        f.write(f"date +%s > {end}\n")
    return True
