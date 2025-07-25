import os


def make_sbatch(
    partition,
    nodes,
    ntasks,
    cpus_per_task,
    gpus,
    software,
    pwd,
    script_path,
    conda_path=None,
    dsdp_env=None,
):
    logdir = f"{pwd}/log/FastMode/{software}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    output_path = f"{logdir}/test.out"
    error_path = f"{logdir}/test.err"
    log_path = f"{logdir}/test.log"
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
        f.write("#SBATCH --cpu-bind=cores\n")
        f.write("#SBATCH --mem-bind=local\n")
        f.write("\n")
        f.write(f"mkdir -p {resultdir}\n")
        f.write(f"cd {resultdir}\n")
        if software == "AMBER":
            mdin = f"{pwd}/dataset/AMBER/61k-atoms/benchmark.in"
            top = f"{pwd}/dataset/AMBER/61k-atoms/benchmark.top"
            rst = f"{pwd}/dataset/AMBER/61k-atoms/benchmark.rst"
            f.write(
                f"mpirun -np {gpus} pmemd.cuda.MPI -O -o ./mdout -i {mdin} -p {top} -c {rst} >& {log_path}\n"
            )
        elif software == "GROMACS":
            tpr = f"{pwd}/dataset/GROMACS/61k-atoms/benchmark.tpr"
            f.write(
                f"gmx mdrun -ntmpi {gpus} -ntomp {cpus_per_task} -s {tpr} -noconfout -pin on -nsteps 50000 -nb gpu -bonded gpu -pme gpu -npme 1 -o ./traj.trr -cpo ./state.cpt -e ./ener.edr -g ./md.log >& {log_path}\n"
            )
        elif software == "SPONGE":
            mdin = f"{pwd}/dataset/SPONGE/data/sponge_nvt.in"
            parm7 = f"{pwd}/dataset/SPONGE/data/Q.parm7"
            rst7 = f"{pwd}/dataset/SPONGE/data/Q.rst7"
            f.write(
                f"SPONGE -mdin {mdin} -amber_parm7 {parm7} -amber_rst7 {rst7} >& {log_path}\n"
            )
        elif software == "DSDP":
            dataset = f"{pwd}/dataset/DSDP/DSDP_dataset"
            if conda_path is not None and dsdp_env is not None:
                f.write(f"source {conda_path}/bin/activate {dsdp_env}\n")
            f.write(
                f"DSDP blind -i {dataset} -o ./test_output --exhaustiveness 384 --search_depth 40 --top_n 1 >& {log_path}\n"
            )
    return True
