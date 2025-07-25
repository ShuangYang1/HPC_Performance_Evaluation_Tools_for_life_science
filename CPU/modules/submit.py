import subprocess
import time

from filelock import FileLock
from modules.SingleNodeMode.make_sbatch import make_multitasks


# 提交 Slurm 作业并返回作业ID
def submit_sbatch_job(sbatch_command):
    result = subprocess.run(sbatch_command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.split()[-1]
        return job_id
    else:
        raise Exception("Slurm job submission failed")


# 检查作业是否完成
def check_job_status(job_id):
    result = subprocess.run(
        f"sacct -j {job_id} --format=State,ExitCode -n -X",
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Exception(f"Failed to check job status for job {job_id}")
    status_info = result.stdout.strip().split("\n")
    state_total, exit_code_total = [], []
    if status_info:
        for status in status_info:
            if not status.strip():
                continue
            parts = status.split()
            if len(parts) < 2:
                print(f"Warning: Unexpected line format: {status}")
                continue
            state, exit_code = parts[:2]
            state_total.append(state)
            exit_code_total.append(exit_code)
    if "RUNNING" in state_total:
        return "RUNNING", None
    elif "PENDING" in state_total:
        return "PENDING", None
    elif "FAILED" in state_total:
        return "FAILED", exit_code_total[state_total.index("FAILED")]
    elif list(set(state_total)) == ["COMPLETED"]:
        return "COMPLETED", None
    return None, None


# 等待作业完成
def wait_for_job_to_complete(job_id):
    while True:
        state, exit_code = check_job_status(job_id)
        if state in ["COMPLETED", "FAILED"]:
            if state == "FAILED":
                print(f"Job {job_id} failed with exit code {exit_code}.")
                return False
            else:
                print(f"Job {job_id} completed successfully.")
                return True
        else:
            # print(f"Job {job_id} is still running, waiting...")
            time.sleep(30)


# 提交 Slurm 作业并等待完成
def run_slurm_job(sbatch_command):
    job_id = submit_sbatch_job(sbatch_command)
    print(f"Job {job_id} submitted successfully.")
    job_status = wait_for_job_to_complete(job_id)
    return job_status


def get_least_loaded_node(node_list):
    cmd = "squeue -r --format '%F,%n' -h | sort | uniq"
    try:
        output = subprocess.check_output(cmd, shell=True, text=True)
    except subprocess.CalledProcessError as e:
        print("执行 squeue 出错：", e)
        return None, []
    node_jobs = {node: [] for node in node_list}
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != 2:
            continue
        job_id = parts[0].strip()
        req_node = parts[1].strip()
        if req_node in node_jobs:
            node_jobs[req_node].append(job_id)
    selected_node = None
    min_count = None
    for node in node_list:
        count = len(node_jobs[node])
        if min_count is None or count < min_count:
            min_count = count
            selected_node = node
    return selected_node, node_jobs[selected_node]


def submit_singlenode_multitasks(
    partition,
    nodelist,
    cpus_per_task,
    software,
    pwd,
    parallel,
    multitasks_count,
    multitasks_script_path,
):
    lock_file = "/tmp/submit_multitasks.lock"  # 锁文件路径
    lock = FileLock(lock_file, timeout=20)
    with lock:
        target_node, node_jobs = get_least_loaded_node(nodelist)
        make_multitasks(
            partition=partition,
            nodes=1,
            ntasks=1,
            cpus_per_task=cpus_per_task,
            software=software,
            pwd=pwd,
            nodelist=target_node,
            parallel=parallel,
            multitasks_count=multitasks_count,
            script_path=multitasks_script_path,
        )
        if len(node_jobs) == 0:
            job_id = submit_sbatch_job(f"sbatch {multitasks_script_path}")
            print("sbatch done")
        else:
            job_id = submit_sbatch_job(
                f"sbatch --dependency=afterany:{','.join(node_jobs)} {multitasks_script_path}"
            )
            print("sbatch -d done")
        print(f"Job {job_id} submitted successfully.")
        return target_node, job_id
