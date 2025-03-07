import subprocess
import time


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
