mode="SingleNodeMode"
configfile: f"config/{mode}/config.yaml"

softwares = []
with open("SoftwaresWeight.txt","r") as infile:
    for line in infile:
        if line.strip().split()[1]!="0":
            softwares.append(line.strip().split()[0])

from modules.SingleNodeMode.cal_singletask_scale import cal_singletask_scale
from modules.SingleNodeMode.cal_multitasks_scale import cal_multitasks_scale,cal_gatk_multitasks_scale
from modules.SingleNodeMode.make_report import multitasks_best_parallelmode,generate_report
import os
from modules.SingleNodeMode.make_sbatch import make_singletask,make_singletask_repeat,make_multitasks,find_best_singletask_threads
from modules.submit import run_slurm_job,submit_sbatch_job,wait_for_job_to_complete,submit_singlenode_multitasks
pwd=os.getcwd()
cpus=config.get("cpus")
mem=config.get("mem_mb")
partition=config.get("partition")
repeat=config.get("repeat")
multitasks_count=config.get("multitasks_count")
node=config.get("node")
nodelist=node.split(",")
node_exporter_port=config.get("node_exporter_port")
server_ip=config.get("server_ip")
server_port=config.get("server_port")
delete=config.get("delete-intermediate-files")


rule all:
    input:
        expand("{pwd}/log/{mode}/{software}/singletask/run.done",mode=mode,software=softwares,pwd=pwd),
        expand("{pwd}/log/{mode}/{software}/multitasks/run.done",mode=mode,software=softwares,pwd=pwd),
        expand("{pwd}/log/{mode}/{software}/singletask/repeat.done",mode=mode,software=softwares,pwd=pwd)
    run:
        generate_report(multitasks_count, repeat, cpus, mem,partition,nodelist,node_exporter_port,server_ip,server_port)

rule bwa:
    input:
        genome="{pwd}/dataset/BWA/hg18/hg18.fa.gz",
        fastq1="{pwd}/dataset/BWA/ERR000589/ERR000589_1.fastq",
        fastq2="{pwd}/dataset/BWA/ERR000589/ERR000589_2.fastq"
    output:
        touch("{pwd}/log/{mode}/BWA/singletask/run.done")
    params:
        software="BWA"
    run:
        threads=cal_singletask_scale(cpus,params.software)
        for cpus_per_task in threads:
            script_path=f"{pwd}/result/{mode}/BWA/script"
            if not os.path.exists(script_path):
                os.makedirs(script_path)
            singletask_script_path=f"{script_path}/{cpus_per_task}.sh"
            make_singletask(partition=partition,nodes=1,ntasks=1,cpus=cpus,cpus_per_task=cpus_per_task,software="BWA",pwd=pwd,nodelist=node,script_path=singletask_script_path)
            run_slurm_job(f"sbatch {singletask_script_path}")
            if delete:
                os.system(f"rm -rf {pwd}/result/SingleNodeMode/{params.software}/singletask/*")
            else:
                pass

rule spades:
    input:
        fastq1="{pwd}/dataset/SPAdes/EAS20_8/s_6_1.fastq",
        fastq2="{pwd}/dataset/SPAdes/EAS20_8/s_6_2.fastq"
    output:
        touch("{pwd}/log/{mode}/SPAdes/singletask/run.done")
    params:
        software="SPAdes"
    run:
        threads=cal_singletask_scale(cpus,params.software)
        for cpus_per_task in threads:
            script_path=f"{pwd}/result/{mode}/SPAdes/script"
            if not os.path.exists(script_path):
                os.makedirs(script_path)
            singletask_script_path=f"{script_path}/{cpus_per_task}.sh"
            make_singletask(partition=partition,nodes=1,ntasks=1,cpus=cpus,cpus_per_task=cpus_per_task,software="SPAdes",pwd=pwd,nodelist=node,script_path=singletask_script_path)
            run_slurm_job(f"sbatch {singletask_script_path}")
            if delete:
                os.system(f"rm -rf {pwd}/result/SingleNodeMode/{params.software}/singletask/*")
            else:
                pass

rule bismark:
    input:
        genome="{pwd}/dataset/Bismark/hg18/",
        fastq="{pwd}/dataset/Bismark/SRR020138.fastq"
    output:
        touch("{pwd}/log/{mode}/Bismark/singletask/run.done")
    params:
        software="Bismark"
    run:
        threads=cal_singletask_scale(cpus//4,params.software)
        for cpus_per_task in threads:
            script_path=f"{pwd}/result/{mode}/Bismark/script"
            if not os.path.exists(script_path):
                os.makedirs(script_path)
            singletask_script_path=f"{script_path}/{cpus_per_task}.sh"
            make_singletask(partition=partition,nodes=1,ntasks=1,cpus=cpus,cpus_per_task=cpus_per_task*4,software="Bismark",pwd=pwd,nodelist=node,script_path=singletask_script_path)
            run_slurm_job(f"sbatch {singletask_script_path}")
            if delete:
                os.system(f"rm -rf {pwd}/result/SingleNodeMode/{params.software}/singletask/*")
            else:
                pass

rule star:
    input:
        genome="{pwd}/dataset/STAR/mm39StarIndex",
        fastq="{pwd}/dataset/STAR/SRR6821753.fastq"
    output:
        touch("{pwd}/log/{mode}/STAR/singletask/run.done")
    params:
        software="STAR"
    run:
        threads=cal_singletask_scale(cpus,params.software)
        for cpus_per_task in threads:
            script_path=f"{pwd}/result/{mode}/STAR/script"
            if not os.path.exists(script_path):
                os.makedirs(script_path)
            singletask_script_path=f"{script_path}/{cpus_per_task}.sh"
            make_singletask(partition=partition,nodes=1,ntasks=1,cpus=cpus,cpus_per_task=cpus_per_task,software="STAR",pwd=pwd,nodelist=node,script_path=singletask_script_path)
            run_slurm_job(f"sbatch {singletask_script_path}")
            if delete:
                os.system(f"rm -rf {pwd}/result/SingleNodeMode/{params.software}/singletask/*")
            else:
                pass

rule cellranger:
    input:
        genome="{pwd}/dataset/Cellranger/refdata-gex-GRCh38-2024-A/",
        fastq="{pwd}/dataset/Cellranger/pbmc_1k_v3_fastqs"
    output:
        touch("{pwd}/log/{mode}/Cellranger/singletask/run.done")
    params:
        software="Cellranger"
    run:
        threads=cal_singletask_scale(cpus,params.software)
        for cpus_per_task in threads:
            script_path=f"{pwd}/result/{mode}/Cellranger/script"
            if not os.path.exists(script_path):
                os.makedirs(script_path)
            singletask_script_path=f"{script_path}/{cpus_per_task}.sh"
            make_singletask(partition=partition,nodes=1,ntasks=1,cpus=cpus,cpus_per_task=cpus_per_task,software="Cellranger",pwd=pwd,nodelist=node,script_path=singletask_script_path)
            run_slurm_job(f"sbatch {singletask_script_path}")
            if delete:
                os.system(f"rm -rf {pwd}/result/SingleNodeMode/{params.software}/singletask/*")
            else:
                pass

rule gatk:
    input:
        genome="{pwd}/dataset/GATK/GRCh38_full_analysis_set_plus_decoy_hla.fa",
        bam="{pwd}/dataset/GATK/NA12878_24RG_small.hg38.bam"
    output:
        touch("{pwd}/log/{mode}/GATK/singletask/run.done")
    params:
        software="GATK"
    run:
        threads=cal_singletask_scale(cpus,params.software)
        for cpus_per_task in threads:
            script_path=f"{pwd}/result/{mode}/GATK/script"
            if not os.path.exists(script_path):
                os.makedirs(script_path)
            singletask_script_path=f"{script_path}/{cpus_per_task}.sh"
            make_singletask(partition=partition,nodes=1,ntasks=1,cpus=cpus,cpus_per_task=cpus_per_task,software="GATK",pwd=pwd,nodelist=node,script_path=singletask_script_path)
            run_slurm_job(f"sbatch {singletask_script_path}")
            if delete:
                os.system(f"rm -rf {pwd}/result/SingleNodeMode/{params.software}/singletask/*")
            else:
                pass

rule bwa_repeat:
    input:
        genome="{pwd}/dataset/BWA/hg18/hg18.fa.gz",
        fastq1="{pwd}/dataset/BWA/ERR000589/ERR000589_1.fastq",
        fastq2="{pwd}/dataset/BWA/ERR000589/ERR000589_2.fastq",
        flag="{pwd}/log/{mode}/BWA/singletask/run.done"
    output:
        touch("{pwd}/log/{mode}/BWA/singletask/repeat.done")
    params:
        software="BWA"
    run:
        script_path=f"{pwd}/result/{mode}/BWA/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        singletask_repeat_script_path=f"{script_path}/singletask_repeat.sh"
        threads=cal_singletask_scale(cpus,params.software)
        best_thread,best_tim=find_best_singletask_threads(params.software, threads)
        make_singletask_repeat(partition=partition,nodes=1,ntasks=1,cpus=cpus,cpus_per_task=best_thread,software="BWA",pwd=pwd,nodelist=node,repeat=repeat,script_path=singletask_repeat_script_path)
        run_slurm_job(f"sbatch {singletask_repeat_script_path}")
        if delete:
            os.system(f"rm -rf {pwd}/result/SingleNodeMode/{params.software}/singletask/*")
        else:
            pass

rule spades_repeat:
    input:
        fastq1="{pwd}/dataset/SPAdes/EAS20_8/s_6_1.fastq",
        fastq2="{pwd}/dataset/SPAdes/EAS20_8/s_6_2.fastq",
        flag="{pwd}/log/{mode}/SPAdes/singletask/run.done"
    output:
        touch("{pwd}/log/{mode}/SPAdes/singletask/repeat.done")
    params:
        software="SPAdes"
    run:
        script_path=f"{pwd}/result/{mode}/SPAdes/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        singletask_repeat_script_path=f"{script_path}/singletask_repeat.sh"
        threads=cal_singletask_scale(cpus,params.software)
        best_thread,best_tim=find_best_singletask_threads(params.software, threads)
        make_singletask_repeat(partition=partition,nodes=1,ntasks=1,cpus=cpus,cpus_per_task=best_thread,software="SPAdes",pwd=pwd,nodelist=node,repeat=repeat,script_path=singletask_repeat_script_path)
        run_slurm_job(f"sbatch {singletask_repeat_script_path}")
        if delete:
            os.system(f"rm -rf {pwd}/result/SingleNodeMode/{params.software}/singletask/*")
        else:
            pass

rule bismark_repeat:
    input:
        genome="{pwd}/dataset/Bismark/hg18/",
        fastq="{pwd}/dataset/Bismark/SRR020138.fastq",
        flag="{pwd}/log/{mode}/Bismark/singletask/run.done"
    output:
        touch("{pwd}/log/{mode}/Bismark/singletask/repeat.done")
    params:
        software="Bismark"
    run:
        script_path=f"{pwd}/result/{mode}/Bismark/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        singletask_repeat_script_path=f"{script_path}/singletask_repeat.sh"
        threads=cal_singletask_scale(cpus,params.software)
        best_thread,best_tim=find_best_singletask_threads(params.software, threads)
        make_singletask_repeat(partition=partition,nodes=1,ntasks=1,cpus=cpus,cpus_per_task=best_thread,software="Bismark",pwd=pwd,nodelist=node,repeat=repeat,script_path=singletask_repeat_script_path)
        run_slurm_job(f"sbatch {singletask_repeat_script_path}")
        if delete:
            os.system(f"rm -rf {pwd}/result/SingleNodeMode/{params.software}/singletask/*")
        else:
            pass

rule star_repeat:
    input:
        genome="{pwd}/dataset/STAR/mm39StarIndex",
        fastq="{pwd}/dataset/STAR/SRR6821753.fastq",
        flag="{pwd}/log/{mode}/STAR/singletask/run.done"
    output:
        touch("{pwd}/log/{mode}/STAR/singletask/repeat.done")
    params:
        software="STAR"
    run:
        threads=sorted(cal_singletask_scale(cpus,params.software), reverse=True)
        thread=cpus
        for i in threads:
            log_file=f"log/{mode}/{params.software}/singletask/{i}.log"
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()
                    if "ERROR" not in content:
                        thread=i
                        break
                    else:
                        continue
        script_path=f"{pwd}/result/{mode}/STAR/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        singletask_repeat_script_path=f"{script_path}/singletask_repeat.sh"
        make_singletask_repeat(partition=partition,nodes=1,ntasks=1,cpus=cpus,cpus_per_task=thread,software="STAR",pwd=pwd,nodelist=node,repeat=repeat,script_path=singletask_repeat_script_path)
        run_slurm_job(f"sbatch {singletask_repeat_script_path}")
        if delete:
            os.system(f"rm -rf {pwd}/result/SingleNodeMode/{params.software}/singletask/*")
        else:
            pass

rule cellranger_repeat:
    input:
        genome="{pwd}/dataset/Cellranger/refdata-gex-GRCh38-2024-A/",
        fastq="{pwd}/dataset/Cellranger/pbmc_1k_v3_fastqs",
        flag="{pwd}/log/{mode}/Cellranger/singletask/run.done"
    output:
        touch("{pwd}/log/{mode}/Cellranger/singletask/repeat.done")
    params:
        software="Cellranger"
    run:
        script_path=f"{pwd}/result/{mode}/Cellranger/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        singletask_repeat_script_path=f"{script_path}/singletask_repeat.sh"
        make_singletask_repeat(partition=partition,nodes=1,ntasks=1,cpus=cpus,cpus_per_task=cpus,software="Cellranger",pwd=pwd,nodelist=node,repeat=repeat,script_path=singletask_repeat_script_path)
        run_slurm_job(f"sbatch {singletask_repeat_script_path}")
        if delete:
            os.system(f"rm -rf {pwd}/result/SingleNodeMode/{params.software}/singletask/*")
        else:
            pass

rule gatk_repeat:
    input:
        genome="{pwd}/dataset/GATK/GRCh38_full_analysis_set_plus_decoy_hla.fa",
        bam="{pwd}/dataset/GATK/NA12878_24RG_small.hg38.bam",
        flag="{pwd}/log/{mode}/GATK/singletask/run.done"
    output:
        touch("{pwd}/log/{mode}/GATK/singletask/repeat.done")
    params:
        software="GATK"
    run:
        script_path=f"{pwd}/result/{mode}/GATK/script"
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        singletask_repeat_script_path=f"{script_path}/singletask_repeat.sh"
        make_singletask_repeat(partition=partition,nodes=1,ntasks=1,cpus=cpus,cpus_per_task=cpus,software="GATK",pwd=pwd,nodelist=node,repeat=repeat,script_path=singletask_repeat_script_path)
        run_slurm_job(f"sbatch {singletask_repeat_script_path}")
        if delete:
            os.system(f"rm -rf {pwd}/result/SingleNodeMode/{params.software}/singletask/*")
        else:
            pass

rule checkpoint:
    input:
        expand("{pwd}/log/{mode}/{software}/singletask/run.done",mode=mode,software=softwares,pwd=pwd),
        expand("{pwd}/log/{mode}/{software}/singletask/repeat.done",mode=mode,software=softwares,pwd=pwd)
    output:
        touch("{pwd}/log/{mode}/checkpoint.done")
    run:
        pass

rule bwa_multi:
    input:
        genome="{pwd}/dataset/BWA/hg18/hg18.fa.gz",
        fastq="{pwd}/dataset/BWA/ERR000589",
        checkpoint="{pwd}/log/{mode}/checkpoint.done"
    output:
        touch("{pwd}/log/{mode}/BWA/multitasks/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/BWA/script"
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/BWA_multitasks/{count}")
            os.system(f"cp -r {pwd}/dataset/BWA/ERR000589 {pwd}/dataset/BWA_multitasks/{count}/")
        run_mode=cal_multitasks_scale(cpus)
        for parallelmode in run_mode:
            cpus_per_task=int(parallelmode.split("_")[0])
            parallel=int(parallelmode.split("_")[1])
            multitasks_script_path=f"{script_path}/multitasks_{cpus_per_task}_{parallel}.sh"
            target_node,job_id=submit_singlenode_multitasks(partition,nodelist,cpus_per_task,"BWA",pwd,parallel,multitasks_count,multitasks_script_path)
            wait_for_job_to_complete(job_id)
            logdir = f"{pwd}/log/SingleNodeMode/BWA/multitasks/{cpus_per_task}_{parallel}"
            with open(f"{logdir}/hostname", "w") as f:
                f.write(target_node)
            if delete:
                os.system(f"rm -rf {pwd}/result/SingleNodeMode/BWA/multitasks/*")
            else:
                pass

rule spades_multi:
    input:
        fastq="{pwd}/dataset/SPAdes/EAS20_8",
        checkpoint="{pwd}/log/{mode}/checkpoint.done"
    output:
        touch("{pwd}/log/{mode}/SPAdes/multitasks/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/SPAdes/script"
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/SPAdes_multitasks/{count}")
            os.system(f"cp -r {pwd}/dataset/SPAdes/EAS20_8 {pwd}/dataset/SPAdes_multitasks/{count}/")
        run_mode=cal_multitasks_scale(cpus)
        for parallelmode in run_mode:
            cpus_per_task=int(parallelmode.split("_")[0])
            parallel=int(parallelmode.split("_")[1])
            multitasks_script_path=f"{script_path}/multitasks_{cpus_per_task}_{parallel}.sh"
            target_node,job_id=submit_singlenode_multitasks(partition,nodelist,cpus_per_task,"SPAdes",pwd,parallel,multitasks_count,multitasks_script_path)
            wait_for_job_to_complete(job_id)
            logdir = f"{pwd}/log/SingleNodeMode/SPAdes/multitasks/{cpus_per_task}_{parallel}"
            with open(f"{logdir}/hostname", "w") as f:
                f.write(target_node)
            if delete:
                os.system(f"rm -rf {pwd}/result/SingleNodeMode/SPAdes/multitasks/*")
            else:
                pass

rule bismark_multi:
    input:
        genome="{pwd}/dataset/Bismark/hg18/",
        fastq="{pwd}/dataset/Bismark/SRR020138.fastq",
        checkpoint="{pwd}/log/{mode}/checkpoint.done"
    output:
        touch("{pwd}/log/{mode}/Bismark/multitasks/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/Bismark/script"
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/Bismark_multitasks/{count}")
            os.system(f"cp {pwd}/dataset/Bismark/SRR020138.fastq {pwd}/dataset/Bismark_multitasks/{count}/")
        run_mode=cal_multitasks_scale(cpus//4)
        for parallelmode in run_mode:
            cpus_per_task=int(parallelmode.split("_")[0])
            parallel=int(parallelmode.split("_")[1])
            multitasks_script_path=f"{script_path}/multitasks_{cpus_per_task}_{parallel}.sh"
            target_node,job_id=submit_singlenode_multitasks(partition,nodelist,cpus_per_task,"Bismark",pwd,parallel,multitasks_count,multitasks_script_path)
            print(target_node,job_id)
            wait_for_job_to_complete(job_id)
            logdir = f"{pwd}/log/SingleNodeMode/Bismark/multitasks/{cpus_per_task}_{parallel}"
            with open(f"{logdir}/hostname", "w") as f:
                f.write(target_node)
            if delete:
                os.system(f"rm -rf {pwd}/result/SingleNodeMode/Bismark/multitasks/*")
            else:
                pass

rule star_multi:
    input:
        genome="{pwd}/dataset/STAR/mm39StarIndex",
        fastq="{pwd}/dataset/STAR/SRR6821753.fastq",
        checkpoint="{pwd}/log/{mode}/checkpoint.done"
    output:
        touch("{pwd}/log/{mode}/STAR/multitasks/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/STAR/script"
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/STAR_multitasks/{count}")
            os.system(f"cp {pwd}/dataset/STAR/SRR6821753.fastq {pwd}/dataset/STAR_multitasks/{count}/")
        run_mode=cal_multitasks_scale(cpus)
        for parallelmode in run_mode:
            cpus_per_task=int(parallelmode.split("_")[0])
            parallel=int(parallelmode.split("_")[1])
            multitasks_script_path=f"{script_path}/multitasks_{cpus_per_task}_{parallel}.sh"
            target_node,job_id=submit_singlenode_multitasks(partition,nodelist,cpus_per_task,"STAR",pwd,parallel,multitasks_count,multitasks_script_path)
            wait_for_job_to_complete(job_id)
            logdir = f"{pwd}/log/SingleNodeMode/STAR/multitasks/{cpus_per_task}_{parallel}"
            with open(f"{logdir}/hostname", "w") as f:
                f.write(target_node)
            if delete:
                os.system(f"rm -rf {pwd}/result/SingleNodeMode/STAR/multitasks/*")
            else:
                pass

rule cellranger_multi:
    input:
        genome="{pwd}/dataset/Cellranger/refdata-gex-GRCh38-2024-A/",
        fastq="{pwd}/dataset/Cellranger/pbmc_1k_v3_fastqs",
        checkpoint="{pwd}/log/{mode}/checkpoint.done"
    output:
        touch("{pwd}/log/{mode}/Cellranger/multitasks/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/Cellranger/script"
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/Cellranger_multitasks/{count}")
            os.system(f"cp -r {pwd}/dataset/Cellranger/pbmc_1k_v3_fastqs {pwd}/dataset/Cellranger_multitasks/{count}/")
        run_mode=cal_multitasks_scale(cpus)
        for parallelmode in run_mode:
            cpus_per_task=int(parallelmode.split("_")[0])
            parallel=int(parallelmode.split("_")[1])
            multitasks_script_path=f"{script_path}/multitasks_{cpus_per_task}_{parallel}.sh"
            target_node,job_id=submit_singlenode_multitasks(partition,nodelist,cpus_per_task,"Cellranger",pwd,parallel,multitasks_count,multitasks_script_path)
            wait_for_job_to_complete(job_id)
            logdir = f"{pwd}/log/SingleNodeMode/Cellranger/multitasks/{cpus_per_task}_{parallel}"
            with open(f"{logdir}/hostname", "w") as f:
                f.write(target_node)
            if delete:
                os.system(f"rm -rf {pwd}/result/SingleNodeMode/Cellranger/multitasks/*")
            else:
                pass

rule gatk_multi:
    input:
        genome="{pwd}/dataset/GATK/GRCh38_full_analysis_set_plus_decoy_hla.fa",
        bam="{pwd}/dataset/GATK/NA12878_24RG_small.hg38.bam",
        bai="{pwd}/dataset/GATK/NA12878_24RG_small.hg38.bam.bai",
        checkpoint="{pwd}/log/{mode}/checkpoint.done"
    output:
        touch("{pwd}/log/{mode}/GATK/multitasks/run.done")
    run:
        script_path=f"{pwd}/result/{mode}/GATK/script"
        for count in range(1,multitasks_count+1):
            os.system(f"mkdir -p {pwd}/dataset/GATK_multitasks/{count}")
            os.system(f"cp {pwd}/dataset/GATK/NA12878_24RG_small.hg38.bam* {pwd}/dataset/GATK_multitasks/{count}/")
        run_mode=cal_gatk_multitasks_scale(cpus,mem)
        for parallelmode in run_mode:
            cpus_per_task=int(parallelmode.split("_")[0])
            parallel=int(parallelmode.split("_")[1])
            multitasks_script_path=f"{script_path}/multitasks_{cpus_per_task}_{parallel}.sh"
            target_node,job_id=submit_singlenode_multitasks(partition,nodelist,cpus_per_task,"GATK",pwd,parallel,multitasks_count,multitasks_script_path)
            wait_for_job_to_complete(job_id)
            logdir = f"{pwd}/log/SingleNodeMode/GATK/multitasks/{cpus_per_task}_{parallel}"
            with open(f"{logdir}/hostname", "w") as f:
                f.write(target_node)
            if delete:
                os.system(f"rm -rf {pwd}/result/SingleNodeMode/GATK/multitasks/*")
            else:
                pass
