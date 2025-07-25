import os


def find_best_singletask_threads(software, threads):
    best_thread = None
    best_time = None
    for t in threads:
        if os.path.exists(
            f"log/SingleNodeMode/{software}/singletask/{t}.start.timestamp"
        ) and os.path.exists(
            f"log/SingleNodeMode/{software}/singletask/{t}.end.timestamp"
        ):
            with open(
                f"log/SingleNodeMode/{software}/singletask/{t}.start.timestamp", "r"
            ) as f:
                start_timestamp = int(f.read())
            with open(
                f"log/SingleNodeMode/{software}/singletask/{t}.end.timestamp", "r"
            ) as f:
                end_timestamp = int(f.read())
            if best_time is None or end_timestamp - start_timestamp < best_time:
                best_time = end_timestamp - start_timestamp
                best_thread = t
    return best_thread, best_time


def make_singletask(
    partition,
    nodes,
    ntasks,
    cpus,
    cpus_per_task,
    software,
    pwd,
    nodelist,
    script_path,
):
    logdir = f"{pwd}/log/SingleNodeMode/{software}/singletask"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if software == "Bismark":
        cpus_per_task = cpus_per_task // 4
    output_path = f"{logdir}/{cpus_per_task}.out"
    error_path = f"{logdir}/{cpus_per_task}.err"
    log_path = f"{logdir}/{cpus_per_task}.log"
    start = f"{logdir}/{cpus_per_task}.start.timestamp"
    hostname = f"{logdir}/{cpus_per_task}.hostname"
    end = f"{logdir}/{cpus_per_task}.end.timestamp"
    resultdir = f"{pwd}/result/SingleNodeMode/{software}/singletask/"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --partition={}\n".format(partition))
        f.write("#SBATCH --nodes={}\n".format(nodes))
        f.write("#SBATCH --ntasks={}\n".format(ntasks))
        if software == "Bismark":
            f.write("#SBATCH --cpus-per-task={}\n".format(cpus_per_task * 4))
        else:
            f.write("#SBATCH --cpus-per-task={}\n".format(cpus_per_task))
        f.write("#SBATCH --job-name={}\n".format(software))
        f.write("#SBATCH --output={}\n".format(output_path))
        f.write("#SBATCH --error={}\n".format(error_path))
        f.write(f"#SBATCH --nodelist={nodelist}\n")
        f.write("#SBATCH --time=7-00:00:00\n")
        f.write("#SBATCH --exclusive\n")
        f.write("#SBATCH --cpu-bind=cores\n")
        f.write("#SBATCH --mem-bind=local\n")
        f.write("\n")
        f.write(f'WORKDIR="{resultdir}/{cpus_per_task}"\n')
        f.write("mkdir -p $WORKDIR\n")
        f.write("cd $WORKDIR\n")
        f.write(f"hostname > {hostname}\n")
        f.write(f"date +%s > {start}\n")
        if software == "BWA":
            genome = f"{pwd}/dataset/BWA/hg18/hg18.fa.gz"
            fastq1 = f"{pwd}/dataset/BWA/ERR000589/ERR000589_1.fastq"
            fastq2 = f"{pwd}/dataset/BWA/ERR000589/ERR000589_2.fastq"
            f.write(
                f"(time bwa mem -t {cpus_per_task} {genome} {fastq1} {fastq2} > ./result.sam) >& {log_path}\n"
            )
        elif software == "SPAdes":
            fastq1 = f"{pwd}/dataset/SPAdes/EAS20_8/s_6_1.fastq"
            fastq2 = f"{pwd}/dataset/SPAdes/EAS20_8/s_6_2.fastq"
            f.write(
                f"(time spades.py -t {cpus_per_task} --isolate --pe1-1 {fastq1} --pe1-2 {fastq2} -o ./) >& {log_path}\n"
            )
        elif software == "Bismark":
            genome = f"{pwd}/dataset/Bismark/hg18/"
            fastq = f"{pwd}/dataset/Bismark/SRR020138.fastq"
            f.write(
                f"(time bismark --parallel {cpus_per_task} --genome {genome} {fastq} -o {resultdir}/{cpus_per_task}) >& {log_path}\n"
            )
        elif software == "STAR":
            genome = f"{pwd}/dataset/STAR/mm39StarIndex"
            fastq = f"{pwd}/dataset/STAR/SRR6821753.fastq"
            f.write("ulimit -n $(ulimit -Hn)\n")
            f.write(
                f"(time STAR --runThreadN {cpus_per_task} --genomeDir {genome} --readFilesIn {fastq} --outFileNamePrefix {resultdir}/{cpus_per_task}/ --runMode alignReads --quantMode TranscriptomeSAM GeneCounts --outSAMtype BAM SortedByCoordinate) >& {log_path}\n"
            )
        elif software == "Cellranger":
            genome = f"{pwd}/dataset/Cellranger/refdata-gex-GRCh38-2024-A/"
            fastq = f"{pwd}/dataset/Cellranger/pbmc_1k_v3_fastqs"
            run_id = "run_count_1kpbmcs"
            sample = "pbmc_1k_v3"
            f.write(
                f"(time cellranger count --localcores={cpus_per_task} --fastqs={fastq} --transcriptome {genome} --output-dir=./result/ --id={run_id} --sample={sample} --create-bam true) >& {log_path}\n"
            )
        elif software == "GATK":
            genome = f"{pwd}/dataset/GATK/GRCh38_full_analysis_set_plus_decoy_hla.fa"
            bam = f"{pwd}/dataset/GATK/NA12878_24RG_small.hg38.bam"
            f.write(
                f"(time gatk HaplotypeCaller --native-pair-hmm-threads {cpus_per_task} -R {genome} -I {bam} -O test.vcf) >& {log_path}\n"
            )
        f.write(f"date +%s > {end}\n")
    return True


def make_singletask_repeat(
    partition,
    nodes,
    ntasks,
    cpus,
    cpus_per_task,
    software,
    pwd,
    nodelist,
    repeat,
    script_path,
):
    logdir = f"{pwd}/log/SingleNodeMode/{software}/singletask"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if software == "Bismark":
        cpus_per_task = cpus_per_task // 4
    output_path = f"{logdir}/{cpus_per_task}_%a.out"
    error_path = f"{logdir}/{cpus_per_task}_%a.err"
    log_path = f"{logdir}/{cpus_per_task}_${{SLURM_ARRAY_TASK_ID}}.log"
    start = f"{logdir}/{cpus_per_task}_${{SLURM_ARRAY_TASK_ID}}.start.timestamp"
    hostname = f"{logdir}/{cpus_per_task}_${{SLURM_ARRAY_TASK_ID}}.hostname"
    end = f"{logdir}/{cpus_per_task}_${{SLURM_ARRAY_TASK_ID}}.end.timestamp"
    resultdir = f"{pwd}/result/SingleNodeMode/{software}/singletask/"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --partition={}\n".format(partition))
        f.write("#SBATCH --nodes={}\n".format(nodes))
        f.write("#SBATCH --ntasks={}\n".format(ntasks))
        if software == "Bismark":
            f.write("#SBATCH --cpus-per-task={}\n".format(cpus_per_task * 4))
        else:
            f.write("#SBATCH --cpus-per-task={}\n".format(cpus_per_task))
        f.write("#SBATCH --job-name={}_repeat\n".format(software))
        f.write("#SBATCH --output={}\n".format(output_path))
        f.write("#SBATCH --error={}\n".format(error_path))
        f.write(f"#SBATCH --array=2-{repeat}\n")
        f.write(f"#SBATCH --nodelist={nodelist}\n")
        f.write("#SBATCH --time=7-00:00:00\n")
        f.write("#SBATCH --exclusive\n")
        f.write("#SBATCH --cpu-bind=cores\n")
        f.write("#SBATCH --mem-bind=local\n")
        f.write("\n")
        f.write(f'WORKDIR="{resultdir}/{cpus_per_task}_${{SLURM_ARRAY_TASK_ID}}"\n')
        f.write("mkdir -p $WORKDIR\n")
        f.write("cd $WORKDIR\n")
        f.write(f"hostname > {hostname}\n")
        f.write(f"date +%s > {start}\n")
        if software == "BWA":
            genome = f"{pwd}/dataset/BWA/hg18/hg18.fa.gz"
            fastq1 = f"{pwd}/dataset/BWA/ERR000589/ERR000589_1.fastq"
            fastq2 = f"{pwd}/dataset/BWA/ERR000589/ERR000589_2.fastq"
            f.write(
                f"(time bwa mem -t {cpus_per_task} {genome} {fastq1} {fastq2} > ./result.sam) >& {log_path}\n"
            )
        elif software == "SPAdes":
            fastq1 = f"{pwd}/dataset/SPAdes/EAS20_8/s_6_1.fastq"
            fastq2 = f"{pwd}/dataset/SPAdes/EAS20_8/s_6_2.fastq"
            f.write(
                f"(time spades.py -t {cpus_per_task} --isolate --pe1-1 {fastq1} --pe1-2 {fastq2} -o ./) >& {log_path}\n"
            )
        elif software == "Bismark":
            genome = f"{pwd}/dataset/Bismark/hg18/"
            fastq = f"{pwd}/dataset/Bismark/SRR020138.fastq"
            f.write(
                f"(time bismark --parallel {cpus_per_task} --genome {genome} {fastq} -o {resultdir}/{cpus_per_task}_${{SLURM_ARRAY_TASK_ID}}) >& {log_path}\n"
            )
        elif software == "STAR":
            genome = f"{pwd}/dataset/STAR/mm39StarIndex"
            fastq = f"{pwd}/dataset/STAR/SRR6821753.fastq"
            f.write("ulimit -n $(ulimit -Hn)\n")
            f.write(
                f"(time STAR --runThreadN {cpus_per_task} --genomeDir {genome} --readFilesIn {fastq} --outFileNamePrefix {resultdir}/{cpus_per_task}_${{SLURM_ARRAY_TASK_ID}}/ --runMode alignReads --quantMode TranscriptomeSAM GeneCounts --outSAMtype BAM SortedByCoordinate) >& {log_path}\n"
            )
        elif software == "Cellranger":
            genome = f"{pwd}/dataset/Cellranger/refdata-gex-GRCh38-2024-A/"
            fastq = f"{pwd}/dataset/Cellranger/pbmc_1k_v3_fastqs"
            run_id = "run_count_1kpbmcs"
            sample = "pbmc_1k_v3"
            f.write(
                f"(time cellranger count --localcores={cpus_per_task} --fastqs={fastq} --transcriptome {genome} --output-dir=./result/ --id={run_id} --sample={sample} --create-bam true) >& {log_path}\n"
            )
        elif software == "GATK":
            genome = f"{pwd}/dataset/GATK/GRCh38_full_analysis_set_plus_decoy_hla.fa"
            bam = f"{pwd}/dataset/GATK/NA12878_24RG_small.hg38.bam"
            f.write(
                f"(time gatk HaplotypeCaller --native-pair-hmm-threads {cpus_per_task} -R {genome} -I {bam} -O test.vcf) >& {log_path}\n"
            )
        f.write(f"date +%s > {end}\n")
    return True


def make_multitasks(
    partition,
    nodes,
    ntasks,
    cpus_per_task,
    software,
    pwd,
    nodelist,
    parallel,
    multitasks_count,
    script_path,
):
    logdir = (
        f"{pwd}/log/SingleNodeMode/{software}/multitasks/{cpus_per_task}_{parallel}"
    )
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
        if software == "Bismark":
            f.write("#SBATCH --cpus-per-task={}\n".format(cpus_per_task * 4))
        else:
            f.write("#SBATCH --cpus-per-task={}\n".format(cpus_per_task))
        f.write("#SBATCH --job-name={}_multitasks\n".format(software))
        f.write("#SBATCH --output={}\n".format(output_path))
        f.write("#SBATCH --error={}\n".format(error_path))
        f.write(f"#SBATCH --array=1-{multitasks_count}\n")
        f.write(f"#SBATCH --nodelist={nodelist}\n")
        f.write("#SBATCH --time=7-00:00:00\n")
        f.write("#SBATCH --cpu-bind=cores\n")
        f.write("#SBATCH --mem-bind=local\n")
        if software == "GATK":
            f.write("#SBATCH --mem=10G\n")
        f.write("\n")
        f.write(
            f'WORKDIR="{resultdir}/{cpus_per_task}_{parallel}/${{SLURM_ARRAY_TASK_ID}}"\n'
        )
        f.write("mkdir -p $WORKDIR\n")
        f.write("cd $WORKDIR\n")
        f.write(f"date +%s > {start}\n")
        if software == "BWA":
            genome = f"{pwd}/dataset/BWA/hg18/hg18.fa.gz"
            fastq1 = f"{pwd}/dataset/BWA_multitasks/${{SLURM_ARRAY_TASK_ID}}/ERR000589/ERR000589_1.fastq"
            fastq2 = f"{pwd}/dataset/BWA_multitasks/${{SLURM_ARRAY_TASK_ID}}/ERR000589/ERR000589_2.fastq"
            f.write(
                f"(time bwa mem -t {cpus_per_task} {genome} {fastq1} {fastq2} > ./result.sam) >& {log_path}\n"
            )
        elif software == "SPAdes":
            fastq1 = f"{pwd}/dataset/SPAdes_multitasks/${{SLURM_ARRAY_TASK_ID}}/EAS20_8/s_6_1.fastq"
            fastq2 = f"{pwd}/dataset/SPAdes_multitasks/${{SLURM_ARRAY_TASK_ID}}/EAS20_8/s_6_2.fastq"
            f.write(
                f"(time spades.py -t {cpus_per_task} --isolate --pe1-1 {fastq1} --pe1-2 {fastq2} -o ./) >& {log_path}\n"
            )
        elif software == "Bismark":
            genome = f"{pwd}/dataset/Bismark/hg18/"
            fastq = f"{pwd}/dataset/Bismark_multitasks/${{SLURM_ARRAY_TASK_ID}}/SRR020138.fastq"
            f.write(
                f"(time bismark --parallel {cpus_per_task} --genome {genome} {fastq} -o {resultdir}/{cpus_per_task}_{parallel}/${{SLURM_ARRAY_TASK_ID}}/) >& {log_path}\n"
            )
        elif software == "STAR":
            genome = f"{pwd}/dataset/STAR/mm39StarIndex"
            fastq = f"{pwd}/dataset/STAR_multitasks/${{SLURM_ARRAY_TASK_ID}}/SRR6821753.fastq"
            f.write("ulimit -n $(ulimit -Hn)\n")
            f.write(
                f"(time STAR --runThreadN {cpus_per_task} --genomeDir {genome} --readFilesIn {fastq} --outFileNamePrefix {resultdir}/{cpus_per_task}_{parallel}/${{SLURM_ARRAY_TASK_ID}}/ --runMode alignReads --quantMode TranscriptomeSAM GeneCounts --outSAMtype BAM SortedByCoordinate) >& {log_path}\n"
            )
        elif software == "Cellranger":
            genome = f"{pwd}/dataset/Cellranger/refdata-gex-GRCh38-2024-A/"
            fastq = f"{pwd}/dataset/Cellranger_multitasks/${{SLURM_ARRAY_TASK_ID}}/pbmc_1k_v3_fastqs"
            run_id = "run_count_1kpbmcs"
            sample = "pbmc_1k_v3"
            f.write(
                f"(time cellranger count --localcores={cpus_per_task} --fastqs={fastq} --transcriptome {genome} --output-dir=./result/ --id={run_id} --sample={sample} --create-bam true) >& {log_path}\n"
            )
        elif software == "GATK":
            genome = f"{pwd}/dataset/GATK/GRCh38_full_analysis_set_plus_decoy_hla.fa"
            bam = f"{pwd}/dataset/GATK_multitasks/${{SLURM_ARRAY_TASK_ID}}/NA12878_24RG_small.hg38.bam"
            f.write(
                f"(time gatk HaplotypeCaller --native-pair-hmm-threads {cpus_per_task} -R {genome} -I {bam} -O test.vcf) >& {log_path}\n"
            )
        f.write(f"date +%s > {end}\n")
    return True
