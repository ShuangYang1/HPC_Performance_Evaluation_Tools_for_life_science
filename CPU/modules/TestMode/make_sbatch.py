import os


def make_sbatch(partition, nodes, ntasks, cpus_per_task, software, pwd, script_path):
    if not os.path.exists(f"{pwd}/log/TestMode/{software}"):
        os.makedirs(f"{pwd}/log/TestMode/{software}")
    output_path = f"{pwd}/log/TestMode/{software}/test.out"
    error_path = f"{pwd}/log/TestMode/{software}/test.err"
    log_path = f"{pwd}/log/TestMode/{software}/test.log"
    resultdir = f"{pwd}/result/TestMode/{software}/"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --partition={}\n".format(partition))
        f.write("#SBATCH --nodes={}\n".format(nodes))
        f.write("#SBATCH --ntasks={}\n".format(ntasks))
        f.write("#SBATCH --cpus-per-task={}\n".format(cpus_per_task))
        f.write("#SBATCH --job-name={}\n".format(software))
        f.write("#SBATCH --output={}\n".format(output_path))
        f.write("#SBATCH --error={}\n".format(error_path))
        f.write("#SBATCH --time=7-00:00:00\n")
        f.write("\n")
        f.write(f"mkdir -p {resultdir}\n")
        f.write(f"cd {resultdir}\n")
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
                f"(time bismark --parallel {cpus_per_task // 4} --genome {genome} {fastq} -o {resultdir}) >& {log_path}\n"
            )
        elif software == "STAR":
            genome = f"{pwd}/dataset/STAR/mm39StarIndex"
            fastq = f"{pwd}/dataset/STAR/SRR6821753.fastq"
            f.write("ulimit -n $(ulimit -Hn)\n")
            f.write(
                f"(time STAR --runThreadN {cpus_per_task} --genomeDir {genome} --readFilesIn {fastq} --outFileNamePrefix {resultdir} --runMode alignReads --quantMode TranscriptomeSAM GeneCounts --outSAMtype BAM SortedByCoordinate) >& {log_path}\n"
            )
        elif software == "Cellranger":
            genome = f"{pwd}/dataset/Cellranger/refdata-gex-GRCh38-2024-A/"
            fastq = f"{pwd}/dataset/Cellranger/pbmc_1k_v3_fastqs"
            run_id = "run_count_1kpbmcs"
            sample = "pbmc_1k_v3"
            f.write(
                f"(time cellranger count --localcores={cpus_per_task} --fastqs={fastq} --transcriptome {genome} --output-dir=./test/ --id={run_id} --sample={sample} --create-bam true) >& {log_path}\n"
            )
        elif software == "GATK":
            genome = f"{pwd}/dataset/GATK/GRCh38_full_analysis_set_plus_decoy_hla.fa"
            bam = f"{pwd}/dataset/GATK/NA12878_24RG_small.hg38.bam"
            f.write(
                f"(time gatk HaplotypeCaller --native-pair-hmm-threads {cpus_per_task} -R {genome} -I {bam} -O test.vcf) >& {log_path}\n"
            )
    return True
