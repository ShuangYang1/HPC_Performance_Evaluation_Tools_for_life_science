# CPU版集群性能测评工具
针对生命科学计算的高性能集群性能分析及评测框架，使用代表性生物学计算软件集合，通过Prometheus监控软件对运行时 CPU、内存、IO等资源需求特征进行监控，使用计算用时、计算效率、CPU核时等关键评价指标对集群性能进行测评和打分，帮助指导生命科学计算集群的建设、提升生命科学计算生产效率。


## 测评工具能干什么？
1. 对已建集群进行性能评测，评估集群性能，帮助分析性能瓶颈，指导集群建设。
2. 对集群环境配置的性能进行综合打分，能够直观对比不同配置下的性能差异，帮助指导集群环境配置和集群设备采购。
3. 对不同生命科学计算软件的性能和资源需求进行分析，帮助指导计算任务优化和集群资源调度。

## 测评工具的特点
1. **同时提供软件运行效率分析及节点性能评测**。
2. **支持多种测评模式**（详见下文“测评内容和流程”）。
3. **支持自定义测评的计算软件**：用户可根据自身需求选择部分计算软件进行测评，模块化的测评工具也易于添加新的计算软件。
4. **支持不同集群架构**。
5. **提供详细的可视化测评报告**：包括各种图表以及性能打分。


## 测评内容和流程
CPU版评测工具选用BWA，SPAdes，Bismark，STAR，Cellranger，GATK作为测试软件。  
测评内容都是使用每个软件提供的算例进行实际运行计算，  
- 根据任务计算期间的资源使用情况绘制雷达图，帮助分析集群配置是否存在瓶颈。
- 根据任务计算时间、CPU核时、节点能耗、计算用时稳定性等多个指标进行打分。

针对不同的测评需求，我们提供了四种测评模式：
1. 测试模式：测试软件的安装情况，测试软件是否能够正常运行
2. 快速测评模式：每个软件只测试一个计算任务，根据计算用时进行打分（通常用于对节点的性能进行初步快速了解，更细致的测评分析需进行后续的单节点和集群测评）。
3. 单节点深度测评模式：使用单个节点测试软件的运行性能，进行深度测评。主要测评内容包括单任务多线程并行计算效率分析、计算用时稳定性分析、大批量任务计算效率分析，根据计算用时、计算用时稳定性、计算所用的能耗进行分析和打分，输出测试报告。报告内容和格式参考[单节点深度测评报告](./report_example/SingleNodeMode_report.md)。
4. 集群测评模式：在多个节点组成的集群上测试每个软件大批量任务的计算效率，根据计算用时、计算所用的能耗进行分析和打分，输出测试报告。报告内容和格式参考[集群测评报告](./report_example/ClusterMode_report.md)。（建议先进行单节点深度测评模式测试，根据单节点测评报告中每个软件大批量任务测试中的最佳并行规模修改集群测评模式的配置文件，然后开始集群测评）

完整的测评流程耗时较长，对于需要快速了解集群性能的用户，我们建议：
1. 先进行测试模式，验证各软件能够正常运行。
2. 再进行快速测评模式，根据测试结果中的得分，了解集群性能。

对于需要详细了解集群性能的用户，我们建议：
1. 先进行测试模式，验证各软件能够正常运行。
2. 进行单节点深度测评模式，根据输出的报告分析测评的节点配置环境是否存在瓶颈，也可比较不同配置节点的得分。
3. 根据单节点深度测评报告中每个软件大批量任务测试中的最佳并行规模修改集群测评模式的配置文件，然后进行集群测评模式。根据输出的报告分析测评的集群配置环境是否存在瓶颈，也可比较不同配置集群的得分。

完整测评需要至少30TB的存储空间（默认参数的情况下），请根据实际情况准备充足的存储空间。

## 测评工具怎么用？
### 准备环境
1. slurm作业调度系统
2. Python>=3.11， Python第三方库：matplotlib，numpy，filelock，snakemake(>=8.0.0)
3. 安装对应版本的测试软件，确保测试软件已正确安装，并将可执行文件添加到环境变量PATH中
    - [BWA](https://github.com/lh3/bwa)：v0.7.18-r1243-dirty
    - [SPAdes](https://github.com/ablab/spades)：v4.0.0
    - [Bismark](https://github.com/FelixKrueger/Bismark)：v0.24.2
    - [STAR](https://github.com/alexdobin/STAR)：v2.7.11b
    - [Cellranger](https://www.10xgenomics.com/support/software/cell-ranger/downloads/previous-versions)：v8.0.1
    - [GATK](https://github.com/broadinstitute/gatk)：v4.6.0.0

    上述软件版本为测评工具开发时使用的软件版本，为了保证测评工具的兼容性和测试结果的可比较性，建议使用相同软件版本进行测评。  
4. 安装其他依赖软件，需将可执行文件添加到环境变量PATH中
    - bowtie2：v2.5.2（Bismark依赖）
    - gzip
    - bgzip
    - tabix
    - vcftools（需要vcf-compare工具）
    - samtools
5. 下载测试数据
    需安装[sratoolkit](https://github.com/ncbi/sra-tools/wiki/01.-Downloading-SRA-Toolkit)，可通过`conda install sra-tools`安装或下载[源码](https://github.com/ncbi/sra-tools/wiki/01.-Downloading-SRA-Toolkit)安装。
    - BWA测试数据：在dataset/BWA/目录下执行`prefetch ERR000589`下载数据
    - BWA测试使用的参考基因组：https://hgdownload.soe.ucsc.edu/goldenPath/hg18/bigZips/hg18.fa.gz
    - SPAdes测试数据：http://ftp.sra.ebi.ac.uk/vol1/run/ERR008/ERR008613/200x100x100-081224_EAS20_0008_FC30TBBAAXX-6.tar.gz
    - Bismark测试数据：在dataset/Bismark/目录下执行`prefetch SRR020138`下载数据
    - STAR测试数据：在dataset/STAR/目录下执行`prefetch SRR6821753`下载数据
    - STAR测试使用的参考基因组：https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.fa.gz
    - Cellranger测试数据：https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_v3/pbmc_1k_v3_fastqs.tar
    - Cellranger测试使用的参考基因组：https://cf.10xgenomics.com/supp/cell-exp/refdata-gex-GRCh38-2024-A.tar.gz
    - GATK测试数据：https://42basepairs.com/download/s3/gatk-test-data/wgs_bam/NA12878_24RG_hg38/NA12878_24RG_small.hg38.bam
    - GATK测试使用的参考基因组：https://ftp-trace.ncbi.nih.gov/1000genomes/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa

6. 数据预处理
    - BWA测试数据：在dataset/BWA/目录下执行`fasterq-dump ERR000589 --outdir ERR000589/`解压数据，然后删除ERR000589/ERR000589.sra文件
    - BWA测试使用的参考基因组：将下载好的hg18.fa.gz放到dataset/BWA/hg18/目录下，然后在该目录下执行`bwa index hg18.fa.gz`（需要确保bwa已经正确安装且能正常运行）
    - SPAdes测试数据：解压测试数据，将解压后得到的EAS20_8目录移动到dataset/SPAdes/目录下
    - Bismark测试数据：在dataset/Bismark/目录下执行`fasterq-dump SRR020138`解压数据
    - Bismark测试使用的参考基因组：将dataset/BWA/hg18/hg18.fa.gz复制到dataset/Bismark/hg18/目录下，然后在dataset/Bismark/目录下执行`bismark_genome_preparation hg18/`（需要确保bismark已经正确安装且能正常运行）
    - STAR测试数据：在dataset/STAR/目录下执行`fasterq-dump SRR6821753`解压数据
    - STAR测试使用的参考基因组：将下载好的mm39.fa.gz放到dataset/STAR/目录下后解压该文件，然后在dataset/STAR/目录下`gunzip GENCODE_mm39.gtf.gz`解压gtf文件，最后在该目录下执行`STAR --runThreadN 10 --runMode genomeGenerate --genomeDir mm39StarIndex --genomeFastaFiles mm39.fa --sjdbGTFfile GENCODE_mm39.gtf`（需要确保STAR已经正确安装且能正常运行）
    - Cellranger测试数据：将下载得到的pbmc_1k_v3_fastqs.tar放到dataset/Cellranger/目录下，然后解压该文件
    - Cellranger测试使用的参考基因组：将下载得到的refdata-gex-GRCh38-2024-A.tar.gz文件放到dataset/Cellranger/目录下，然后解压该文件
    - GATK测试数据：将下载得到的NA12878_24RG_small.hg38.bam文件放到dataset/GATK/目录下，然后在dataset/GATK/目录下执行`samtools index NA12878_24RG_small.hg38.bam`（需要确保samtools已经正确安装且能正常运行）
    - GATK测试使用的参考基因组：将下载得到的GRCh38_full_analysis_set_plus_decoy_hla.fa文件放到dataset/GATK/目录下，然后在该目录下执行`gatk CreateSequenceDictionary -R GRCh38_full_analysis_set_plus_decoy_hla.fa`和`samtools faidx GRCh38_full_analysis_set_plus_decoy_hla.fa`（需要确保gatk和samtools已经正确安装且能正常运行）
7. 部署Prometheus监控软件，参考[Prometheus官网](https://prometheus.ac.cn/)
    - 在计算节点上启动node_exporter，并确保node_exporter能够正常运行
    - 在监控服务器上修改prometheus.yml文件
    - 配置文件中添加计算节点名称和node_exporter端口，以及计算节点名称和gpu_exporter端口，如 `- targets: ["nodeName:9100"]`（需要使用计算节点名称，且和后边config.yaml文件中指定的节点名称保持一致）
    - 启动prometheus
    - 检查prometheus是否能够获取到计算节点的监控数据


### 文件说明
- dataset/：测试使用数据集
- modules/：测试使用的一些代码模块
- config/：测试使用的snakemake配置文件（需要自行调整）
- SoftwaresWeight.txt：需要测试的软件名称及对应权重，对应测试报告计算总分时每个软件的权重，权重越大，该软件在总分中的占比越大（若某些软件不需要测试，可将该软件一行删除或将权重设为0）
- 若某个软件权重为0，则该软件不会进行测试。（可自行调整）
- TestMode.snakefile：测试模式脚本
- FastMode.snakefile：快速测评模式脚本
- SingleNodeMode.snakefile：单节点深度测评模式脚本
- ClusterMode.snakefile：集群测评模式脚本


### 开始测评
测试前请根据实际情况调整config/*/中的config.yaml文件，然后选择相应的测评模式开始测评。
1. 测试模式：  
    根据实际情况调整config/TestMode/config.yaml中以下内容：
    - partition：待测试的队列
    - cpus：单个节点的CPU核心数量

    然后在此目录下执行：  
    `snakemake -s TestMode.snakefile`  
    检查输出是否报错，如果报错检查软件是否正确安装。
2. 快速测评模式：  
    根据实际情况调整config/FastMode/config.yaml中以下内容：
    - partition：待测试的队列
    - cpus：单个节点的CPU核心数量

    然后在此目录下执行：  
    `snakemake -s FastMode.snakefile`  
    输出名为FastMode_result.txt的文件，包含每个软件的测试得分，以及当前环境测试的总得分。
3. 单节点深度测评模式：  
    根据实际情况调整config/SingleNodeMode/config.yaml中以下内容：
    - partition：待测试的节点所在队列
    - node：待测试的节点（可以指定多个节点，能够并行处理不同软件的测试任务，更快地完成测试，但可能会占用较大的IO资源，若存储IO性能较低可能影响测试结果）
    - cpus：该节点的CPU核心数量
    - mem_mb：单个节点的内存大小（单位：MB）
    - multitasks_count：每款计算软件大批量任务测试时，提交的任务数量（默认为50）
    - repeat：每款计算软件单任务计算用时稳定性分析测试时，相同任务重复的次数（默认10次）
    - node_exporter_port：计算节点监控软件的端口
    - server_ip：监控server节点的IP地址
    - server_port：监控server节点的端口
    - delete-intermediate-files：是否在测试流程中途删除结果文件（true/false），若存储空间充足，建议设置为false，否则设置为true

    然后在此目录下执行：  
    `snakemake -s SingleNodeMode.snakefile`  
    输出名为SingleNodeMode_report.md的文件，为单节点测评模式的详细测试报告。
4. 集群测评模式：  
    建议先进行单节点深度测评模式测试，对节点的性能和计算软件可运行的规模有个初步了解，根据每个计算软件在单节点测评大批量任务测试中的最佳线程数调整集群测试的配置文件。  
    根据实际情况调整config/ClusterMode/config.yaml中以下内容：
    - partition：待测试的队列
    - nodelist：待测试的节点列表，用逗号分隔
    - cpus_per_node：单个节点的CPU核心数量
    - multitasks_count：每款计算软件大批量任务测试时，提交的任务数量，默认为节点数量*50
    - bwa_cpus_per_task：每个BWA计算任务分配的CPU核心数量,默认为1
    - spades_cpus_per_task：每个SPAdes计算任务分配的CPU核心数量,默认为1
    - bismark_parallel：每个Bismark计算任务的parallel参数值,默认为1
    - star_cpus_per_task：每个STAR计算任务分配的CPU核心数量,默认为1
    - cellranger_cpus_per_task：每个Cellranger计算任务分配的CPU核心数量,默认为1
    - gatk_cpus_per_task：每个GATK计算任务分配的CPU核心数量,默认为1  

    （以上软件的gpus_per_task参数需要根据单节点测评报告中的大批量任务测试的最佳并行模式进行调整）
    - node_exporter_port：计算节点监控软件的端口
    - server_ip：监控server节点的IP地址
    - server_port：监控server节点的端口
    - delete-intermediate-files：是否在测试流程中途删除结果文件（true/false），若存储空间充足，建议设置为false，否则设置为true

    然后在此目录下执行：  
    snakemake -s ClusterMode.snakefile  
    输出名为ClusterMode_report.md的文件，为单节点测评模式的详细测试报告。
