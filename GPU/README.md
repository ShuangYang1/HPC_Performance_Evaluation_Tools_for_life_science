# GPU版集群性能测评工具
针对生命科学计算的高性能集群性能分析及评测框架，使用代表性生物学计算软件集合，通过[Prometheus](https://prometheus.ac.cn/)监控软件对运行时 CPU、GPU、内存、IO等资源需求特征进行监控，使用计算用时、计算效率、CPU核时、GPU卡时等关键评价指标对集群性能进行测评和打分，帮助指导生命科学计算集群的建设、提升生命科学计算生产效率。

## 测评内容和流程
GPU测评模块主要测评GPU集群进行生命科学计算的性能，选用SPONGE，GROMACS，AMBER，DSDP作为测试软件。  
测评内容都是使用每个软件提供的算例进行实际运行计算，  
- 根据任务计算期间的资源使用情况绘制雷达图，帮助分析集群配置是否存在瓶颈。
- 根据任务计算时间、GPU卡时、节点能耗、计算用时稳定性等多个指标进行打分。

针对不同的测评需求，我们提供了四种测评模式：
1. **测试模式**：测试软件的安装情况，测试软件是否能够正常运行
2. **快速测评模式**：每个软件只测试一个计算任务，根据计算用时进行打分（通常用于对节点的性能进行初步快速了解，更细致的测评分析需进行后续的单节点和集群测评）。
3. **单节点深度测评模式**：使用单个节点测试软件的运行性能，进行深度测评。主要测评内容包括单任务多GPU卡并行计算效率分析（部分不支持多GPU卡并行的软件除外）、计算用时稳定性分析、大批量任务计算效率分析，根据分子动力学模拟性能、计算用时、计算用时稳定性、计算所用的能耗进行分析和打分，输出测试报告。报告内容和格式参考[单节点深度测评报告](./report_example/SingleNodeMode_report.md)。
4. **集群测评模式**：在多个节点组成的集群上测试每个软件大批量任务的计算效率，根据计算用时、计算所用的能耗进行分析和打分，输出测试报告。报告内容和格式参考[集群测评报告](./report_example/ClusterMode_report.md)。（建议先进行单节点深度测评模式测试，根据单节点测评报告中每个软件大批量任务测试中的最佳并行规模修改集群测评模式的配置文件，然后开始集群测评）

完整的测评流程耗时较长，对于需要快速了解集群性能的用户，我们建议：
1. 先进行测试模式，验证各软件能够正常运行。
2. 再进行快速测评模式，根据测试结果中的得分，了解集群性能。

对于需要详细了解集群性能的用户，我们建议：
1. 先进行测试模式，验证各软件能够正常运行。
2. 进行单节点深度测评模式，根据输出的报告分析测评的节点配置环境是否存在瓶颈，也可比较不同配置节点的得分。
3. 根据单节点深度测评报告中每个软件大批量任务测试中的最佳并行规模修改集群测评模式的配置文件，然后进行集群测评模式。根据输出的报告分析测评的集群配置环境是否存在瓶颈，也可比较不同配置集群的得分。

完整测评需要至少350GB的存储空间（默认参数的情况下），请根据实际情况准备充足的存储空间。

## 测评工具怎么用？
### 准备环境
1. slurm作业调度系统
2. Python>=3.11， Python第三方库：matplotlib，numpy，scipy，statsmodels，snakemake(>=8.0.0)
3. 安装对应版本的测试软件，确保测试软件已正确安装，并将可执行文件添加到环境变量PATH中
    - [GROMACS](https://manual.gromacs.org/): 2023.3
    - [AMBER](https://ambermd.org/GetAmber.php): 24.0
    - [SPONGE](https://spongemm.cn/zh/%E4%B8%8B%E8%BD%BD/CudaSPONGE%E7%A8%8B%E5%BA%8F): 1.4
    - [DSDP](https://spongemm.cn/zh/DSDP/home): 1.0

    上述软件版本为测评工具开发时使用的软件版本，为了保证测评工具的兼容性和测试结果的可比较性，建议使用相同软件版本进行测评。  
    （DSDP所需环境可能和snakemake所需环境冲突，可以使用conda创建一个独立的DSDP环境，如果DSDP使用的是独立的conda环境需要在后边的config.yaml文件中进行指定）  
4. 部署Prometheus监控软件，参考[Prometheus官网](https://prometheus.ac.cn/)
    - 在计算节点上启动node_exporter和nvidia_gpu_exporter，并确保node_exporter和nvidia_gpu_exporter能够正常运行（如果不是Nvidia GPU卡，需要将nvidia_gpu_exporter替换为适配的gpu_exporter）
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
    - gpus：单个节点GPU卡的数量
    - cpus：单个节点的CPU核心数量
    - conda_path：conda的安装路径（如果DSDP不用独立的conda环境，请在配置文件中删除这一行）
    - dsdp_env：DSDP软件使用的conda环境名称（如果DSDP不用独立的conda环境，请在配置文件中删除这一行）

    然后在此目录下执行：  
    `snakemake -s TestMode.snakefile`  
    检查输出是否报错，如果报错检查软件是否正确安装。
2. 快速测评模式：  
    根据实际情况调整config/FastMode/config.yaml中以下内容：
    - partition：待测试的队列
    - gpus：单个节点GPU卡的数量
    - cpus：单个节点的CPU核心数量
    - conda_path：conda的安装路径（如果DSDP不用独立的conda环境，请在配置文件中删除这一行）
    - dsdp_env：DSDP软件使用的conda环境名称（如果DSDP不用独立的conda环境，请在配置文件中删除这一行）

    然后在此目录下执行：  
    `snakemake -s FastMode.snakefile`  
    输出名为FastMode_result.txt的文件，包含每个软件的测试得分，以及当前环境测试的总得分。
3. 单节点深度测评模式：  
    根据实际情况调整config/SingleNodeMode/config.yaml中以下内容：
    - partition：待测试的节点所在队列
    - node：待测试的节点
    - gpus：该节点GPU卡的数量
    - cpus：该节点的CPU核心数量
    - multitasks_count：每款计算软件大批量任务测试时，提交的任务数量（默认为50）
    - repeat：每款计算软件单任务计算用时稳定性分析测试时，相同任务重复的次数（默认10次）
    - node_exporter_port：计算节点监控软件的端口
    - gpu_exporter_port：计算节点GPU监控软件的端口
    - server_ip：监控server节点的IP地址
    - server_port：监控server节点的端口
    - conda_path：conda的安装路径（如果DSDP不用独立的conda环境，请在配置文件中删除这一行）
    - dsdp_env：DSDP软件使用的conda环境名称（如果DSDP不用独立的conda环境，请在配置文件中删除这一行）

    然后在此目录下执行：  
    `snakemake -s SingleNodeMode.snakefile`  
    输出名为SingleNodeMode_report.md的文件，为单节点测评模式的详细测试报告。
4. 集群测评模式：  
    建议先进行单节点深度测评模式测试，对节点的性能和计算软件可运行的规模有个初步了解，根据每个计算软件在单节点测评大批量任务测试中的最佳GPU卡数调整集群测试的配置文件。  
    根据实际情况调整config/ClusterMode/config.yaml中以下内容：
    - partition：待测试的队列
    - nodelist：待测试的节点列表，用逗号分隔
    - gpus_per_node：单个节点GPU卡的数量
    - cpus_per_node：单个节点的CPU核心数量
    - multitasks_count：每款计算软件大批量任务测试时，提交的任务数量，默认为节点数量*50
    - sponge_gpus_per_task：SPONGE软件测试时，每个计算任务分配的GPU卡的数量，默认为1
    - gromacs_gpus_per_task：GROMACS软件测试时，每个计算任务分配的GPU卡的数量，默认为1
    - amber_gpus_per_task：AMBER软件测试时，每个计算任务分配的GPU卡的数量，默认为1
    - dsdp_gpus_per_task：DSDP软件测试时，每个计算任务分配的GPU卡的数量，默认为1  
    
    （以上软件的gpus_per_task参数需要根据单节点测评报告中的大批量任务测试的最佳并行模式进行调整）
    - sponge_singlenode_score：SPONGE软件在单节点深度测评中的得分
    - gromacs_singlenode_score：GROMACS软件在单节点深度测评中的得分
    - amber_singlenode_score：AMBER软件在单节点深度测评中的得分
    - dsdp_singlenode_score：DSDP软件在单节点深度测评中的得分
    - node_exporter_port：计算节点监控软件的端口
    - gpu_exporter_port：计算节点GPU监控软件的端口
    - server_ip：监控server节点的IP地址
    - server_port：监控server节点的端口
    - conda_path：conda的安装路径（如果DSDP不用独立的conda环境，请在配置文件中删除这一行）
    - dsdp_env：DSDP软件使用的conda环境名称（如果DSDP不用独立的conda环境，请在配置文件中删除这一行）

    然后在此目录下执行：  
    snakemake -s ClusterMode.snakefile  
    输出名为ClusterMode_report.md的文件，为单节点测评模式的详细测试报告。
