<div align="center">
  <img src="assets/branding/lmsv-logo.svg" alt="LMSV Logo" width="360" />
  <br />
  <h1>MindSpore Large Model Networks Structure Variantions</h1>

围绕大模型训练一致性验证与变异测试构建的自动化工具集  
统一入口、任务编排、结果归档、历史分析、单轮复现、内置 WebUI

</div>

<div align="center">

[快速开始](#quick-start) ·
[配置说明](#configuration) ·
[使用指引](#usage-guide) ·
[输出结果](#outputs) ·
[代码结构](#architecture) ·
[排查路径](#troubleshooting)

</div>

---

## 项目概览

> 这份 README 分成两条阅读路径：
>
> - 使用者视角：怎么部署、怎么配置、怎么执行、怎么查看结果、怎么复现
> - 开发者视角：入口在哪、目录负责什么、任务链路怎样串起来

LMSV 当前已经将原先较分散的脚本工作流重构为以 Python 为主的统一入口，覆盖以下能力：

| 能力 | 说明 |
| --- | --- |
| Task1 | LLM Model Mutation（整网泛化变异测试） |
| Task2 | LLM Within-Module Mutation（模块内组件泛化测试） |
| Task3 | LLM Inter-Module Mutation（模块间泛化组合变异测试） |
| Task4 | Multimodal Inter-Module Mutation（多模态模块间组合变异测试） |
| Task5 | Multimodal Within-Module Mutation（多模态模块内组件变异测试） |
| Task6 | Multimodal Model Mutation and Validation（多模态整网变异和验证） |
| 运行链路 | PTA / MSA / MindFormer 多运行链路编排 |
| 结果管理 | 运行日志归档、历史结果分析、失败复现 |
| 使用方式 | 命令行配置生成器与内置 WebUI |

### 你可以用它做什么

- 发起 Task1~Task6（整网、模块内、模块间、多模态模块间、多模态模块内、多模态整网）任务
- 自动生成运行配置并统一调度 PTA / MSA / MF
- 精确归档脚本、日志、权重和变异输入
- 对历史 `output/` 重新分析并生成报告
- 从历史任务中抽取单轮现场进行复现

---

## 目录

- [核心概念](#core-concepts)
- [环境准备](#environment)
- [快速开始](#quick-start)
- [配置说明](#configuration)
- [详细使用指引](#usage-guide)
- [输出目录与结果解读](#outputs)
- [代码结构介绍](#architecture)
- [任务执行链路总览](#workflow)
- [常见排查路径](#troubleshooting)
- [建议阅读顺序](#reading-order)
- [后续可继续完善的方向](#roadmap)

### 按场景继续阅读

README 只保留主线信息；如果你正好在处理某个专项问题，下面这些文档最值得继续看：

- 架构与入口链路：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Task1 使用与扩展：[docs/task1.md](docs/task1.md)、[docs/TASK1_MODEL_EXTENSION.md](docs/TASK1_MODEL_EXTENSION.md)
- Task3 专项说明：[docs/TASK3.md](docs/TASK3.md)
- WebUI 设计与接口：[docs/webui.md](docs/webui.md)
- Task1/2/3 门禁测试：[docs/TASK2_GATE_GUIDE.md](docs/TASK2_GATE_GUIDE.md)
- Task6 完整手册：[docs/task6.md](docs/task6.md)
- 新任务接入方式：[docs/how-to-add-a-new-task.md](docs/how-to-add-a-new-task.md)

---

## <a id="core-concepts"></a>核心概念

为方便交流，仓库内默认采用以下术语：

| 术语 | 含义 |
| --- | --- |
| `run` | 一次训练执行，通常由一条 `msrun` 或 `torchrun` 启动 |
| `test` | 一次完整验证，通常包含一组对应的 PTA / MSA / MF 运行 |
| `task` | 一次完整的 LMSV 任务，由 `do.py` 启动，包含多轮 `test` |
| `iteration` / `iter` | 单个 task 中某一轮测试的编号 |
| `step` | 单次 run 中的训练步数，是日志和精度对齐时的最小单位 |

---

## <a id="environment"></a>环境准备

### 基础依赖

请先确保当前机器满足以下前提：

- 已安装 `conda`，且可在 `bash` 中直接使用
- 若 Task1/Task2/Task3/Task4/Task5 启用多机：主节点需可用 `ssh`、`rsync`，并与从节点配置好免密登录
- 已准备好 PTA、MSA 对应代码仓
- 已准备好训练/验证数据集
- 当前 Python 环境已安装本仓库依赖

项目最小依赖见 [requirements.txt](requirements.txt)：

```bash
pip install -r requirements.txt
```

这些依赖主要用于：

- 配置读写与 YAML 处理
- 结果分析
- WebUI / 报告数据加工

### 推荐 conda 环境

仓库默认假设你至少有两个 conda 环境：

- `PTA_NAME`：运行 PTA 侧训练 / 保存逻辑
- `MSA_NAME`：运行 MSA 侧加载 / 验证逻辑

如果对比模式使用 `pta_mf`，还会用到：

- `mindf_py311`：默认的 MF 环境名，可在复现逻辑或模板中调整

### Task6 一键安装 Conda 环境（推荐）

如果你要运行 **Task6**（或多模态相关任务），本项目已提供包含全部手动修改的导出环境，支持一键安装。

**本仓库包含：**
- `task6_conda_envs_export/mindspeed.tar.gz` — PTA 完整环境
- `task6_conda_envs_export/msadapter.tar.gz` — MSA 完整环境（含 transformers 签名适配等手动修改）
- `task6_conda_envs_export/install_envs.sh` — 一键安装脚本

**从零开始完整部署（新机器/环境重建）：**

```bash
# 1. 前置条件检查
ls /usr/local/Ascend/ascend-toolkit/set_env.sh   # CANN
ls /shared/mindspeed-mm/MindSpeed-MM             # MindSpeed-MM
ls /data2/dataset                                # 数据集

# 2. 删除旧环境（如需重建）
conda remove --name mindspeed --all -y 2>/dev/null || true
conda remove --name msadapter --all -y 2>/dev/null || true
conda remove --name msa-m --all -y 2>/dev/null || true
conda remove --name ptaa --all -y 2>/dev/null || true

# 3. 安装新环境
cd task6_conda_envs_export
bash install_envs.sh ~/conda_envs

# 4. 配置 config.json（使用绝对路径）
cat > config.json << 'EOF'
{
  "task_type": 6,
  "PTA_NAME": "mindspeed",
  "MSA_NAME": "msadapter",
  "MINDSPEED_MM_PATH": "/mindspeed-mm",
  "DATASET_ROOT": "/data2/dataset",
  "SAVE_ABNORMAL_WEIGHTS": true,
  "tasks": {
    "6": {
      "MODEL_NAME": "internvl3",
      "TOTAL_ITER": 1,
      "MUTNM": 2,
      "COMPARE_MODE": "pta_msa",
      "TRAIN_ITER": 2,
      "BASE_SEED": 43,
      "PTA_MAX_RUNTIME": 3000,
      "MSA_MAX_RUNTIME": 3000
    }
  }
}
EOF

# 5. 执行
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./lmsv do
```

安装完成后：
- 物理路径：`~/conda_envs/mindspeed` 和 `~/conda_envs/msadapter`
- 已自动注册到 conda，可直接使用 `conda activate mindspeed` / `conda activate msadapter`

**注意：** 这些导出环境已内置所有手动补丁（如 `transformers` 兼容性修复、`libstdc++.so.6` 路径自动适配、`msadapter` bfloat16 fallback 等），新机器无需再手动改任何环境脚本即可直接运行。

### 外部路径准备

运行前通常需要准备以下路径，并写入配置：

- 原始数据集：默认放到 `assets/datasets/`，由工具在任务启动前自动发现并预处理
- `MINDSPEED_MM_PATH`：MindSpeed-MM 工作区根目录（兼容旧版 `PTA_PATH` / `MSA_PATH`，自动推导 MindSpeed-MM 子目录）
- `PTA_NAME`：PTA conda 环境名称
- `MSA_NAME`：MSA conda 环境名称

---

## <a id="quick-start"></a>快速开始

### 最快方式

直接执行：

```bash
./lmsv
```

这会按顺序完成两件事：

1. 调起配置生成器 `python genconf.py`
2. 读取 `config.json` 并执行 `python do.py`

### 命令行入口

[lmsv](lmsv) 是统一入口脚本，可用命令如下：

```text
Usage: lmsv [command]

Commands:
  webui   Run web UI (python webui.py)
  conf    Generate config (python genconf.py)
  do      Execute task (python do.py)
  test    Run task1/2/3 pre-merge gate tests (python tests/gate/task2_gate.py)
  repro   Reproduce a single run (python repro.py)
  analyze Regenerate analysis (python analyze.py)
  help    Show this help message

No command: run conf + do
```

常见用法：

```bash
./lmsv conf
./lmsv do
./lmsv test --task 2 --list
./lmsv test --task 1 --dry-run
./lmsv test --task 3 --dry-run
./lmsv test --task 2 --cases qwen3-main,deepseekv3-main
./lmsv webui
./lmsv analyze --latest
./lmsv repro
```

Task1/2/3 门禁测试说明：

- 详细文档：`docs/TASK2_GATE_GUIDE.md`

- 用例清单文件：
  - `tests/gate/task1_cases.json`
  - `tests/gate/task2_cases.json`
  - `tests/gate/task3_cases.json`
- 配置格式尽量贴近原始 `config.json`：
  - `common.env_name`：默认 conda 环境名
  - `base_config_patch`：公共默认配置补丁（例如 `task_type`、`tasks.<task>` 的公共字段）
  - `cases[*].config_patch`：单条用例覆盖项
- 默认门禁模式为 `COMPARE_MODE=pta_msa`
- 开关方式：每条用例都有 `enabled: true/false`
- 执行 `./lmsv test` 时，脚本会按用例逐个临时写入 `config.json` 并调用 `python do.py`（或 `conda run -n <env> python do.py`）
- 测试结束后会自动恢复原始 `config.json`
- 常用参数：
  - `--task {1|2|3}`：选择门禁任务，默认 `2`
  - `--list`：仅列出用例
  - `--dry-run`：只展示本次将运行哪些用例
  - `--limit N`：仅执行前 N 个用例
  - `--cases a,b,c`：只执行指定 case id
  - `--env-name xxx`：覆盖全部用例环境名

### 一眼看懂工作流

```text
生成配置 -> 执行任务 -> 归档输出 -> 生成分析 -> 定位问题 / 单轮复现
```

---

## <a id="configuration"></a>配置说明

### 配置文件来源

| 文件 | 作用 |
| --- | --- |
| [config.json.example](config.json.example) | 默认配置模板 |
| `config.json` | 实际运行配置 |
| [genconf.py](genconf.py) | 交互式配置生成器 |

如果 `config.json` 不存在，[do.py](do.py) 会自动用示例文件创建一份。

### 全局配置字段

所有任务共享以下字段：

| 字段 | 含义 |
| --- | --- |
| `task_type` | 任务类型，取值为 `1` / `2` / `3` / `4` / `5` / `6` |
| `PTA_NAME` | PTA conda 环境名称 |
| `MF_NAME` | MF conda 环境名称（使用 `COMPARE_MODE=pta_mf` 时需要） |
| `MSA_NAME` | MSA conda 环境名称 |
| `MINDSPEED_MM_PATH` | MindSpeed-MM 安装路径 |
| `MULTI_NODE` | 多机启动配置；推荐写在对应 `tasks.<task_id>.MULTI_NODE` 中，由主节点通过 ssh 直接拉起从节点 |
### 任务类型与专属参数

#### Task1：LLM Model Mutation（整网泛化变异测试）

对应 `config.tasks["1"]`，关键字段包括：

| 字段 | 含义 |
| --- | --- |
| `MODEL_NAME` | 单模型名，例如 `qwen2` |
| `TOTAL_ITER` | 总迭代轮次 |
| `PTA_MAX_RUNTIME` | PTA 单次最长执行时间，秒 |
| `MSA_MAX_RUNTIME` | MSA 单次最长执行时间，秒 |
| `LOG_INIT_WAIT` | MSA 日志初始化等待时间，秒 |
| `LOG_STABLE_THRESHOLD` | MSA 日志稳定判定阈值，秒 |
| `COMPARE_MODE` | 对比模式，支持 `pta_msa` 或 `pta_mf` |
| `ENABLE_WEIGHT_CONVERT` | 仅在 `COMPARE_MODE=pta_mf` 时生效，控制是否执行 PTA->MF 权重转换 |
| `ENABLE_MF_WEIGHT_LOAD` | 仅在 `COMPARE_MODE=pta_mf` 时生效，控制是否将权重加载到 MF（为 `false` 时跳过转换和加载） |
| `BASE_SEED` | 基础随机种子 |
| `MUTNM` | 每轮变异参数数量 |
| `SAVE_STEPS` | PTA SAVE 模式训练步数 |
| `LOAD_STEPS` | PTA LOAD 模式训练步数 |

#### Task2：LLM Within-Module Mutation（模块内组件泛化测试）

对应 `config.tasks["2"]`，关键字段包括：

| 字段 | 含义 |
| --- | --- |
| `MODELS` | 模型列表 |
| `SUBMODULES` | 子模块编号列表，需与 `MODELS` 一一对应 |
| `TOTAL_ITER` | 总迭代轮次 |
| `PTA_MAX_RUNTIME` | PTA 单次最长执行时间 |
| `MSA_MAX_RUNTIME` | MSA 单次最长执行时间 |
| `LOG_INIT_WAIT` | MSA 日志初始化等待时间 |
| `LOG_STABLE_THRESHOLD` | MSA 日志稳定判定阈值 |
| `BASE_SEED` | 基础随机种子 |
| `MUTNM` | 每轮变异参数数量 |
| `SAVE_STEPS` | PTA SAVE 模式训练步数，默认 `1` |
| `LOAD_STEPS` | PTA LOAD / 对比模式训练步数，默认 `15` |
| `COMPARE_MODE` | 对比模式，支持 `pta_msa` 或 `pta_mf` |
| `MF_ARGS_PATH` | MF 侧参数模板路径（传给 `load_and_forward_submodule.py` 的 `--args_path`） |
| `ENABLE_MF_WEIGHT_LOAD` | 仅在 `COMPARE_MODE=pta_mf` 时生效，是否把共享权重转换并加载到 MF |

说明：

- `SUBMODULES` 的取值范围在配置生成器中限制为 `0~10`
- `MODELS` 与 `SUBMODULES` 数量必须一致

#### Task3：LLM Inter-Module Mutation（模块间泛化组合变异测试）

对应 `config.tasks["3"]`，关键字段包括：

| 字段 | 含义 |
| --- | --- |
| `MODELS` | 参与组合变异的模型列表 |
| `TOTAL_ITER` | 变异轮次 |
| `PTA_MAX_RUNTIME` | PTA 单次最长执行时间 |
| `MSA_MAX_RUNTIME` | MSA 单次最长执行时间 |
| `LOG_INIT_WAIT` | MSA 日志初始化等待时间 |
| `LOG_STABLE_THRESHOLD` | MSA 日志稳定判定阈值 |
| `MAX_MUTATION_WAIT` | 变异产物等待时间 |
| `BASE_SEED` | 基础随机种子 |
| `MUTNM` | 每轮变异参数数量 |
| `SAVE_STEPS` | PTA SAVE 模式训练步数，默认 `1` |
| `LOAD_STEPS` | PTA LOAD / 对比模式训练步数，默认 `15` |
| `COMPARE_MODE` | 对比模式，支持 `pta_msa` 或 `pta_mf` |
| `MF_ARGS_PATH` | MF 侧参数模板路径（传给 `load_and_forward_graph.py` 的 `--args_path`） |
| `ENABLE_MF_WEIGHT_LOAD` | 仅在 `COMPARE_MODE=pta_mf` 时生效，是否把共享权重转换并加载到 MF |

#### Task1/Task2/Task3：多机启动配置（`MULTI_NODE`）

Task1、Task2、Task3 的多机多卡模式已经和 Task5 对齐：只在主节点配置 `MULTI_NODE`，从节点不再需要准备 `config.json`，也不需要手动执行 `./lmsv slave`。主节点会通过免密 `ssh` 直接在从节点拉起对应阶段，并用 `rsync` 分发脚本、权重、图结构和回收 `msrun_log/` 等产物。

推荐流程：

1. 在主节点的 `config.json` 中，把 `MULTI_NODE` 写入当前任务的 `tasks.<task_id>` 配置。
2. 确认主节点可以 `ssh <HOST>` 免密登录每台从节点。
3. 确认从节点的 `LMSV_PATH`、PTA/MSA/MF 环境名和代码路径可用。
4. 只在主节点执行 `./lmsv do`。

Task1/2/3 会自动计算本机和从节点卡数，注入 `MASTER_ADDR`、`MASTER_PORT`、`NNODES`、`NODE_RANK`、`WORLD_SIZE` 等分布式参数。旧版 `CLUSTER` / `./lmsv slave` 入口仍保留兼容，但新配置建议统一使用 `MULTI_NODE`。

#### Task4：Multimodal Inter-Module Mutation（多模态模块间组合变异测试）

对应 `config.tasks["4"]`，关键字段包括：

| 字段 | 含义 |
| --- | --- |
| `TOTAL_ITER` | 总迭代轮次 |
| `PTA_MAX_RUNTIME` | PTA 单次最长执行时间 |
| `MSA_MAX_RUNTIME` | MSA 单次最长执行时间 |
| `LOG_INIT_WAIT` | MSA 日志初始化等待时间 |
| `LOG_STABLE_THRESHOLD` | MSA 日志稳定判定阈值 |
| `COMPARE_MODE` | 对比模式，支持 `pta_msa` 或 `pta_mf` |
| `SAVE_STEPS` | PTA SAVE 模式训练步数，默认 `1` |
| `RUN_STEPS` | PTA/MSA/MF 训练步数，默认 `20` |
| `ENABLE_MF_WEIGHT_LOAD` | 仅在 `COMPARE_MODE=pta_mf` 时生效，是否把共享权重转换并加载到 MF |
| `SHARED_WEIGHT_TMP_ROOT` | 共享权重中间产物目录 |

#### Task5：Multimodal Within-Module Mutation（多模态模块内组件变异测试）

对应 `config.tasks["5"]`，关键字段包括：

| 字段 | 含义 |
| --- | --- |
| `TOTAL_ITER` | 总迭代轮次 |
| `PTA_MAX_RUNTIME` | PTA 单次最长执行时间 |
| `MSA_MAX_RUNTIME` | MSA 单次最长执行时间 |
| `LOG_INIT_WAIT` | MSA 日志初始化等待时间 |
| `LOG_STABLE_THRESHOLD` | MSA 日志稳定判定阈值 |
| `COMPARE_MODE` | 对比模式，支持 `pta_msa` 或 `pta_mf` |
| `SAVE_STEPS` | PTA SAVE 模式训练步数，默认 `1` |
| `RUN_STEPS` | PTA/MSA/MF 训练步数，默认 `20` |
| `MUTATE_STEPS` | 每轮内部变异步数，默认 `10` |
| `MODULE_TYPE` | 组件类型过滤，默认 `all` |
| `ENABLE_MF_WEIGHT_LOAD` | 仅在 `COMPARE_MODE=pta_mf` 时生效，是否把共享权重转换并加载到 MF |
| `SHARED_WEIGHT_TMP_ROOT` | 共享权重中间产物目录 |

#### Task1/Task2/Task3/Task4/Task5：多机启动配置（`MULTI_NODE`）

Task1 到 Task5 支持在 `pta_msa` 链路中开启多机执行，核心行为如下：

- Task1/2/3 会按阶段把脚本、共享权重、图结构或配置同步到各从节点
- Task4/5 的 `pta mutate`、`pta save` 保持在主节点执行，`pta save` 完成后同步当前迭代目录
- `pta load` / `msa load` / `mf` 阶段会在主从节点并行启动
- 主节点同样会注入多机参数：`--master-addr`、`--nnodes`、`--node-rank 0`
- 从节点通过 `ssh` 远程执行；若配置容器，则通过 `docker exec` 进入容器执行

`config.tasks["1"]` 到 `config.tasks["5"]` 内可配置：

| 字段 | 含义 |
| --- | --- |
| `MULTI_NODE.ENABLED` | 是否启用多机 |
| `MULTI_NODE.MASTER_ADDR` | 主节点地址，传给训练脚本的 `--master-addr` |
| `MULTI_NODE.MASTER_PORT` | 可选，Task1/2/3 训练 master port；不填默认 `6000` |
| `MULTI_NODE.NNODES` | 节点总数（实际会按 `OTHER_NODES` 自动校正） |
| `MULTI_NODE.OTHER_NODES` | 从节点列表 |
| `OTHER_NODES[].HOST` | 从节点 SSH 地址（可含用户名，如 `user@host`） |
| `OTHER_NODES[].SSH_PORT` | SSH 端口，默认 `22` |
| `OTHER_NODES[].LMSV_PATH` | 从节点上的项目根目录（用于路径映射和目录同步） |
| `OTHER_NODES[].PTA_NAME` | 从节点 PTA conda 环境名 |
| `OTHER_NODES[].MSA_NAME` | 从节点 MSA conda 环境名 |
| `OTHER_NODES[].PTA_PATH` | 从节点 PTA 代码路径（导出到 `PTAPATH`） |
| `OTHER_NODES[].MSA_PATH` | 从节点 MSA 代码路径（导出到 `MSAPATH`） |
| `OTHER_NODES[].MF_NAME` | 可选，从节点 MF conda 环境名；不填使用主节点全局 `MF_NAME` |
| `OTHER_NODES[].NPUS_PER_NODE` | 可选，从节点本地卡数；不填由 ssh 预检自动探测 |
| `OTHER_NODES[].HAS_CONTAINER` | 从节点是否在容器内执行 |
| `OTHER_NODES[].CONTAINER_NAME` | 容器名（`HAS_CONTAINER=true` 时必填） |

多机示例（Task1/2/3/4/5 均适用）：

```json
{
  "task_type": 4,
  "PTA_NAME": "mindspeed",
  "MSA_NAME": "msadapter",
  "PTA_PATH": "/data/pta",
  "MSA_PATH": "/data/msa",
  "tasks": {
    "4": {
      "TOTAL_ITER": 5,
      "RUN_STEPS": 20,
      "COMPARE_MODE": "pta_msa",
      "MULTI_NODE": {
        "ENABLED": true,
        "MASTER_ADDR": "10.0.0.10",
        "NNODES": 2,
        "OTHER_NODES": [
          {
            "HOST": "root@10.0.0.11",
            "SSH_PORT": 22,
            "LMSV_PATH": "/data/yd/lm-sv",
            "PTA_NAME": "mindspeed",
            "MSA_NAME": "msadapter",
            "PTA_PATH": "/data/pta",
            "MSA_PATH": "/data/msa",
            "HAS_CONTAINER": true,
            "CONTAINER_NAME": "lmsv-worker"
          }
        ]
      }
    }
  }
}
```

多机模式常见前置检查：

- 主节点执行用户需能直接 `ssh <HOST>` 免密登录从节点
- 主节点需安装 `ssh` 与 `rsync`（可通过 `LMSV_SSH_BIN` / `LMSV_RSYNC_BIN` 指定自定义路径）
- 从节点若在容器中执行，需确保容器内可用 `conda` 与训练依赖
- 从节点的 `LMSV_PATH` 必须与仓库结构对应，保证路径映射正确

#### Task6：Multimodal Model Mutation and Validation（多模态整网变异和验证）

对应 `config.tasks["6"]`，专门针对多模态大模型（视觉-语言模型、视频生成模型）进行自动化变异测试和跨框架对比验证：

| 字段 | 含义 |
| --- | --- |
| `MODEL_NAME` | 模型名称，支持 `internvl3` / `qwenvl` / `opensora` / `cogvideox` |
| `TOTAL_ITER` | 总迭代轮次（有效突变次数） |
| `MUTNM` | 每轮变异参数数量 |
| `TRAIN_ITER` | 每轮训练/推理步数（兼容旧版 `SAVE_STEPS` / `TRAIN_ITERS`） |
| `COMPARE_MODE` | 对比模式，支持 `pta_msa` 或 `pta_mf` |
| `BASE_SEED` | 基础随机种子，控制每轮变异的确定性（默认 `43`） |
| `PTA_MAX_RUNTIME` | PTA 单次最长执行时间，秒（默认 `3000`） |
| `MSA_MAX_RUNTIME` | MSA 单次最长执行时间，秒（默认 `3000`） |

**支持模型状态：**

| 模型 | 类型 | PTA状态 | MSA状态 | 备注 |
|------|------|---------|---------|------|
| InternVL3-8B | 训练 | ✅ 正常 | ✅ 正常 | 基准模型，精度差异约20% |
| QwenVL2.5-7B | 推理 | ✅ 正常 | ❌ 失败 | MSA环境InnerInplaceIndexPut shape mismatch（框架问题） |
| OpenSora1.2 | 推理 | ✅ 正常 | ❌ 失败 | MSA环境safetensors加载错误（框架问题） |
| CogVideoX-5B | 训练 | ✅ 正常 | ✅ 正常 | 经环境修复后MSA已可正常运行 |

**模型类型区分：**
- **训练模型**（InternVL3、CogVideoX）：有loss输出，需检查loss、显存、时间
- **推理模型**（QwenVL、OpenSora）：无loss输出，判断成功标准是返回码是否为0

**Task6 执行示例：**

```bash
# 方式1: 使用 lmsv 统一入口（与Task1-5完全一致）
./lmsv conf  # 选择 Task6，交互式配置
./lmsv do

# 方式2: 直接修改配置文件执行
# 按需修改 config.json 中的模型、迭代次数等参数
./lmsv do
```

**四模型快速切换：**

四个模型共用同一 `config.json` 和 `mutable_params_pool.yaml`，只需修改 `MODEL_NAME`：

| 模型 | `MODEL_NAME` | 建议 `TRAIN_ITER` |
|------|-------------|-------------------|
| InternVL3-8B | `internvl3` | `2~5` |
| QwenVL2.5-7B | `qwenvl` | `1~2` |
| OpenSora1.2 | `opensora` | `1~2` |
| CogVideoX-5B | `cogvideox` | `2~5` |

`mutable_params_pool.yaml` 自动在项目根目录加载，四个模型共用同一变异参数池。

**Task6 核心特性：**

1. **增量变异机制**：每轮基于前一轮变异结果继续变异，而非每次都从基础配置开始
2. **异常回滚机制**：PTA执行失败视为无效突变，自动撤销并回滚；MSA失败视为有效突变（发现框架问题），记录问题并继续
3. **双环境对比**：PTA (PyTorch Ascend) vs MSA (MindSpore Adapter)
4. **YAML配置化**：变异参数池通过YAML文件配置，无需修改代码

Task6 当前的运行语义还可以再补一句：

- InternVL3、CogVideoX 目前按“训练模型”处理，核心对齐指标包含 loss / 显存 / 时间
- QwenVL、OpenSora 目前按“推理模型”处理，重点看执行成功率、显存、耗时和错误类型，而不是 loss

**变异参数池配置（YAML）：**

Task6的变异参数池支持通过YAML文件配置，无需修改Python代码：

```bash
# 默认配置文件
mutable_params_pool.yaml

# 自定义配置文件路径
export MUTABLE_PARAMS_POOL_PATH=/path/to/your/mutable_params_pool.yaml
```

YAML格式示例：
```yaml
# 数值型参数（范围变异）
numeric_params:
  mlp_ratio:
    min_val: 2.0
    max_val: 8.0
    min_factor: 0.7
    max_factor: 1.5

# 枚举型参数（离散值变异）
enum_params:
  activation_func:
    - gelu
    - silu
```

**Task6 专属文档：**

- `docs/task6.md` - 完整使用文档
- `docs/environment-modifications.md` - **从零开始环境搭建指南（含一键安装脚本）**
- `docs/task6_skill.md` - 开发经验与避坑指南（含14个避坑点）
- `docs/task6_model_handling.md` - 四模型处理逻辑详解
- `docs/task6_statistics.md` - 统计规则说明

### Task1/Task2/Task3 中 MF 运行方式

统一触发条件：

- Task1/Task2/Task3 通过 `COMPARE_MODE=pta_mf` 开启 MF 链路
- 开启后会关闭 MSA 对比分支，执行对变为 `PTA + MF`
- MF 环境来自全局 `MF_NAME`（兼容旧字段时会做兜底）

#### Task1（LLM Model Mutation）中的 MF 链路

每轮大致顺序是：

1. 先跑 PTA-SAVE / PTA-LOAD，拿到 PTA 侧权重与日志
2. 基于本轮 PTA 脚本生成 MF YAML（`utils.runtime.mf_converter`）
3. 若 `ENABLE_MF_WEIGHT_LOAD=true` 且 `ENABLE_WEIGHT_CONVERT=true`，执行 PTA->MF 转换（`pth -> npz -> ckpt`）
4. 若 `ENABLE_MF_WEIGHT_LOAD=true`，回填 MF YAML 的 `load_checkpoint`
5. 用 `scripts/runtime/mf_start.sh` 启动 MF 训练
6. 等待 MF 完成标记并读取 `res/training_log_mf/training_log-<iter>.csv`
7. 做 PTA vs MF loss 对齐（使用 `MF_LOSS_TOLERANCE`）

当前 Task1 的 PTA->MF 对齐，README 层面建议重点记住这四类已落地能力：

- 结构参数对齐：如 `hidden_size`、`num_layers`、`num_attention_heads`、`ffn_hidden_size`、`head_dim`
- 并行与 batch 对齐：如 `world_size`、TP/PP/CP/EP、`micro_batch_size`、`global_batch_size`
- 训练语义对齐：如 `seed`、`lr`、`weight_decay`、warmup / decay 相关参数
- 权重链路对齐：通过 `convert.sh + convert_ckpt.py` 将 PTA 产物转换为 MF 可加载权重

如果要继续看“为什么现在这样对齐”以及具体映射范围，推荐直接看：

- [docs/PTA_MF_PRECISION_ALIGNMENT.md](docs/PTA_MF_PRECISION_ALIGNMENT.md)
- [docs/静态图整网变异对齐进展.md](docs/静态图整网变异对齐进展.md)

补充：当 `ENABLE_MF_WEIGHT_LOAD=false` 时，Task1 会跳过 MF 权重转换与加载，仅用于流程联通和基础回归检查。

补充：`MODEL_NAME=deepseekv3` 且 `COMPARE_MODE=pta_mf` 时，Task1 会强制关闭 MF 权重转换与加载（硬编码行为，忽略相关配置项）。

专项说明文档：`docs/TASK1_DEEPSEEKV3_PTA_MF_WEIGHT_POLICY.md`

#### Task2（LLM Within-Module Mutation）中的 MF 链路

每轮大致顺序是：

1. 先跑 PTA 侧子模块变异 + PTA-SAVE 产出共享权重
2. 可选执行共享权重转换（`ENABLE_MF_WEIGHT_LOAD=true` 时：`pth -> npz -> ckpt`）
3. 跑 PTA-LOAD 产出 PTA 对齐日志
4. 运行 `utils/runtime/mf_mutate_and_forward/load_and_forward_submodule.py`
5. 将 MF 结果写入 `res/submodule_execution_mf.csv`，并归档 `res/training_log_mf/training_log-<iter>.csv`
6. 对 PTA 与 MF 的同轮 loss 做一致性检查，必要时备份问题现场

补充：当 `ENABLE_MF_WEIGHT_LOAD=false` 时，Task2 会执行 MF 流程但不加载 PTA 共享权重，主要用于流程联通和基础回归检查。目前该功能仍然不完善，需要进一步进行适配

注：submodule中，mf主要支持0-2，7-10模块，3-9由于flash-attn限制，mf中未能找到合适实现进行对比，目前这些模块映射存在较大误差。

#### Task3（LLM Inter-Module Mutation）中的 MF 链路

每轮大致顺序是：

1. 先跑 PTA 模块间变异，等待 `mutating-<iter>.json` 等产物
2. 跑 PTA-SAVE 生成共享权重，按需做 `pth -> npz -> ckpt` 转换
3. 跑 PTA-LOAD 产出 PTA 对齐日志
4. 运行 `utils/runtime/mf_mutate_and_forward/load_and_forward_graph.py`
5. 将 MF 结果写入 `res/execution_mf.csv`，并归档 `res/training_log_mf/training_log-<iter>.csv`
6. 对 PTA 与 MF 的同轮 loss 做一致性检查，并在异常时备份权重与现场

补充：Task3 在 `COMPARE_MODE=pta_mf` 下同样是 `PTA + MF` 双链路，且支持通过 `MF_ARGS_PATH` 切换 MF 侧模板参数。

### 典型配置示例

```json
{
  "task_type": 1,
  "PTA_NAME": "mindspeed",
  "MSA_NAME": "msadapter",
  "MINDSPEED_MM_PATH": "/workspace/mm-new",  // 指向 workspace root，自动推导 MindSpeed-MM 子目录
  "tasks": {
    "1": {
      "MODEL_NAME": "qwen2",
      "TOTAL_ITER": 10,
      "PTA_MAX_RUNTIME": 3000,
      "MSA_MAX_RUNTIME": 3000,
      "COMPARE_MODE": "pta_msa",
      "BASE_SEED": 43,
      "MUTNM": 2,
      "SAVE_STEPS": 1,
      "LOAD_STEPS": 30
    }
  }
}
```

### MSA Profiler 使用说明

当 `COMPARE_MODE=pta_msa` 时，Task1 / Task2 / Task3 都支持对外部 MSA profiler 产物做离线分析。当前版本不再通过 `config.json`、`genconf.py` 或 WebUI 控制 profiler 开关，而是由用户在外部 MSA 训练链路中自行开启 profiling；LMSV 负责在迭代结束后自动发现产物并生成分析报告。

使用方式：

- 在外部 MSA 训练侧自行开启 profiler
- 正常执行 `python do.py` 或通过 WebUI 启动任务
- 确保 profiling 产物最终写入每轮迭代目录下的 `profiler/msa-load/`
- 任务结束后到 `analysis/msa-profiler/` 查看报告

结果产物：

- 原始 profiling 数据会归档到 `output/<run>/iters/iter_<n>/profiler/msa-load/`
- 离线分析摘要会生成到 `output/<run>/iters/iter_<n>/analysis/msa-profiler/summary.md`
- 结构化分析结果会生成到 `output/<run>/iters/iter_<n>/analysis/msa-profiler/summary.json`

当前离线分析会给出三层信息：

- `Intelligent Analysis`：整体状态、最高严重级别、总结结论、优先关注项
- `Findings`：问题判断、证据、建议动作
- `Advice`：面向调优的启发式建议

当前会结合逐 step 执行时间、波动、显存峰值和采样覆盖情况，给出一些“接近专家结论”的判断，例如：

- step 时间抖动较大时，判断可能存在数据准备、主机侧同步、图编译缓存或传输抖动
- 慢 step 明显时，判断可能存在热点算子、通信等待、重复构图或异常同步点
- 显存峰值偏高时，判断存在内存压力，并给出 micro-batch-size、重计算、seq-length 等调优建议
- 采样维度不足时，明确提示当前证据不足，并建议补充 CPU 或 memory profiling

---

## <a id="usage-guide"></a>详细使用指引

### 1. 生成配置

交互式生成：

```bash
./lmsv conf
```

或：

```bash
python genconf.py
```

[genconf.py](genconf.py) 会：

- 读取现有 `config.json` 作为默认值
- 先让你选择任务类型
- 再按对应任务类型收集参数
- 可选进入“高级配置”，修改运行超时、SAVE 步数等参数
- 最终写回新的 `config.json`

### 2. 执行任务

使用当前配置执行：

```bash
./lmsv do
```

或：

```bash
python do.py
```

[do.py](do.py) 的执行流程如下：

1. 确认 `config.json` 存在
2. 检查是否已有其他 `do.py` 进程在运行
3. 创建新的 `output/<时间戳>/`
4. 复制本次配置快照到输出目录
5. 设置日志路径与输出路径环境变量
6. 读取配置并导出运行环境变量
7. 根据 `task_type` 分派到 Task1 / Task2 / Task3 / Task4 / Task5 / Task6

### 3. 使用 WebUI

启动方式：

```bash
./lmsv webui
```

或：

```bash
python webui.py
```

[webui.py](webui.py) 是一个内置的轻量 HTTP 服务，主要提供：

- 配置编辑
- 任务启动
- 实时日志查看
- 历史 `output` 浏览
- 分析结果重生成入口

如果你希望给非命令行使用者提供操作界面，优先使用这个入口。

### 4. 历史结果重新分析

如果已经有历史 `output/`，但想重新生成 `analysis/`：

```bash
./lmsv analyze
./lmsv analyze --latest
./lmsv analyze --list
./lmsv analyze 2026-03-08-11-53-17
python analyze.py output/2026-03-08-11-53-17
```

[analyze.py](analyze.py) 支持：

- 交互式选择某个 output
- 直接指定 output 目录
- 直接选择最近一次 output
- 输出 JSON 格式结果
- 覆盖模型名、任务类型、计划轮次等元信息

适合以下场景：

- 分析脚本升级后，对旧任务补跑报告
- output 已保留，但 `analysis/` 缺失或不完整
- 需要重新整理 `summary.json` / `report.html`

### 5. 单轮复现

启动方式：

```bash
./lmsv repro
```

或：

```bash
python repro.py
```

[repro.py](repro.py) 用于从历史 output 中选择某次任务、某一轮迭代，以及对应的运行脚本进行复现，适合：

- 复查 PTA / MSA / MF 某一阶段异常
- 使用已归档脚本和权重重跑现场
- 对失败轮次做局部定位

### 6. 中断与残留进程清理

任务在运行中可能产生 `torchrun`、`msrun`、`pretrain_gpt.py` 等子进程。仓库内的 [utils/control/clean.py](utils/control/clean.py) 提供了残留清理逻辑，用于：

- 识别全局残留的训练相关运行进程
- 额外识别并清理占用 `6000` 端口的残留进程
- 强制清理残留训练进程
- 清理临时运行目录下的 HCCL / 分布式残留文件

如果你在中断任务后遇到环境未释放、端口或临时目录残留，优先从这里排查。

---

## <a id="outputs"></a>输出目录与结果解读

### output 总体结构

每次执行 [do.py](do.py) 都会创建一个新的输出目录：

```text
output/<时间戳>/
├── config.json
├── log.txt
├── analysis/
│   ├── report.html
│   ├── summary.md
│   ├── assets/
│   └── data/
│       ├── summary.json
│       ├── iterations.csv
│       └── issues.json
└── iters/
    ├── iter_1/
    ├── iter_2/
    └── ...
```

各目录含义：

| 路径 | 说明 |
| --- | --- |
| `config.json` | 本次任务的配置快照 |
| `log.txt` | 任务级总日志 |
| `analysis/` | 汇总分析结果与报告 |
| `iters/` | 逐轮归档的运行材料 |

### 单轮迭代目录

单个 `iters/iter_x/` 通常包含：

- `report.md`：单轮简版分析结论
- `status.json`：本轮各阶段状态
- `runtime_logs/`：运行时日志归档
- `msrun_log/`：`msrun` 日志快照
- `scripts/`：本轮生成或实际执行的脚本
- `weights/`：共享权重或权重备份
- `mutation_inputs/`：本轮变异输入
- `FAILED_FLAG` / `failure_info.txt`：失败时保留的核心现场
- `*.csv` / `*.json` / `*.txt`：运行过程中的结果快照

**注：关于日志，pta整网、模块间、模块内的日志均位于runtime_logs中，msa的启动日志在runtime_logs中，而schedule、worker日志均位于msrun_log中,对于mf而言，整网使用msrun启动，所以日志在msrun_log中，而模块间模块内使用python启动单卡，所以日志位于runtime_logs中**

### 运行期目录与归档目录的关系

README 里最容易混淆的是“运行时目录”和“最终归档目录”。当前代码的实际约定是：

- 运行中的中间产物主要落在仓库根目录下的 `tmp/<task>/runtime_workspace/`
- 仓库根目录下的 `res/`、`msrun_log/`、`pta/`、`ms/` 更像稳定入口或兼容层
- 任务结束后，关键文件会被精准归档到 `output/<时间戳>/iters/iter_x/`

也就是说：

- 排查“当前正在跑什么”，先看根目录运行时目录
- 排查“历史某轮到底跑了什么”，优先看 `output/` 中对应的 `iter_x`

---

## <a id="architecture"></a>代码结构介绍

这一节重点说明目录职责，不是简单把文件树抄一遍。

### 顶层入口层

仓库顶层是面向用户的直接入口：

| 路径 | 职责 |
| --- | --- |
| [lmsv](lmsv) | 统一命令入口，分发到 `conf` / `do` / `test` / `analyze` / `repro` / `webui` |
| [genconf.py](genconf.py) | 交互式生成 `config.json` |
| [do.py](do.py) | 主任务调度器，负责创建 output、导出环境、分派任务 |
| [analyze.py](analyze.py) | 历史 output 的分析重生成入口 |
| [repro.py](repro.py) | 基于归档结果做单轮复现 |
| [webui.py](webui.py) | 内置 WebUI 服务 |

理解这个项目时，建议从这 6 个文件开始。

### `utils/` 总体职责

[utils](utils) 是仓库核心逻辑所在，按照职责可分为 6 层：

| 子目录 | 主要职责 |
| --- | --- |
| [utils/control](utils/control) | 任务分派、守护、清理残留进程 |
| [utils/task](utils/task) | 三类任务的主流程实现与辅助方法 |
| [utils/runtime](utils/runtime) | 运行时脚本拼装、路径、模板、模型 / 变异执行支撑 |
| [utils/analyze](utils/analyze) | 任务结果分析、报告汇总、精度比对 |
| [utils/log](utils/log) | 日志输出 |
| [utils/replace](utils/replace) | 运行时 patch / hook，用于兼容或替换外部训练逻辑 |

<details>
<summary><strong>展开查看各层细分职责</strong></summary>

### 控制层：`utils/control/`

关键文件：

- [utils/control/protect.py](utils/control/protect.py)
- [utils/control/clean.py](utils/control/clean.py)

职责说明：

- `protect.py`
  - 接收 `task_type`
  - 忽略 `SIGTERM`
  - 将任务分发到 `task1` / `task2` / `task3` / `task4` / `task5`
- `clean.py`
  - 清理与当前仓库相关的残留训练进程
  - 清理运行时临时目录残留

可以把这层理解为“任务守护与生命周期管理层”。

### 任务层：`utils/task/`

关键文件：

- [utils/task/task1.py](utils/task/task1.py)
- [utils/task/task2.py](utils/task/task2.py)
- [utils/task/task3.py](utils/task/task3.py)
- [utils/task/task4.py](utils/task/task4.py)
- [utils/task/task5.py](utils/task/task5.py)
- [utils/task/task6.py](utils/task/task6.py)
- [utils/task/runtime_helpers.py](utils/task/runtime_helpers.py)
- [utils/task/data_helpers.py](utils/task/data_helpers.py)

职责说明：

- `task1.py`
  - 整网泛化变异测试主流程
  - 组织 mutation、PTA save/load、MSA 校验、可选 MF 校验
  - 归档脚本、日志、权重、分析结果
- `task2.py`
  - 模块内组件泛化测试主流程
  - 处理 `MODELS + SUBMODULES` 组合
  - 关注共享权重、模块内迭代、MSA 完成判定与精度对齐
  - 默认按当前可见卡数自动启用多卡；仅在显式传入 `TARGET_*` 并行参数时覆盖
- `task3.py`
  - 模块间组合变异测试主流程
  - 管理多模型 mutation 产物等待、PTA / MSA / MF 协同
  - 默认按当前可见卡数自动启用多卡；每轮仅归档 `msrun_log/` 目录，不再单独复制 `worker_x.log`
- `task4.py`
  - Multimodal Inter-Module Mutation 主流程
  - 管理多模态变异、PTA-SAVE/LOAD 与 MSA/MF 分支运行
  - 每轮输出统一状态文件，并接入 `task45_result` 分析
- `task5.py`
  - Multimodal Within-Module Mutation 主流程
  - 支持 `MODULE_TYPE` 过滤与 `MUTATE_STEPS` 控制变异步数
  - 每轮输出统一状态文件，并接入 `task45_result` 分析
- `task6.py`
  - Multimodal Model Mutation and Validation 主流程
  - 支持 InternVL3、QwenVL2.5、OpenSora、CogVideoX 四个模型
  - 实现 PTA/MSA 双环境自动化对比执行
  - 增量变异机制，每轮基于前一轮结果继续变异
  - 异常回滚机制，PTA 失败自动撤销并重新变异
  - 自动生成 Markdown/JSON/HTML 报告
- `runtime_helpers.py`
  - 各 task 复用的通用运行辅助
  - 包含 conda 激活片段、脚本写入、日志重定向、产物归档等
- `data_helpers.py`
  - CSV / JSON / 列表类数据处理
  - 子模块编号、布尔值、历史迭代结果等辅助逻辑

如果你要改“任务执行行为”，大多数修改都会落在这一层。

### 运行时支撑层：`utils/runtime/`

[utils/runtime](utils/runtime) 是项目里最重的一层，承担了很多真正和训练运行贴得很近的能力，主要包括：

- 路径与常量管理
- 模型配置适配
- 训练入口封装
- 变异图构建与前向验证
- MindFormer / MindSpore 相关适配
- 并行变异脚本与参数生成

里面可以再按功能理解成几组：

#### 1. 基础设施

- [utils/runtime/paths.py](utils/runtime/paths.py)
- [utils/runtime/constants.py](utils/runtime/constants.py)
- [utils/runtime/logger.py](utils/runtime/logger.py)
- [utils/runtime/common_utils.py](utils/runtime/common_utils.py)

负责统一路径、常量和底层辅助逻辑。

#### 2. 外部训练适配 / 启动

- [utils/runtime/pretrain_gpt.py](utils/runtime/pretrain_gpt.py)
- [utils/runtime/run_mindformer.py](utils/runtime/run_mindformer.py)
- [scripts/runtime/msrun_launcher.sh](scripts/runtime/msrun_launcher.sh)
- [scripts/runtime/mf_start.sh](scripts/runtime/mf_start.sh)
- [scripts/runtime/submodule_entry.py](scripts/runtime/submodule_entry.py)

负责把仓库内生成的参数转成外部训练系统可以执行的形式。

#### 3. 变异与前向执行

- [utils/runtime/mutate_and_forward](utils/runtime/mutate_and_forward)
- [utils/runtime/ms_mutate_and_forward](utils/runtime/ms_mutate_and_forward)
- [utils/runtime/mf_mutate_and_forward](utils/runtime/mf_mutate_and_forward)
- [utils/runtime/core](utils/runtime/core)

这部分是 LMSV 的核心技术区，主要处理：

- 图结构建模
- 变异策略执行
- 变异后网络的装载与前向
- 不同运行后端的统一适配

#### 4. 并行变异与脚本转换

- [utils/runtime/mutate_and_forward/parallel_mutate](utils/runtime/mutate_and_forward/parallel_mutate)

负责批量并行变异时的：

- YAML / 脚本参数转换
- 配置校验
- 变异脚本生成
- Megatron 训练脚本拼装

#### 5. 配置与模型模板

仓库里的静态资源大多在 `assets/`：

- [assets/runtime/model_config](assets/runtime/model_config)
- [assets/runtime/configs](assets/runtime/configs)
- [assets/runtime/mf_templates](assets/runtime/mf_templates)
- [assets/runtime/tokenizers](assets/runtime/tokenizers)

分别负责：

- 模型配置模板
- 结构 / 变异 schema
- MindFormer 模板
- tokenizer 与模型兼容资源

#### 6. Task6 多模态变异支持（`utils/runtime/mm_mutation/`）

Task6 专属的多模态配置变异模块：

- [utils/runtime/mm_mutation/mm_mutator.py](utils/runtime/mm_mutation/mm_mutator.py)
  - 多模态配置变异器，支持 InternVL3、QwenVL、OpenSora、CogVideoX
  - 实现增量变异机制，每轮基于前一轮结果继续变异
  - **支持 YAML 配置化**：从 `mutable_params_pool.yaml` 加载变异参数池
- [utils/runtime/mm_mutation/mutate_graph.py](utils/runtime/mm_mutation/mutate_graph.py)
  - 变异图执行入口，处理 JSON 配置的变异和应用
- [mutable_params_pool.yaml](mutable_params_pool.yaml)
  - Task6 变异参数池 YAML 配置文件
  - 定义数值型参数（范围变异）和枚举型参数（离散值变异）
  - 支持通过环境变量 `MUTABLE_PARAMS_POOL_PATH` 指定自定义配置

### 分析层：`utils/analyze/`

关键文件：

- [utils/analyze/manual.py](utils/analyze/manual.py)
- [utils/analyze/task1_result.py](utils/analyze/task1_result.py)
- [utils/analyze/task2_result.py](utils/analyze/task2_result.py)
- [utils/analyze/task3_result.py](utils/analyze/task3_result.py)
- [utils/analyze/task45_result.py](utils/analyze/task45_result.py)
- [utils/analyze/task6_result.py](utils/analyze/task6_result.py)
- [utils/analyze/precision.py](utils/analyze/precision.py)
- [utils/analyze/rules.py](utils/analyze/rules.py)

职责说明：

- 发现哪些 output 可以被重新分析
- 对不同任务类型生成汇总报告
- 对 PTA / MSA / MF 结果做精度差异定位
- 生成 `analysis/report.html`、`summary.md`、`data/*.json`

各分析模块具体职责：

- `task1_result.py` - Task1 LLM Model Mutation 结果分析
- `task2_result.py` - Task2 LLM Within-Module Mutation 结果分析
- `task3_result.py` - Task3 LLM Inter-Module Mutation 结果分析
- `task45_result.py` - Task4/Task5 Multimodal Module Mutation 结果分析
- `task6_result.py` - Task6 Multimodal Model Mutation 结果分析
  - 生成 Markdown/JSON/HTML 多格式报告
  - 统计 PTA/MSA 成功率、问题发现率等指标
  - 处理训练模型（有loss）和推理模型（无loss）的差异对比

如果你要改报表字段、统计逻辑、问题判定规则，优先看这一层。

### Patch 与兼容层：`utils/replace/`

关键文件：

- [utils/replace/sitecustomize.py](utils/replace/sitecustomize.py)
- [utils/replace/shared_weight_patch.py](utils/replace/shared_weight_patch.py)
- [utils/replace/training_log_patch.py](utils/replace/training_log_patch.py)

这层主要用于：

- 在运行时注入补丁
- 修改共享权重行为
- 修正训练日志输出路径或格式

它通常不会被终端用户直接调用，但对任务跑通很关键。

### `scripts/`、`assets/`、`mm/`、`legacy/` 的角色

#### `scripts/`

[scripts](scripts) 存放辅助脚本与模板：

- `runtime/`：运行时启动脚本
- `mutation/`：变异相关 shell 脚本
- `templates/`：预训练脚本模板
- `envset/`：环境设置脚本

#### `assets/`

[assets](assets) 存放相对稳定的静态资源：

- 模型 YAML
- tokenizer 文件
- mutation schema
- MindFormer 模板

#### `mm/`

[mm](mm) 是多模态 / 实验性能力区，目前更像独立实验场而不是主流程必经目录，包括：

- 多模态网络变异 demo
- 组合变异模板
- 预训练示例脚本

如果当前只关注主线的 Task1 / Task2 / Task3（或对应多模态链路 Task4 / Task5），可以先不深入这一块。

#### `legacy/`

[legacy](legacy) 用于保留历史版本代码和兼容材料。主流程已不依赖它，但当你需要对照旧版行为时很有价值。

</details>

---

## <a id="workflow"></a>任务执行链路总览

从宏观上看，一次任务的执行链路是：

```text
lmsv
  -> genconf.py（可选）
  -> do.py
     -> utils.control.protect.task()
        -> utils.task.task1/task2/task3/task4/task5/task6.main()
           -> utils.task.runtime_helpers / data_helpers
           -> utils.runtime/*
           -> 外部 PTA / MSA / MF 执行
           -> 归档到 output/<timestamp>/
           -> utils.analyze.* 生成报告
```

如果你在排查问题，可以按这个顺序缩小范围：

1. 入口参数是否正确
2. 任务分派是否正确
3. 运行脚本是否正确生成
4. 外部 PTA / MSA / MF 是否实际启动
5. 日志与权重是否正确归档
6. 分析阶段是否正确读取 output

---

## <a id="troubleshooting"></a>常见排查路径

### 任务没跑起来

优先检查：

- `config.json` 是否存在且字段完整
- `MINDSPEED_MM_PATH` 是否有效（支持指向 workspace root 或 MindSpeed-MM 代码目录，兼容旧版 `PTA_PATH` / `MSA_PATH`）
- `PTA_NAME` / `MSA_NAME` 对应 conda 环境是否存在
- 是否已有其他 `do.py` 进程在运行

### 某轮失败但总任务还在继续

优先查看：

- `output/<时间戳>/iters/iter_x/status.json`
- `output/<时间戳>/iters/iter_x/runtime_logs/`
- `output/<时间戳>/iters/iter_x/msrun_log/`
- `output/<时间戳>/iters/iter_x/scripts/`
- `output/<时间戳>/iters/iter_x/FAILED_FLAG`
- `output/<时间戳>/iters/iter_x/failure_info.txt`

说明：

- `task2/task3` 的 MSA worker 日志统一看 `iter_x/msrun_log/`
- 不再额外在 `iter_x/` 根目录单独备份 `worker_0.log` / `worker_1.log`

### analysis 缺失或过期

直接执行：

```bash
./lmsv analyze --latest
```

或：

```bash
python analyze.py <output目录>
```

### 想复现历史现场

直接执行：

```bash
./lmsv repro
```

然后从历史 output 和 iter 中选择目标轮次。

---

## <a id="reading-order"></a>建议阅读顺序

如果你是第一次接手这个仓库，建议按下面顺序阅读：

1. [README.md](README.md)
2. [lmsv](lmsv)
3. [do.py](do.py)
4. [genconf.py](genconf.py)
5. [utils/control/protect.py](utils/control/protect.py)
6. [utils/task/task1.py](utils/task/task1.py)
7. [utils/task/task2.py](utils/task/task2.py)
8. [utils/task/task3.py](utils/task/task3.py)
9. [utils/task/task4.py](utils/task/task4.py)
10. [utils/task/task5.py](utils/task/task5.py)
11. [utils/task/task6.py](utils/task/task6.py)
12. [utils/analyze/manual.py](utils/analyze/manual.py)
13. [webui.py](webui.py)

这样能最快建立“入口 -> 调度 -> 任务执行 -> 结果分析”的整体认知。

---

## <a id="roadmap"></a>后续可继续完善的方向

当前 README 已覆盖主流程。如果后续还要继续补强，建议优先补这些文档：

- 各任务类型的真实运行样例
- OpenSora/QwenVL MSA 环境框架问题修复进展
- 常见错误码 / 失败模式对照表
- `status.json` 字段说明
- `analysis/data/*.json` 字段说明
- `assets/runtime/model_config/*.yaml` 的模型适配规范
- `mm/` 目录的独立开发文档
