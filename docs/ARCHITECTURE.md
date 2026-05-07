# LMSV_REC 架构文档（持续维护）

## 1. 文档目标

本文档用于两个核心目的：

- 作为 lmsv_rec 的长期架构记录与决策沉淀位置。
- 汇总 task1/task2/task3 代码链路，便于排障、扩展和新人接手。

说明：本文档以中文为主，保留少量必要英文术语（如 entrypoint、runtime、artifact）。

---

## 2. 系统入口层

### 2.1 主要入口

- 命令行统一入口：`lmsv`
- 交互式配置生成：`genconf.py`
- 任务执行入口：`do.py`
- Web 控制台：`webui.py`
- 单轮复现：`repro.py`

### 2.2 任务分发链路

`do.py` 读取 `config.json` 后，导出运行环境变量并分发：

- `utils.control.protect.task(1, params)` -> `utils.task.task1.main`
- `utils.control.protect.task(2, params)` -> `utils.task.task2.main`
- `utils.control.protect.task(3, params)` -> `utils.task.task3.main`

---

## 3. 分层视角

### 3.1 控制层（Control）

- `utils/control/protect.py`
- `utils/control/clean.py`

职责：

- 任务守护与信号处理。
- 清理 `pretrain_gpt.py` 及相关残留进程。

### 3.2 任务编排层（Task Orchestration）

- `utils/task/task1.py`
- `utils/task/task2.py`
- `utils/task/task3.py`
- `utils/task/runtime_helpers.py`

职责：

- 迭代生命周期管理。
- 脚本生成与参数改写。
- PTA/MSA/MF 执行编排。
- 日志搬运、产物备份与迭代收尾。

### 3.3 运行时工具层（Runtime Tooling）

- `utils/runtime/convert_pretrain_script.py`
- `utils/runtime/mf_converter.py`
- `utils/runtime/convert_ckpt.py`
- `scripts/runtime/convert.sh`
- `scripts/runtime/mf_start.sh`
- `scripts/runtime/msrun_launcher.sh`

职责：

- PTA -> MSA 脚本转换。
- PTA 脚本 -> MF yaml 转换。
- PTA 权重 -> MF 可加载权重转换。
- 统一封装运行命令与启动逻辑。

### 3.4 分析与报告层（Analyze & Report）

- `utils/analyze/task1_result.py`
- `utils/analyze/precision.py`
- `analyze.py`

职责：

- 按迭代 / 按任务汇总分析。
- PTA-MSA、PTA-MF 精度差异检测。
- 历史 output 分析重建。

---

## 4. Task 代码链路

## 4.1 Task1：整网泛化变异

核心文件：`utils/task/task1.py`

当前关键流程：

1. 读取配置并归一化运行参数。
2. 解析 `COMPARE_MODE`：
   - `pta_msa`：执行 PTA + MSA 链路。
   - `pta_mf`：执行 PTA + MF 链路。
3. 按模式做模型支持校验：
   - `pta_mf`：当前仅允许 `qwen3`。
   - `pta_msa`：允许 `scripts/templates/pretrain_example` 中已有模板模型。
4. 每轮迭代执行：
   - mutate 产物生成
   - PTA 脚本生成 + SAVE/LOAD 参数配置
   - MSA 脚本生成
   - 可选 MF yaml 生成
   - PTA SAVE（产出权重）
   - PTA LOAD
   - 分支执行：MSA 或 MF
   - 迭代收尾 + 产物归档
5. 全局统计与分析刷新。

Task1 关键函数：

- `run_mutate`
- `generate_pta_script`
- `convert_msa_script`
- `convert_pta_checkpoint_for_mf`
- `run_mf_training`
- `wait_msa_finish` / `wait_mf_finish`
- `finalize_iter`

### 4.1.1 Task1 的 PTA->MF 转换链

- 本地转换入口：`utils/runtime/convert_ckpt.py`
- 可注入参数壳脚本：`scripts/runtime/convert.sh`
- task1 对接函数：`convert_pta_checkpoint_for_mf`

环境要求（重要）：

- 需要 `conda activate <mindspeed_env>`。
- 还需要 `source scripts/envset/pta.sh` 完成 PYTHONPATH 注入。

## 4.2 Task2：模块内组件泛化

核心文件：`utils/task/task2.py`

高层流程：

1. 构建模型-子模块映射。
2. 执行 mutate。
3. PTA SAVE 产出共享权重。
4. PTA LOAD 验证。
5. MSA LOAD 验证。
6. 按配置可进入 MF 路径。
7. 迭代状态落盘与产物快照。

## 4.3 Task3：模块间组合变异

核心文件：`utils/task/task3.py`

详细执行与排障文档：`docs/TASK3.md`

高层流程：

1. 解析 `COMPARE_MODE`。
2. 共享 mutate + 共享权重策略。
3. PTA SAVE + PTA LOAD。
4. 按模式分支：
   - `pta_msa`：MSA LOAD + 精度对比
   - `pta_mf`：权重转换 + MF 校验
5. 统一写入迭代状态和报告。

---

## 5. 配置契约（config.json）

### 5.1 全局字段

- `task_type`
- `DATA_PATH`
- `PTA_NAME` / `PTA_PATH`
- `MSA_NAME` / `MSA_PATH`
- `MF_NAME`（按模式需要）

### 5.2 任务字段

Task1 常用字段：

- `MODEL_NAME`
- `TOTAL_ITER`
- `COMPARE_MODE`（`pta_msa` 或 `pta_mf`）
- `ENABLE_WEIGHT_CONVERT`

Task2 常用字段：

- `MODELS`
- `SUBMODULES`
- `TOTAL_ITER`

Task3 常用字段：

- `MODELS`
- `COMPARE_MODE`
- `TOTAL_ITER`

---

## 6. 输出与产物目录（典型）

- `output/<timestamp>/`
  - `log.txt`
  - `config.json`
  - `iters/iter_x/`
    - `runtime_logs/`
    - `msrun_log/`
    - `artifacts/`
    - `FAILED_FLAG` / `failure_info.txt`

该结构用于：

- 单轮复现（repro）
- 历史分析重跑（analyze）
- 精度问题定位

---

## 7. 架构待办（Backlog）

建议在本节持续维护下一阶段工作。

### 7.1 可推进主题

- task1/task2/task3 状态机抽象统一。
- 错误分类与重试策略标准化。
- CLI 与 WebUI 共享同一份强类型配置 Schema。
- shell 命令拼接与业务编排解耦。
- 每轮统一产出 `manifest.json`（版本化产物清单）。

### 7.2 ADR 模板

后续重大架构决策建议记录为 ADR：

- ADR 编号
- 日期
- 背景（Context）
- 决策（Decision）
- 影响（Consequences）
- 回滚方案（Rollback）

---

## 8. 快速索引

- Task1：`utils/task/task1.py`
- Task2：`utils/task/task2.py`
- Task3：`utils/task/task3.py`
- 模型支持规则：`utils/runtime/model_support.py`
- 权重转换壳脚本：`scripts/runtime/convert.sh`
- 权重转换代码：`utils/runtime/convert_ckpt.py`
