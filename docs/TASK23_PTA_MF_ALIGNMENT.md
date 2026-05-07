# Task2 / Task3 PTA-MF 精度对齐清单（表格版）

本文仅列出当前流程中已经落地并生效的对齐项，便于汇报与复盘。

## 1. 总体流程对齐（Task2 / Task3 通用）

| 对齐环节 | PTA 侧 | MF 侧 | 当前实现 | 目的 |
|---|---|---|---|---|
| 变异参数源一致 | 同一轮 `MUTATE_ARGS` | 同一轮 `MUTATE_ARGS` | 已实现 | 保证两侧比较的是同一组结构/算子扰动 |
| 初始权重一致 | PTA-SAVE 生成共享权重 `*.pth` | 基于同一轮共享权重加载 | 已实现 | 消除“起始权重不同”导致的精度偏差 |
| 权重格式桥接 | `pth` | `ckpt` | 已实现（`pth -> npz -> ckpt`） | 消除框架格式差异引入的加载误差 |
| 分阶段失败隔离 | 每轮分阶段判定与归档 | 每轮分阶段判定与归档 | 已实现 | 防止失败轮结果污染精度统计 |
| 日志与轮次隔离 | 清理旧轮 CSV/进程 | 清理旧轮 CSV/进程 | 已实现 | 防止跨轮残留造成伪差异 |

## 2. Task2 参数对齐（重点）

### 2.1 训练输入与轮次控制

| 参数/开关 | 来源 | 落点 | 对齐说明 |
|---|---|---|---|
| `MUTATE_ROUND` | Task2 迭代器 | PTA / MSA / MF 运行环境变量 | 三侧统一轮次标识 |
| `MUTATE_ARGS` | Task2 统一构造 | PTA / MSA / MF 命令行 | 三侧使用同一套变异参数 |
| `LMSV_TRAIN_ITERS` / `--train-iters` | Task2 配置（SAVE/LOAD 步数） | PTA / MSA / MF 执行命令 | 同轮统一步数语义 |
| `BASE_SEED` | Task2 配置 | 全流程环境 | 固定随机基线，减少随机抖动 |

### 2.2 共享权重与加载链路

| 参数/开关 | 来源 | 落点 | 对齐说明 |
|---|---|---|---|
| `LMSV_SHARED_WEIGHT_PATH` | PTA-SAVE 产物路径 | PTA-LOAD / MSA-LOAD 输入 | 保证对比阶段加载同一共享权重 |
| `LMSV_SHARED_WEIGHT_MODE` | Task2 控制 | PTA / MSA 运行环境变量 | 统一权重使用模式（save/load） |
| `LMSV_SHARED_WEIGHT_CKPT_PATH` | `convert_shared_weight_for_mf` 输出 | MF 运行环境变量 | MF 加载与 PTA 同源权重（格式转换后） |
| `--shared-weight-ckpt`（Task3为主） | 转换后 ckpt 路径 | MF 子图/子模块入口 | 明确传入共享权重 ckpt |

### 2.3 MF 严格一致性开关（Task2 已启用）

| 开关 | 默认值 | 作用 |
|---|---|---|
| `LMSV_ALIGN_ADD_QKV_BIAS` | `1` | 对齐 QKV bias 相关行为，减少 PTA/MF 默认差异 |
| `LMSV_STRICT_ATTN_CONFIG_MATCH` | `1` | 强制 attention 关键配置匹配校验 |
| `LMSV_STRICT_ATTN_PARAM_LOAD` | `1` | 强制 attention 关键参数加载一致性 |
| `LMSV_TASK3_FORCE_MF_SAFE` | `0` | 预留安全模式开关（按需开启） |
| `LMSV_STRICT_DECODER_CONFIG_MATCH` | `0` | decoder 更严格校验开关（按需升级） |

## 3. Task3 参数对齐（重点）

Task3 与 Task2 对齐思想一致，差异在于任务对象是“子图/组合级”。

### 3.1 关键对齐参数

| 参数/开关 | Task3 落点 | 对齐说明 |
|---|---|---|
| `COMPARE_MODE` (`pta_msa` / `pta_mf`) | 主流程分支 | 确保同轮只做一种对比路径，防止混淆 |
| `MODELS` + 任务构造的变异参数 | 变异入口 | 同轮统一模型集与变异组合 |
| `SAVE_STEPS` / `LOAD_STEPS` | PTA-SAVE / PTA-LOAD / MSA/MF-LOAD | 统一各阶段训练步数基线 |
| `ENABLE_MF_WEIGHT_LOAD` | PTA->MF 转换阶段 | 控制是否执行共享权重格式转换并加载 |
| `LMSV_SHARED_WEIGHT_CKPT_PATH` | MF 执行入口 | 使用同源共享权重做 MF 验证 |

### 3.2 PTA->MF 权重转换链路

| 步骤 | 工具 | 输入 | 输出 |
|---|---|---|---|
| 导出中间格式 | `utils/runtime/export_pth_to_npz.py` | `shared_weight.pth` | `shared_weight.npz` |
| 转换为 MF 可加载 | `utils/runtime/convert_npz_to_ckpt.py` | `shared_weight.npz` | `shared_weight.ckpt` |
| MF 加载验证 | `load_and_forward_graph.py` / `load_and_forward_submodule.py` | `shared_weight.ckpt` + 同轮变异参数 | MF 训练/校验日志与指标 |

## 4. 当前已覆盖的精度风险点（已对齐）

| 风险点 | 对齐措施 |
|---|---|
| 两侧变异参数不一致 | 统一 `MUTATE_ARGS` 并同轮注入 |
| 两侧起始权重不一致 | PTA-SAVE 共享权重 + PTA/MSA/MF 共用 |
| 框架权重格式不一致 | 固定 `pth -> npz -> ckpt` 桥接流程 |
| 同轮执行被旧数据污染 | 每轮清理日志/CSV/残留进程 + 失败轮隔离 |
| Attention 关键行为偏差 | MF 严格 attention 配置/参数匹配开关 |

## 5. 评审口径（可直接复述）

1. Task2/Task3 已完成“参数同源、权重同源、格式桥接、分阶段隔离、关键开关约束”五类核心对齐。
2. 文中表格仅列已落地项，不包含未实现策略。
3. 若需继续增加对齐，请评审方给出参数范围（例如：要新增哪些模块参数、哪些加载阶段、哪些严格校验级别），即可按范围快速扩展。
