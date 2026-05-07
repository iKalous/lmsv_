# Task6 四模型处理逻辑文档

> **作者**: 邹英龙
> **更新日期**: 2026-04-11
> **适用范围**: LMSV Task6 多模态整网变异和验证

## 模型分类与状态

### 训练模型
| 模型 | 类型 | PTA状态 | MSA状态 |
|------|------|---------|---------|
| **InternVL3** | train | ✅ 正常 | ✅ 正常 |
| **CogVideoX** | train | ✅ 正常 | ✅ 正常 |

### 推理模型
| 模型 | 类型 | PTA状态 | MSA状态 |
|------|------|---------|---------|
| **QwenVL2.5** | inference | ✅ 正常 | ❌ InnerInplaceIndexPut shape mismatch |
| **OpenSora1.2** | inference | ✅ 正常 | ❌ UntypedStorage错误 |

## 成功判断逻辑

### PTA验证 (`run_pta_verify`)

#### 训练模型 (InternVL3/CogVideoX)
1. 检查日志中是否有真正的错误（排除Warning）
2. 必须有loss输出（`metrics["loss"] is not None`）
3. 返回码非零但无错误时，视为成功但警告

#### 推理模型 (QwenVL/OpenSora)
1. 检查日志中是否有真正的错误（排除Warning）
2. 必须有显存或时间指标（`memory`或`time`不为None）
3. 返回码非零但无错误时，视为成功但警告

### MSA验证 (`run_msa_verify`)

#### 训练模型 (InternVL3/CogVideoX)
1. 检查日志中是否有真正的错误（排除Warning）
2. 必须有loss输出（`metrics["loss"] is not None`）
3. 如果没有loss，提取最后一个error/warning作为错误信息

#### 推理模型 (QwenVL/OpenSora)
1. 检查是否有Python错误（Traceback/OSError等）
2. 检查是否有ERROR（排除WARNING）
3. 检查是否有执行记录（满足任一条件）：
   - 有显存指标（`memory is not None`）
   - 有时间指标（`time is not None`）
   - 日志中有"MSA inference completed"或"elapsed time per iteration"

## 日志解析

### PTA日志解析 (`_parse_pta_log`)
- **Loss模式**: `r"loss:\s+([\d.E+-]+)"`
- **显存模式**: `r"NPU memory.*?([\d.]+)\s*MB"`
- **时间模式**: `r"elapsed time per iteration.*?([\d.]+)"`
- **推理模型额外解析**:
  - `elapsed time per iteration \(ms\): ([\d.]+)`
  - `NPU memory \(MB\): ([\d.]+)`

### MSA日志解析 (`_parse_msa_log`)
- **Loss模式**: `r"loss:\s+([\d.E+-]+)"`
- **显存模式**: `r"npu.*?memory.*?([\d.]+)"`
- **时间模式**: 使用PTA的`TIME_PATTERN_PTA`（统一解析）
- **推理模型额外解析**:
  - `elapsed time per iteration \(ms\): ([\d.]+)`
  - `NPU memory \(MB\): ([\d.]+)`

## 结果分析 (`analyze_results`)

### Loss对比（仅训练模型）
- 如果PTA和MSA都有loss，计算差异
- 如果相对误差>1%且绝对误差>0.01，视为不匹配

### 显存对比（所有模型）
- 如果PTA和MSA都有显存数据，计算差异

### 时间对比（所有模型）
- 如果PTA和MSA都有时间数据，计算差异

## 错误信息提取

### 提取优先级
1. **最后一个ERROR**：`ERROR[:\s]+(.+?)(?:\n|$)`
2. **Python错误**：Traceback/OSError/RuntimeError等
3. **最后一个WARNING**：`WARNING[^\n]*`
4. **默认信息**："MSA执行失败（详见日志）"

### 推理模型特殊处理
- 检查msrun_log中是否有worker日志
- 检查日志中是否有error标记
- 问题描述使用最后一个error的报错信息

## 报告输出

### 训练模型报告
- 包含：总迭代次数、成功迭代次数、发现问题数
- 包含：PTA成功率、MSA成功率、问题发现率
- 包含：平均Loss差异、平均显存差异、平均时间差异
- 包含：每轮PTA/MSA Loss、显存差异、时间差异

### 推理模型报告
- 包含：总迭代次数、成功迭代次数、发现问题数
- 包含：PTA成功率、MSA成功率、问题发现率
- 包含：平均显存差异、平均时间差异（无Loss差异）
- 包含：每轮显存差异、时间差异、问题描述

## 多机模式成功判定

多机模式下使用 `run_pta_verify_multinode` / `run_msa_verify_multinode` 执行验证，本地和远程节点并发运行。

### 训练模型 (多机)
- 本地节点必须有loss输出（`local_metrics.get('loss') is not None`）
- 若本地成功（`local_ok=True`），即使远程节点失败，也优先返回本地成功结果

### 推理模型 (多机)
- 本地节点必须有显存或时间指标（`memory is not None` 或 `time is not None`）
- 若本地成功且满足上述指标条件，即使远程节点失败（如HCCL连接超时），也返回本地成功结果
- 远程节点失败不阻断本地成功判定（推理模型依赖本地执行的显存/时间指标即可）

### MSA失败时的错误信息
- MSA失败时优先使用 `msa_metrics.get("error_info")`（已由 `run_msa_verify` 提取的真实错误）
- 仅当 `error_info` 为空时，才降级从日志中重新解析错误信息
- 这确保了即使日志格式复杂，也能正确捕获真实bug（如 `TypeError: 'UntypedStorage'...`）

## 注意事项

1. **推理模型无Loss**：QwenVL和OpenSora是推理模型，不产生loss输出
2. **错误信息来源**：推理模型的问题描述来自msrun_log中的最后一个error
3. **执行记录判断**：推理模型通过显存/时间指标判断是否实际执行
4. **指标提取统一**：所有模型最终都输出统一格式的指标便于解析
5. **Python作用域陷阱**：不要在函数内部的条件分支中 `import re`，否则会导致 `UnboundLocalError`
