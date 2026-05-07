# Task6 统计规则说明文档

> **作者**: 邹英龙
> **更新日期**: 2026-04-14
>
> 本文档详细说明 Task6 多模态整网变异任务的统计指标计算规则。
>
> **适用范围**: LMSV Task6 多模态整网变异和验证

## 概述

Task6 统计系统用于量化分析多模态模型在 PTA（PyTorch Ascend）和 MSA（MindSpore Adapter）双环境下的执行差异，帮助发现框架级兼容性问题。

## 核心统计指标定义

### 1. 总迭代次数
- **定义**：有效突变轮次数（PTA成功执行的轮次）
- **计算**：总迭代次数 = PTA成功的轮次数
- **说明**：只有PTA成功执行的变异才计入总迭代次数

### 2. 成功迭代次数
- **定义**：PTA和MSA都顺利执行的次数
- **计算**：成功迭代次数 = PTA成功且MSA也成功的轮次数
- **说明**：
  - 对于训练模型：需要PTA和MSA都有loss输出才算成功
  - 对于推理模型：需要msrun_log中有执行日志且没有error才算成功

### 3. 发现问题数
- **定义**：MSA不成功执行的次数
- **计算**：发现问题数 = 总迭代次数 - MSA成功次数
- **说明**：
  - MSA执行失败（日志中有error）
  - 或msrun_log中没有该轮的执行日志

### 4. PTA成功率
- **定义**：有效突变次数占总突变次数的比例
- **计算**：PTA成功率 = 有效突变次数 / 总突变次数
- **说明**：
  - 有效突变次数 = PTA成功的轮次数
  - 总突变次数 = 有效突变次数 + 无效突变次数（被撤回的）

### 5. MSA成功率
- **定义**：PTA成功且MSA也成功的轮次占PTA成功轮次的比例
- **计算**：MSA成功率 = MSA成功次数 / PTA成功次数
- **说明**：分母是PTA成功的轮次，不管MSA是否成功

### 6. 问题发现率
- **定义**：发现问题数占总迭代次数的比例
- **计算**：问题发现率 = 发现问题数 / 总迭代次数

## 支持模型状态

| 模型 | 类型 | PTA状态 | MSA状态 | 说明 |
|------|------|---------|---------|------|
| InternVL3 | 训练 | ✅ 正常 | ✅ 正常 | 基准模型，精度差异约20% |
| QwenVL2.5 | 推理 | ✅ 正常 | ❌ 失败 | InnerInplaceIndexPut shape mismatch |
| OpenSora1.2 | 推理 | ✅ 正常 | ❌ 失败 | UntypedStorage 错误 |
| CogVideoX | 训练 | ✅ 正常 | ✅ 正常 | 经环境修复后已可正常运行 |

## 模型类型特定规则

### 训练模型

#### 顺利执行判定
- PTA必须有loss输出
- MSA必须有loss输出
- 显存和时间差异必须有值

#### 不顺利执行处理
- 只要是有效突变（PTA成功），PTA的loss必须有值
- 如果MSA崩溃，MSA的loss记录为0
- 显存差异和时间差异必须有值（可以为0）

### 推理模型

#### 顺利执行判定
- msrun_log中必须有该轮的执行日志
- 日志中不能有error
- 显存和时间差异必须有值

#### 问题检测
- 如果检测到问题，问题描述为msrun_log中最后一个error的报错信息
- 如果没有error但有warning，记录最后一个warning

## 报告输出格式

### Markdown报告
包含以下统计信息：
- 总迭代次数
- 成功迭代次数
- 发现问题数
- PTA成功率（格式：百分比 (成功次数/总次数)）
- MSA成功率（格式：百分比 (成功次数/PTA成功次数)）
- 问题发现率（格式：百分比 (问题数/总迭代次数)）
- 平均Loss差异
- 平均显存差异
- 平均时间差异

### HTML报告
包含可视化统计卡片：
- 总迭代次数
- 成功迭代数
- 发现问题数
- PTA成功率
- MSA成功率
- 问题发现率

以及详细统计摘要：
- 总突变次数
- 有效突变次数
- PTA成功次数
- MSA成功次数
- 平均Loss差异
- 平均显存差异
- 平均时间差异
- PTA成功率

### JSON报告
包含完整的统计信息：
```json
{
  "statistics": {
    "total_iterations": 10,
    "successful_iterations": 8,
    "issue_count": 2,
    "pta_success_count": 10,
    "msa_success_count": 8,
    "total_mutations": 12,
    "pta_success_rate": 0.833,
    "msa_success_rate": 0.8,
    "issue_rate": 0.2,
    "avg_loss_diff": 0.001,
    "avg_memory_diff": 10.5,
    "avg_time_diff": 5.2
  }
}
```

## 实现说明

### 代码位置
- 主统计逻辑：`utils/task/task6.py`
- 报告生成：`utils/analyze/task6_result.py`

### 关键变量
- `valid_iter_count`：有效突变次数（PTA成功的轮次）
- `invalid_iter_count`：无效突变次数（被撤回的）
- `total_mutations`：总突变次数 = 有效 + 无效
- `pta_success_count`：PTA成功次数
- `msa_success_count`：MSA成功次数
- `issue_count`：发现问题数

### 错误信息提取规则
1. 优先查找最后一个ERROR
2. 如果没有ERROR，查找Traceback或其他异常类型
3. 如果没有异常，查找最后一个WARNING
4. 如果都没有，返回默认错误信息
