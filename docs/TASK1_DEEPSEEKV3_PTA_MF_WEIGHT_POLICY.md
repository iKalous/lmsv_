# Task1 DeepSeekV3 PTA_MF 权重策略说明

## 1. 背景

在 Task1 的 deepseekv3 + pta_mf 场景下，MF 权重转换与加载链路当前不可稳定使用。为避免任务执行过程中反复失败，运行时增加了硬编码策略，统一跳过这两步。

## 2. 生效条件

当同时满足以下条件时，策略生效：

- task_type = 1
- tasks.1.MODEL_NAME = deepseekv3
- tasks.1.COMPARE_MODE = pta_mf

## 3. 硬编码行为

命中条件后，运行时会强制执行以下行为：

- ENABLE_MF_WEIGHT_LOAD = False
- ENABLE_WEIGHT_CONVERT = False

即使配置文件中显式写了 true，也会被忽略。

## 4. 对执行流程的影响

在 Task1 的 MF 分支中：

- 跳过 PTA -> MF 的权重转换
- 跳过将 load_checkpoint 回填到 MF YAML
- MF 按不加载共享权重的路径继续执行，用于流程联通与基础回归检查

同时在 PTA 脚本侧增加了 deepseekv3 特殊兜底：

- 强制移除 `--group-query-attention`，避免 PTA 报错 `group_query_attention should not be enabled`

## 5. 日志标识

命中该硬编码策略时，会在日志中看到类似告警：

- deepseekv3 在 Task1 pta_mf 模式下强制关闭 MF 权重转换与加载，忽略 ENABLE_MF_WEIGHT_LOAD/ENABLE_WEIGHT_CONVERT 配置

## 6. 配置建议

建议在配置中保持语义清晰，即使会被覆盖，也显式写出意图：

{
  "tasks": {
    "1": {
      "MODEL_NAME": "deepseekv3",
      "COMPARE_MODE": "pta_mf",
      "ENABLE_MF_WEIGHT_LOAD": false,
      "ENABLE_WEIGHT_CONVERT": false
    }
  }
}

## 7. 代码位置

核心逻辑位于：

- utils/task/task1.py

具体包含：

- 在参数解析后根据模型与模式执行强制覆盖
- 在 MF 阶段走跳过权重转换与加载的分支

## 8. 边界与后续

- 本策略仅针对 Task1 deepseekv3 pta_mf 生效，不影响其他模型和任务。
- 如果后续 MF 权重链路适配完成，应先移除硬编码，再恢复可配置策略。
