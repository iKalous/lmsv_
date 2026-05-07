# PTA 参数变异流程与代码结构

本文档说明 PTA 在生成变异脚本时的完整流程、关键模块职责，以及并行参数约束（尤其是 MoE/TP/SP 关系）。

## 1. 入口与执行顺序

主入口：
- `lmsv_rec/utils/runtime/mutate_and_forward/parallel_mutate/main.py`

核心流程（按执行顺序）：
1. `bashtoyaml(...)`：把原始 bash 启动脚本解析为中间 YAML。
2. `_merge_mutation_sections(...)`：把 mutation record（如 TransformerConfig/extra_config/spec）合并到模板配置。
3. `InfoParser.parse_file(...)`：归一化配置结构，生成标准化 YAML。
4. `ParallelParameterMutator.mutate_parallel_parameters()`：执行并行参数变异（TP/PP/EP/CP/DP）。
5. `EnhancedMegatronConfigValidator.validate_and_fix()`：做一致性校验并自动修复。
6. `YamlToBashConverter.convert()`：将最终配置回写为可执行 bash 脚本。

## 2. 代码结构

目录：
- `lmsv_rec/utils/runtime/mutate_and_forward/parallel_mutate`

关键文件：
- `main.py`
  - 变异流水线编排。
  - 配置合并、参数提取、最终脚本写出。
- `ParallelParameterMutator.py`
  - 并行参数随机变异与基础约束修正。
  - 负责生成 TP/PP/EP/CP/DP 合法组合。
- `config_validator_moe.py`
  - 全量配置校验与自动修复。
  - 包含并行、MoE、batch、lr、recompute 等约束。
- `yamlToBash.py`
  - YAML 到 bash 参数串转换。
- `BashToYaml.py`
  - bash 到 YAML 解析。
- `InfoParser.py`
  - 配置格式归一化。

## 3. 并行参数约束（重点）

### 3.1 基础约束
- 所有并行维度必须为正整数。
- 总并行规模不应超过 `world_size`。
- `CP > 1` 时，要求 `TP >= 2`。
- `TP = 1` 时，不允许 `sequence_parallel=True`。

### 3.2 MoE 相关强约束
- 当满足以下条件时：
  - 启用 MoE（`num_experts > 1`）
  - 且 `TP > 1`
  - 必须满足 `sequence_parallel = True`
- 否则会在 Megatron 初始化阶段报错：
  - `ValueError: When using expert parallelism and tensor parallelism, sequence parallelism must be used`

### 3.3 其他强约束

以下约束均来自当前 `ParallelParameterMutator.py` 与 `config_validator_moe.py` 的现有逻辑。

#### A. 并行维度与流水线
- `TP * PP * EP * CP` 不能超过 `world_size`，超出时会优先降 `CP`。
- `num_layers` 必须可被 `PP` 整除；若开启 VPP，还必须可被 `PP * VPP` 整除。
- 当 `VPP > 1` 且 `PP == 1` 时，会尝试提升 `PP`；若无法满足并行约束则禁用 VPP。
- `VPP` 需要整除每个 pipeline stage 的层数，否则会自动降为合法因子或禁用。

#### B. MoE 结构合法性
- `num_experts` 必须能被 `EP` 整除。
- `moe_intermediate_size` 必须能被 `TP` 整除。
- `moe_router_topk` 不能为 0，且不能大于 `num_experts`。
- 当 `moe_router_topk == 1` 时，必须开启 `moe_router_pre_softmax=True`。
- MoE 下不允许 `add_bias_linear=True`；会被修复为关闭 bias linear（`disable_bias_linear=True`）。

#### C. MoE 路由特定约束
- 当 `moe_router_load_balancing_type == group_limited_greedy` 时，需要同时满足：
  - `EP > 1`
  - `1 <= moe_router_group_topk < EP`
- 若上述条件不满足，会自动降级为 `aux_loss`。
- 当启用 group-limited 路由且缺失 `moe_router_num_groups` 时，会自动补齐。

#### D. 批次与训练规模
- `global_batch_size` 必须能被 `DP * micro_batch_size` 整除。
- 当全局 batch 过大或与并行约束冲突时，会优先修复为可运行的最小合法值。
- `micro_batch_size` 非法（<=0 或非整数）时会修复为 1。

#### E. 重计算与 SP 互斥
- `sequence_parallel=True` 时，不允许 `distribute_saved_activations=True`。
- `distribute_saved_activations=True` 仅在 `recompute_granularity=full` 下合法。

#### F. TP 相关维度整除
- `hidden_size`、`num_attention_heads`、`ffn_hidden_size` 均需可被 `TP` 整除。
- 在 Swiglu 模式下，`ffn_hidden_size` 还需可被 `2 * TP` 整除。


## 4. 产物与调试建议

流程中间文件（位于输出目录）：
- `bashtoyaml.yaml`
- `config_data.yaml`
- `standard_config.yaml`
- `mutated_config.yaml`
- `validated_config.yaml`
- `sh_arguments.txt`

定位问题建议按以下顺序检查：
1. `mutated_config.yaml`：并行参数是否按预期变异。
2. `validated_config.yaml`：校验器是否应用了修复。
3. 最终 bash：是否包含 `--sequence-parallel`（在 MoE + TP>1 情况下应存在）。

## 5. 扩展约束的推荐位置

新增并行/MoE 约束时，推荐分层：
- 变异阶段可行性约束：`ParallelParameterMutator.py`
- 运行前一致性兜底：`config_validator_moe.py`

这样可以保证：
- 变异空间尽可能合法。
- 即便输入配置脏数据或历史字段混用，也能在校验阶段收敛到可运行配置。
