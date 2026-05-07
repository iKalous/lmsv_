# Task1 DeepSeekV3 PTA/MF 同配大配置 OOM 报告

## 结论

在 Task1 的 `pta_mf` 场景下，DeepSeekV3 使用同一套“大配置”时，PTA 侧可以正常完成流程，但 MF 侧在参数加载/内存分配阶段发生 `Allocate memory failed`，表现为 OOM。

该问题不是单纯的 `seq_length` 导致，`hidden_size`、`ffn_hidden_size`、`moe_intermediate_size`、`num_experts`、`moe_layer_freq`、`first_k_dense_replace` 以及并行配置共同影响最终显存占用。仅做单项缩减不能稳定解决问题。

## 复现入口

- Task1 入口：`do.py`
- 场景：`task_type = 1`
- 对比模式：`COMPARE_MODE = pta_mf`
- 模型：`MODEL_NAME = deepseekv3`

当前仓库里 Task1 的 deepseekv3 pta_mf 路径已经接入了统一低显存 profile，且可通过 Task1 入口跑通；本报告描述的是“大配置同配”下的失败现象，便于提 issue 追踪。

## 大配置参数

以下是 PTA 侧大配置里与这次问题直接相关的关键参数：

- `hidden_size: 896`
- `ffn_hidden_size: 2048`
- `num_attention_heads: 16`
- `num_query_groups: 16`
- `seq_length: 4096`
- `max_position_embeddings: 4096`
- `moe_intermediate_size: 1536`
- `num_experts: 32`
- `moe_layer_freq: 1`
- `first_k_dense_replace: 1`
- `micro_batch_size: 1`
- `global_batch_size: 8`
- `tensor_model_parallel_size: 2`
- `pipeline_model_parallel_size: 2`
- `context_parallel_size: 2`
- `expert_model_parallel_size: 1`
- `qk_nope_head_dim: 128`
- `qk_rope_head_dim: 64`
- `v_head_dim: 128`
- `kv_lora_rank: 512`
- `q_lora_rank: 1536`

对应的 PTA 生成脚本可见：`output/2026-04-15-10-50-06/iters/iter_1/scripts/pta-load_pretrain_mutated_deepseekv3-1.sh`。

## 现象 1：PTA 侧可跑通

在上述大配置下，PTA 侧可以继续完成训练流程，并进入后续对比阶段。与之对应的 MF 单跑则没有成功。

## 现象 2：MF 侧 OOM

MF 单跑在参数分配阶段报错：

- `Allocate memory failed`
- `AllocMemAndCopyForParameter`
- 失败张量示例：`shape:(229376, 4096)`

关键日志：

- [msrun_log_single_20260415_58/worker_0.log](../msrun_log_single_20260415_58/worker_0.log)

日志中可见专家层参数分配失败，典型节点名包括：

- `adam_v.decoder.layers.3.mlp.experts.weight1`
- `mtp.layers.0.transformer_layer.mlp.experts.weight1`

这说明问题主要集中在 MoE 专家参数的显存分配，而不是纯粹的序列长度或激活占用。

## 关键观察

1. `seq_length` 不是主因。
   - 仅把 `seq_length` 降到 1024，仍会在专家层参数分配时 OOM。

2. `hidden_size` 只能降低第一维压力。
   - 把 `hidden_size` 从 896 降到 512 后，专家层张量会从 `229376 x 4096` 降到 `131072 x 4096`，但仍然 OOM。

3. `intermediate_size` 单独下调也不够。
   - 把 `intermediate_size` 从 2048 降到 1024 后，仍然保留 `experts.weight1` 的 `4096` 维度，仍然 OOM。

4. 真正需要同时下调的是 MoE 结构相关参数。
   - `moe_intermediate_size`
   - `num_experts`
   - `moe_layer_freq`
   - `first_k_dense_replace`

## 已验证的低显存规避方案

为了让 Task1 的 deepseekv3 pta_mf 流程可跑通，当前验证过的一组统一低显存 profile 为：

- `hidden_size: 512`
- `ffn_hidden_size: 1024`
- `moe_intermediate_size: 768`
- `num_experts: 16`
- `seq_length: 1024`
- `max_position_embeddings: 1024`
- `moe_layer_freq: 1`
- `first_k_dense_replace: 7`
- `micro_batch_size: 1`

这组配置已经验证过 MF 单跑可以正常完成训练；同时也已经接入 Task1 的 pta_mf 低显存流程。

## 本轮 Task1 低显存流程的补充说明

在最新一次 Task1 端到端复跑中，流程已经进入 PTA-SAVE 阶段，但由于脚本生成阶段把 `moe_layer_freq` 继续压成了 `0`，PTA 在 FLOPs 计算阶段直接触发 `ZeroDivisionError`。与此同时，`first_k_dense_replace` 继续按总层数缩减也会在 `pipeline_model_parallel_size = 2`、`num_layers = 16` 的 DeepSeekV3 配置下把一个 pipeline stage 全部变成 dense 层，从而触发 MindSpeed 的参数校验：

- `Num-layer (8) must be greater than first-k-dense-replace (8) when first-k-dense-replace is set.`

这不是新的显存问题，而是低显存 profile 的边界条件不合法。当前已修正为两条规则：

- `moe_layer_freq >= 1`
- `first_k_dense_replace <= ceil(num_layers / pipeline_model_parallel_size) - 1`

对当前 DeepSeekV3 配置来说，`first_k_dense_replace` 的上限是 `7`，`moe_layer_freq` 则必须保留为 `1`。也就是说 DS 仍然可以跑，但不能再把每个 stage 的所有层都强制改成 dense，也不能把 MoE 层频率压成 `0`。

对 DS 模型变异的直接影响是：`first_k_dense_replace` 的可变异空间从“按总层数自由缩放”收紧为“按 stage 层数封顶”，同时 `moe_layer_freq` 不能再降到 `0`。超出这个上限的变异会被自动截断，否则就会在 PTA 侧提前失败，无法进入后续训练和对比流程。

最终结论是：DS 这条链路可以继续跑通，但 `first_k_dense_replace` 必须受 pipeline stage 约束，实际可用值不能超过每个 stage 的层数减 1；`moe_layer_freq` 必须保持为正整数，不能取 `0`。

## 建议在 issue 里强调的点

建议明确写出：

1. deepseekv3 在同一大配置下，PTA 可跑、MF OOM。
2. OOM 位置是 MoE 专家层参数加载，不是单独的 `seq_length` 问题。
3. 仅减少 `seq_length` / `hidden_size` / `intermediate_size` 仍不足以稳定解决。
4. 需要同步约束 `moe_intermediate_size`、`num_experts`、`moe_layer_freq`、`first_k_dense_replace`，否则 MF 仍会在专家参数分配阶段失败。
5. 当前仓库已经有一版可跑通的低显存 profile，可作为临时规避方案，但不代表原始大配置问题已消失。

## 相关文件

- [utils/task/task1.py](../utils/task/task1.py)
- [utils/runtime/mf_converter.py](../utils/runtime/mf_converter.py)
- [assets/runtime/configs/mf_converter_mapping.yaml](../assets/runtime/configs/mf_converter_mapping.yaml)
- [output/2026-04-15-10-50-06/iters/iter_1/scripts/pta-load_pretrain_mutated_deepseekv3-1.sh](../output/2026-04-15-10-50-06/iters/iter_1/scripts/pta-load_pretrain_mutated_deepseekv3-1.sh)
- [msrun_log_single_20260415_58/worker_0.log](../msrun_log_single_20260415_58/worker_0.log)
- [msrun_log_unified3_20260415/worker_0.log](../msrun_log_unified3_20260415/worker_0.log)

## 建议 issue 标题

`DeepSeekV3 Task1 pta_mf 同配大配置下 MF 在专家层参数分配阶段 OOM`
