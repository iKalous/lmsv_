# 静态图整网 PTA-MF 精度对齐说明（已落地项）

本文只总结当前已经在流程中实现并生效的 PTA（动态图）与 MF（静态图）精度对齐项，不包含尚未完成或仅讨论中的项。

## 1. 模型结构参数对齐（脚本参数 -> MF YAML）

1. 已将 PTA 脚本中的核心结构参数映射到 MF 配置：`hidden_size`、`num_layers`、`num_attention_heads`、`ffn_hidden_size`（映射为 `intermediate_size`）、`seq_length`、`max_position_embeddings`、`vocab_size/padded_vocab_size`、`layer_norm_epsilon`、`attention_dropout`、`hidden_dropout`、`position_embedding_type`、`rotary_base`。

2. 已将 Qwen3 相关结构参数做专项归一化：`num_key_value_heads` 按 `group_query_attention/num_query_groups` 或 `num_attention_heads` 推导；`hidden_act`、`add_bias_linear`、`qkv_concat`、`qk_layernorm`、`head_dim` 与 PTA 参数同步或回填。

3. 已在转换阶段对 `head_dim` 做强制对齐：优先使用 PTA 的 `kv_channels`，否则使用 `hidden_size // num_attention_heads` 推导，避免模板默认值漂移。

4. 已在转换阶段增加 Qwen3 一致性校验：`hidden_size`、`num_attention_heads`、`intermediate_size`、`head_dim` 与源 PTA 参数不一致时直接报错，提前失败，不再拖到训练加载阶段。

5. 变异链路里出现的 `num_query_groups`、`group_query_attention`、`qk_layernorm`、`swiglu`、`disable_bias_linear`、`kv_channels` 等参数，会分别落到 `num_key_value_heads`、`qkv_concat`、`hidden_act`、`add_bias_linear`、`head_dim` 等 MF 字段，尽量保持两侧表达同义。

## 2. 并行与批大小参数对齐

1. 已根据脚本参数统一计算并写入并行拓扑：`world_size`、`data_parallel`、`tensor_model_parallel_size`、`pipeline_model_parallel_size`、`context_parallel_size`、`expert_model_parallel_size`。

2. 已统一处理 batch 相关参数：`micro_batch_size`、`global_batch_size`、`gradient_accumulation_steps`、`micro_batch_num`，并在流水并行场景做可训练约束处理（保证 pipeline 场景下配置可运行）。

3. 已按真实数据列动态生成 `dataset_strategy`，避免策略长度与输入列不一致导致静态图训练行为偏离。

## 3. 训练行为与随机性对齐

1. 已对齐随机种子：`seed` 同步写入全局和数据集构建配置，保证 PTA/MF 在同一轮次下数据切分与采样更可比。

2. 已对齐训练步数相关配置：`train_iters` 会结合断点恢复迭代信息计算有效训练样本规模，并同步到数据集 `sizes` 与 `lr_schedule.total_steps`。

3. 已对齐优化器和学习率关键参数：`adam_beta1`、`adam_beta2`、`weight_decay`、`lr`、`min_lr`、`lr_warmup_fraction`、`lr_decay_style`。

## 4. 权重转换与加载链路对齐（你明天可重点讲这部分）

1. PTA 训练先产出 Megatron 侧权重目录（`mg` 格式语义）。

2. Task1 在 MF 运行前调用统一转换入口：`convert.sh + convert_ckpt.py`，按 `mg -> hf` 进行权重转换，并显式传入 `target TP/PP/EP`，保证转换目标并行规格可控。

3. 转换完成后会更新 MF YAML 的 `load_checkpoint` 到转换产物目录，并以 `load_ckpt_format: safetensors` 走 MindFormers safetensors 加载路径。

4. MF 训练阶段通过 `run_mindformer.py` 进入 MindFormers 的 checkpoint 加载流程（含分布式权重加载），与上一步转换产物直接衔接。

## 5. 运行环境与设备侧约束对齐

1. 转换和训练入口都做了 Ascend 环境与运行时变量注入，降低因环境差异导致的非算法误差。

2. `max_device_memory` 在 MF 侧有统一上限策略，避免不同机器默认内存策略差异引发图编译/执行行为偏移。

## 6. 本说明的边界

1. 本文仅列“已经实现并在代码链路中生效”的对齐项。

2. 若评审方希望新增参数对齐，请直接给出“参数范围”（例如：模型结构参数范围、并行参数范围、优化器参数范围、checkpoint 兼容范围），我们可按范围快速补齐并接入同一转换校验链路。

## 7. 本轮精度现状（qwen3, Task1, pta_mf）

实验批次：`output/2026-04-03-15-30-59`

### 7.1 结果概览

1. 执行轮次：2
2. PTA 成功：2/2
3. MF 成功：2/2
4. 精度问题：2/2（均为 CRITICAL）

### 7.2 定量结果

1. 最大 loss 绝对差：
	- iter1: `0.0581884384`
	- iter2: `0.0504207611`
2. 平均最大 loss 绝对差：`0.0543045998`

### 7.3 当前阶段结论

1. 结构/配置同义项已进入可比较区间，流程性问题不是主要瓶颈。
2. 差异模式集中在训练步推进后的数值分叉，重点应从“参数是否写入”转向“更新路径是否一致”。
3. 下一阶段优先关注：优化器生效参数、梯度聚合/规约行为、loss scale 与步进节奏。
