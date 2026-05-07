# MF_CONVERTER工具文档

## 1.概述
mf_converter是用于模型配置脚本转换的一个工具，位于tools/mf_converter.py。它的主要功能是将pta脚本的参数转换为mf配置yaml格式的文件，方便进行脚本训练迁移和对齐。实现转换时需要借助mf_templates/{model_name}.yaml模板文件来完成参数映射，并且根据需要进行一些参数的特化处理。

## 2.整体转换流程

转换流程可以分为“解析输入 -> 归一化参数 -> 模板填充 -> 特化修正 -> 写出YAML”五个阶段：

1. 读取输入脚本并提取变量定义。
1. 提取多段参数字符串（GPT_ARGS、DATA_ARGS、OUTPUT_ARGS、MLA_ARGS、MOE_ARGS、ROPE_ARGS、PATH_ARGS）。
1. 展开变量引用（支持 ${VAR} 和 $VAR），并将参数字符串解析为键值字典。
1. 合并命令行中的 --save / --load、环境变量（如 DATA_PATH、TOKENIZER_PATH、SEQ_LEN）以及显式传入的 model_name/train_iters。
1. 加载模板 YAML，并调用 update_template_with_args 完成映射与特化处理。
1. 将最终配置以 yaml.dump 写出。

## 3.输入解析与参数归一化

### 3.1 变量提取

- 通过正则匹配以下形式：
	- export VAR=value
	- VAR=value
	- VAR="value" / VAR='value'
- 行内注释会被移除。
- 提取结果用于后续参数段落中的变量展开。

### 3.2 参数段落提取

- 通过 extract_args_section 按段提取字符串内容。
- 支持多行参数和反斜杠续行。
- 提取后会立刻进行变量展开。

### 3.3 参数字符串解析

- 解析器支持引号感知，避免把带空格的字符串错误切分。
- 参数键名统一做转换：--hidden-size -> hidden_size。
- 参数值类型转换策略：
	- true/false -> bool
	- 整数/浮点/科学计数法 -> 数值
	- 引号包裹值保留为字符串
	- 其他保持原始字符串

## 4.参数映射机制（ARG_MAPPING）

工具采用“源参数名 -> YAML路径”的静态映射方式。路径通过 set_nested_value 写入，支持字典与列表索引。

### 4.1 基础字段

- seed -> seed
- save -> output_dir
- data_path -> train_dataset.data_loader.config.data_path（通过基础逻辑写入）

### 4.2 model.model_config 相关

常见映射包括：

- hidden_size -> model.model_config.hidden_size
- num_layers -> model.model_config.num_hidden_layers
- num_attention_heads -> model.model_config.num_attention_heads
- num_key_value_heads -> model.model_config.num_key_value_heads
- max_position_embeddings -> model.model_config.max_position_embeddings
- seq_length -> model.model_config.seq_length
- vocab_size / padded_vocab_size -> model.model_config.vocab_size
- layer_norm_epsilon / rms_norm_eps -> model.model_config.rms_norm_eps
- ffn_hidden_size -> model.model_config.intermediate_size
- hidden_dropout -> model.model_config.hidden_dropout
- attention_dropout -> model.model_config.attention_dropout
- hidden_act -> model.model_config.hidden_act
- rotary_base -> model.model_config.rope_theta
- kv_channels -> model.model_config.head_dim
- use_flash_attn -> model.model_config.use_flash_attention
- add_qkv_bias -> model.model_config.attention_bias
- add_bias_linear -> model.model_config.add_bias_linear
- untie_embeddings_and_output_weights -> model.model_config.untie_embeddings_and_output_weights
- qk_layernorm -> model.model_config.qk_layernorm
- qkv_concat -> model.model_config.qkv_concat
- model_type / architectures -> model.model_config 对应字段

### 4.3 优化器与学习率

- weight_decay -> optimizer.weight_decay
- adam_beta1 -> optimizer.betas[0]
- adam_beta2 -> optimizer.betas[1]
- lr -> lr_schedule.learning_rate
- min_lr -> lr_schedule.min_lr
- lr_decay_style -> lr_schedule.type
- lr_warmup_fraction -> lr_schedule.warmup_ratio

### 4.4 并行与训练控制

- micro_batch_size -> runner_config.batch_size
- tensor_model_parallel_size -> parallel_config.model_parallel
- pipeline_model_parallel_size -> parallel_config.pipeline_stage
- context_parallel_size -> parallel_config.context_parallel
- sequence_parallel -> parallel_config.use_seq_parallel
- expert_model_parallel_size -> parallel_config.expert_parallel
- use_distributed_optimizer -> parallel.enable_parallel_optimizer

### 4.5 显式忽略或延后计算

以下参数当前不直接映射（标记为 None 或注释说明）：

- normalization
- make_vocab_size_divisible_by
- gradient_accumulation_steps（由 global_batch_size / micro_batch_size 派生）

## 5.特化处理逻辑

除通用映射外，脚本在 update_template_with_args 中加入了一组“规则化修正”，用于保证目标 YAML 可训练且与 MindFormers 约束一致。

### 5.1 基础兜底与类型设置

- auto_trans_ckpt 固定为 False，避免在 load_checkpoint 为空时触发恢复训练约束错误。
- compute_dtype：
	- 存在 bf16 参数时设为 bfloat16
	- 否则设为 float32
- context.max_device_memory 固定为 40GB（DEFAULT_MAX_DEVICE_MEMORY）。

### 5.2 模型行为开关

- swiglu 存在时，强制 model.model_config.hidden_act = swiglu。
- disable_bias_linear 存在时，转换为 add_bias_linear = not disable_bias_linear。
- clip_grad 存在时，启用 runner_wrapper.use_clip_grad = True。
- num_query_groups 存在时，映射为 num_key_value_heads。
- qkv_concat 在 qwen3 场景下默认回填为 True，以保持与 PTA 的融合投影语义一致。

### 5.3 Qwen3专项处理

当 model_type=qwen3 时：

- 删除 model_config 中 MindFormers 不接受的融合开关：
	- use_fused_swiglu
	- use_fused_rmsnorm
	- use_fused_rotary_pos_emb
- 若未显式传入 qkv_concat，则默认置为 True。
- 若未显式传入 qk_layernorm，则默认置为 True。
- 若未显式传入 enable_alltoall，则 parallel.enable_alltoall 默认 True。

### 5.4 注意力维度归一化

为减少 shape mismatch 风险，会做以下一致性修正：

- head_dim 若不是 8 的倍数，向上对齐到 8 的倍数。
- 当 num_attention_heads 与 head_dim 同时可用时，令 hidden_size = num_attention_heads * head_dim。
- 若 hidden_size 被重算，intermediate_size 按原始比例同步缩放并做 8 对齐。
- num_key_value_heads 约束在 (0, num_attention_heads]；缺省时回填为 num_attention_heads。
- 当 context_parallel > 1 时，强制 use_flash_attention=True。

### 5.5 并行参数联动

- 依据 tp/pp/cp 计算 dp = max(1, 8 // (tp*pp*cp))（当前实现默认 8 卡场景）。
- 写回 parallel_config：
	- data_parallel/model_parallel/context_parallel/pipeline_stage/use_seq_parallel
- 当 tp=1 时，use_seq_parallel 强制 False。
- 当 pp>1 时，保障 micro_batch_num>=2，并同步到：
	- parallel_config.micro_batch_num
	- runner_wrapper.micro_batch_num
	- train_dataset.micro_batch_num
	- train_dataset_task.dataset_config.micro_batch_num（若存在）

### 5.6 Pipeline offset修正

- 当 model.model_config.offset 为列表且 num_hidden_layers 为整数时：
	- 重新按 num_hidden_layers 和 pp 计算非负 offset
	- 每个 stage 至少保留 1 层
	- 若 num_hidden_layers 可被 pp 整除，则 offset 统一为 0
	- 若不能整除，则把余数从前往后依次分配到各 stage
	- 避免生成类似 [-1, -1, 1, 1] 这类可能触发中间 stage 为空的偏移

### 5.7 批大小与监控回填

- micro_bs = micro_batch_size（默认1）
- global_bs = global_batch_size（默认micro_bs）
- grad_acc = max(1, global_bs // micro_bs)
- 回填到 callbacks 中 MFLossMonitor：
	- global_batch_size
	- gradient_accumulation_steps

### 5.8 train_iters驱动的数据量对齐

当传入 --train-iters 时，会通过以下公式回推训练样本数，写入 train_dataset.data_loader.sizes[0]：

$$
effective\_batch\_size = batch\_size \times micro\_batch\_num \times micro\_batch\_interleave\_num
$$

$$
target\_samples = \left\lceil \frac{train\_iters \times effective\_batch\_size}{epochs} \right\rceil
$$

目的是让 MindFormers 的步数行为尽量贴近原始 MindSpeed 的 train_iters 控制。

## 6.模板与输出

- 模板优先级：
	- 命令行 --template 指定模板
	- 否则使用默认模板 mf_template.yaml
- 输出文件命名：
	- 显式 --output 优先
	- 否则使用 model_name 或输入脚本名作为基名
- YAML 输出策略：
	- sort_keys=False，保持可读性
	- allow_unicode=True，支持中文
	- width=inf，避免自动折行引发转义可读性问题

## 7.已知约束与建议

1. 当前 dp 计算默认总卡数为 8，异构卡数场景建议引入可配置总卡数参数。
2. swiglu、normalization 等参数仍存在框架语义差异，后续建议补充枚举映射与冲突校验。
3. ARG_MAPPING 为静态表，建议按模型族拆分子映射并增加单元测试覆盖。
4. 目前错误处理以 warning/try-except 为主，建议在关键字段缺失时提供 fail-fast 选项。