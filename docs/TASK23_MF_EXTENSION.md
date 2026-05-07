# Task2/Task3 的 MF 新模块扩展流程（静态图）

本文档说明如何把新的模块能力接入到 Task2（模块内变异）与 Task3（模块间变异）的 MF 对比链路。

适用范围聚焦本仓可维护代码：

- `utils/task/*`
- `utils/runtime/mf_mutate_and_forward/*`
- `assets/runtime/*`
- 配置入口（`genconf.py` / `webui.py` / `config.json`）

## 1. 总体分层（先判断改哪一层）

1. 配置入口层
- `genconf.py` / `webui.py` / `config.json` 中是否能把新模块参数传进 task2/task3。

2. 任务编排层
- `utils/task/task2.py` 与 `utils/task/task3.py` 是否正确把参数传给 MF 脚本。

3. MF 静态图构建层
- `utils/runtime/mf_mutate_and_forward/sub_graph.py`（task2）
- `utils/runtime/mf_mutate_and_forward/graph.py`（task3）

4. 对齐校验层
- `utils/runtime/mf_mutate_and_forward/load_and_forward_submodule.py`
- `utils/runtime/mf_mutate_and_forward/load_and_forward_graph.py`

## 2. Task2（模块内变异）接入步骤

Task2 的关键是“子模块编号 -> MF 子模块实现”的映射。

1. 明确新增子模块编号
- 当前配置入口把 `SUBMODULES` 限制在 `0~10`（见 `genconf.py` / `webui.py`）。
- 若新增编号超出范围，先放宽输入校验与默认值说明。

```python
# genconf.py
submodules = ask_int_list(
    "SUBMODULES（取值范围 0~11）",
    default_submodules,
    min_value=0,
    max_value=11,
)
```

```python
# webui.py（后端校验）
task2_submodules = _ensure_int_list(
    task2_raw.get("SUBMODULES"),
    "任务2 SUBMODULES",
    min_value=0,
    max_value=11,
)
```

```javascript
// webui.py 内嵌前端校验逻辑（JS）
if (value < 0 || value > 11) {
  throw new Error(`${field.label} 取值范围必须在 0~11 之间`);
}
```

2. 在 MF 子图里补映射
- 文件：`utils/runtime/mf_mutate_and_forward/sub_graph.py`
- 位置：`Graph.load(...)` 中 `submodule_num` 分支。
- 当前已有映射示例：
  - `0`: `input_layernorm`
  - `1`: `self_attention`
  - `2`: `self_attention.core_attention`
  - `5`: `self_attention.linear_proj`
  - `6`: `self_attention.linear_qkv`
  - `8`: `mlp`
  - `9`: `mlp.linear_fc1`
  - `10`: `mlp.linear_fc2`
- 新子模块必须在这里明确 `node.block = ...`，否则 MF 会落到不完整路径。

  ```python
  # utils/runtime/mf_mutate_and_forward/sub_graph.py
  submodule_num = node.block_num
  if submodule_num == 0:
    node.block = transformer_block.layers[0].input_layernorm
  elif submodule_num == 1:
    node.block = transformer_block.layers[0].self_attention
  elif submodule_num == 2:
    node.block = transformer_block.layers[0].self_attention.core_attention
  elif submodule_num == 8:
    node.block = transformer_block.layers[0].mlp
  elif submodule_num == 9:
    node.block = transformer_block.layers[0].mlp.linear_fc1
  elif submodule_num == 10:
    node.block = transformer_block.layers[0].mlp.linear_fc2
  elif submodule_num == 11:
    # 示例：新增 post_attention_layernorm
    node.block = transformer_block.layers[0].post_attention_layernorm
  ```

  同时在 `construct(...)` 执行分支补形状处理（否则容易在运行时炸维度）：

  ```python
  # utils/runtime/mf_mutate_and_forward/sub_graph.py
  elif submodule_num == 11:
    hidden_size = cur_node.config.hidden_size
    input_data = reshape_tensor_nd(
      input_data,
      (input_data.shape[0], input_data.shape[1], hidden_size),
    )
    output = cur_block(input_data)
  ```

3. 如需引入新的 decoder 类/算子，补操作集合
- 文件：`utils/runtime/OperatorSet.py`
- 若需要参与随机插入/变异候选，补 `llm_operators` / `insert_operators`。

  ```python
  # utils/runtime/OperatorSet.py
  llm_operators = {
    "Qwen2DecoderLayer",
    "ChatGLMDecoderLayer",
    "YourNewDecoderLayer",  # 新增
  }

  insert_operators = {
    *activation_operators,
    *seq_math_operators,
    *llm_operators,
  }
  ```

4. 准备 MF 模板配置
- 新模型建议新增 `assets/runtime/mf_templates/<model>.yaml`。
- 在 task2 配置里设置 `MF_ARGS_PATH` 指向模板。

  ```json
  {
    "tasks": {
    "2": {
      "COMPARE_MODE": "pta_mf",
      "MF_ARGS_PATH": "assets/runtime/mf_templates/your_model.yaml",
      "ENABLE_MF_WEIGHT_LOAD": false
    }
    }
  }
  ```

  Task2 最终会把参数传给 MF 运行脚本（可用于核对链路是否打通）：

  ```bash
  python utils/runtime/mf_mutate_and_forward/load_and_forward_submodule.py \
    --load-path <mutating-json> \
    --args_path assets/runtime/mf_templates/your_model.yaml \
    --train-iters <iters>
  ```

5. 运行联调
- `COMPARE_MODE=pta_mf`
- 建议先 `ENABLE_MF_WEIGHT_LOAD=false` 走通流程，再切到 `true` 验证权重加载一致性。

## 3. Task3（模块间变异）接入步骤

Task3 的关键是“图节点配置能否被 MF Graph 正确构建并执行”。

1. 先确认节点语义
- 文件：`utils/runtime/mf_mutate_and_forward/graph.py`
- 当前默认把 `node_id>0` 视作 `mutated_decoder`，`node_id=0` 视作 `embedding`。
- 如果你新增的是“非 decoder/embedding 类型节点”，需要显式扩展：
  - `Graph.load(...)` 的 `node.str_op` 与 `node.block` 构建逻辑
  - `Graph.construct(...)` 的执行分支

  ```python
  # utils/runtime/mf_mutate_and_forward/graph.py
  node = Node(config=init_config, index=node_id)
  node.str_op = "mutated_decoder" if node_id > 0 else "embedding"

  if node.str_op == "embedding":
    node.block = LanguageModelEmbedding(...)
  else:
    node.block = TransformerBlock(...)
  ```

  如果新增“非 decoder/embedding 节点”，建议显式扩展为：

  ```python
  # utils/runtime/mf_mutate_and_forward/graph.py
  if node_id == 0:
    node.str_op = "embedding"
  elif node_type == "adapter":
    node.str_op = "adapter"
  else:
    node.str_op = "mutated_decoder"

  if node.str_op == "adapter":
    node.block = MyAdapterBlock(node.config)
  ```

  并在 `construct(...)` 增加执行分支：

  ```python
  if cur_node.str_op == "embedding":
    output = cur_block(input_ids=input_ids, position_ids=position_ids)
  elif "decoder" in cur_node.str_op:
    output = cur_block(input_data, attention_mask)
  elif cur_node.str_op == "adapter":
    output = cur_block(input_data)
  ```

2. 补齐 TransformerConfig 字段兼容
- `Graph.load(...)` 会过滤不在 `TransformerConfig` dataclass 的字段。
- 若新模块依赖关键字段被过滤，需补映射或在构图前转换。

  ```python
  valid_fields = set(TransformerConfig.__dataclass_fields__.keys())
  filtered_cfg = {k: v for k, v in raw_cfg.items() if k in valid_fields}
  unknown_fields = set(raw_cfg.keys()) - valid_fields
  ```

  常见做法是在过滤前做一次键名映射：

  ```python
  FIELD_ALIAS = {
    "num_heads": "num_attention_heads",
    "eps": "layernorm_epsilon",
  }
  raw_cfg = {FIELD_ALIAS.get(k, k): v for k, v in raw_cfg.items()}
  ```

3. 扩展 PTA->MF 对齐字段（必要时）
- attention 对齐集合：`ATTENTION_ALIGN_FIELDS`（`graph.py`）
- decoder 关键对齐集合：`DECODER_CRITICAL_ALIGN_FIELDS`（`graph.py`）
- 对齐校验关注集合：`DECODER_FOCUS_FIELDS`（`load_and_forward_graph.py`）
- 当新增模块引入关键配置（且会影响数值路径）时，应同步补齐以上集合。

  ```python
  # graph.py
  ATTENTION_ALIGN_FIELDS = (
    "num_attention_heads",
    "num_query_groups",
    "kv_channels",
    "attention_dropout",
    "hidden_dropout",
    "normalization",
    "layernorm_epsilon",
    "masked_softmax_fusion",
    "attention_softmax_in_fp32",
    "apply_query_key_layer_scaling",
    "my_new_attention_field",  # 新增
  )

  DECODER_CRITICAL_ALIGN_FIELDS = (
    "bias_dropout_fusion",
    "apply_rope_fusion",
    "context_parallel_algo",
    "use_flash_attn",
    "my_new_decoder_field",  # 新增
  )
  ```

  ```python
  # load_and_forward_graph.py
  DECODER_FOCUS_FIELDS = (
    "hidden_size",
    "ffn_hidden_size",
    "num_attention_heads",
    "my_new_decoder_field",  # 新增
  )
  ```

4. 验证参数加载是否完整
- 日志关注：`attention参数加载统计`、`attention未加载参数样例`。
- 如出现系统性未加载，优先检查：
  - 权重转换产物是否正确（`pth -> npz -> ckpt`）
  - 参数命名与结构是否与 MF 节点对齐。

## 4. Task2/Task3 通用清单

1. 配置检查
- `COMPARE_MODE=pta_mf`
- `MF_ARGS_PATH` 指向有效模板
- 新模块编号/模型名可被配置入口接受

2. 任务侧检查
- `task2.py` 调用 `load_and_forward_submodule.py`
- `task3.py` 调用 `load_and_forward_graph.py`

```bash
# task3 对应 MF 调用（用于手工复现）
python utils/runtime/mf_mutate_and_forward/load_and_forward_graph.py \
  --load-path <mutating-json> \
  --args_path assets/runtime/mf_templates/your_model.yaml \
  --train-iters <iters>
```

3. 结果文件检查
- task2: `res/submodule_execution_mf.csv`
- task3: `res/execution_mf.csv`
- 步级日志: `res/training_log_mf/training_log-<iter>.csv`

4. 运行时日志检查
- `output/<timestamp>/iters/iter_i/runtime_logs/mf_iter{i}.log`
- 关注关键词：`RuntimeError`、`attention配置严格对齐失败`、`decoder配置严格对齐失败`

5. 对齐开关（按需）
- `LMSV_ALIGN_ADD_QKV_BIAS`
- `LMSV_STRICT_ATTN_CONFIG_MATCH`
- `LMSV_STRICT_ATTN_PARAM_LOAD`
- `LMSV_STRICT_DECODER_CONFIG_MATCH`

## 5. 建议接入顺序

1. 先在 Task2 打通单子模块（便于缩小问题面）。
2. 再迁移到 Task3 组合图，验证多节点串联行为。
3. 最后开启严格对齐开关做回归门禁。
