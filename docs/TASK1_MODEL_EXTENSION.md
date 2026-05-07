# Task1 语言模型整网变异扩展指南

本文面向后续开发者，说明如何在本工具中接入新的语言模型，使其参与 `Task1` 的整网泛化变异测试，并按需要接入 `pta_msa` / `pta_mf` 两条链路。

---

## 1. 先理解 Task1 的主链路

当前 `Task1` 的关键入口在 `utils/task/task1.py`，一轮完整流程大致如下：

1. `run_mutate()`  
   入口：`utils/task/task1.py`  
   实际调用：`scripts/mutation/mutate-auto.sh` -> `utils/runtime/mutate_and_forward/mutate_graph-auto.py`
2. `generate_pta_script()`  
   根据本轮 `mutating-<iter>.json` 和模型模板脚本，生成实际 PTA 训练脚本。
3. `convert_msa_script()`  
   将 PTA 脚本转换为 MSA 脚本。
4. 可选 `generate_mf_script()`  
   将 PTA 脚本转换为 MF YAML。
5. 可选 `convert_pta_checkpoint_for_mf()`  
   将 PTA 产出的 checkpoint 转成 MF 可加载格式。

从“模型接入”的视角看，Task1 里一个模型通常有 3 份表示：

- `assets/runtime/model_config/<model>.yaml`  
  作为整网变异的基线配置输入。
- `scripts/templates/pretrain_example/pretrain_mutated_<model>.sh`  
  作为 PTA 训练脚本模板。
- `assets/runtime/mf_templates/<model>.yaml`  
  作为可选的 MF 模板。

因此，扩展一个新模型通常分成 4 层：

1. 扩展整网变异基线配置；
2. 扩展 PTA 模板脚本；
3. 扩展变异参数 schema 与脚本合并逻辑；
4. 如需 `pta_mf`，再扩展 MF 模板与权重转换支持。

---

## 2. 最小接入路径：先让模型跑通 `pta_msa`

如果你的目标只是让新模型先进入整网变异，并走通 `PTA + MSA`，通常按下面流程即可。

### 步骤 1：新增模型基线配置 YAML

在 `assets/runtime/model_config/` 下新增：

```text
assets/runtime/model_config/<model>.yaml
```

文件名必须与 `MODEL_NAME` 一致。  
例如模型名是 `mygpt`，则文件必须叫：

```text
assets/runtime/model_config/mygpt.yaml
```

如果文件不存在，整网变异会直接失败。

### 步骤 2：按当前工具约定组织 YAML 结构

当前整网变异脚本主要识别以下几类字段：

- `TransformerConfig`
- `MLATransformerConfig`
- `extra_config`
- `get_gpt_layer_local_spec`
- `input`
- `attention_mask`

最常见的普通 Transformer 模型可以参考下面骨架：

```yaml
TransformerConfig:
  num_layers: 24
  hidden_size: 2048
  ffn_hidden_size: 6144
  num_attention_heads: 16
  num_query_groups: 8
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  normalization: "RMSNorm"
  layernorm_epsilon: 1.0e-6
  attention_dropout: 0.0
  hidden_dropout: 0.0
  add_bias_linear: false
  init_method_std: 0.01

extra_config:
  seq_length: 4096
  max_position_embeddings: 4096
  position_embedding_type: "rope"
  rotary_base: 1000000
  micro_batch_size: 1
  global_batch_size: 32
  lr: 1.25e-6
  min_lr: 1.25e-7
  weight_decay: 0.1
  clip_grad: 1.0
  train_iters: 2000

get_gpt_layer_local_spec:
  num_experts: null
  moe_grouped_gemm: false
  qk_layernorm: false
  multi_latent_attention: false
  normalization: "RMSNorm"
  qk_l2_norm: false

input:
  shape: [32, 1, 2048]
  dtype: torch.float32
  device: "cuda"

attention_mask: default
```

如果模型是 MLA/MoE 风格，也可以像 `deepseekv3.yaml` 那样使用 `MLATransformerConfig`。  
`mutate_graph-auto.py` 最终会通过 `utils/runtime/model_helpers.py` 中的 `extract_graph_transformer_config_from_yaml()` 提取出可用于图变异的基础配置。

### 步骤 3：注意哪些字段会真正参与图变异

整网图变异阶段只会消费“图结构真正需要的那部分参数”。当前代码里有两个重要事实：

1. `TransformerConfig` / `MLATransformerConfig` 中的部分字段会被清洗后再进入图构建。  
   例如 `params_dtype`、`bf16`、`fp16`、`sequence_parallel`、`multi_latent_attention` 等不会直接作为图实例化参数使用。
2. 真正决定结构扰动范围的是 `assets/runtime/configs/mutation_schema.yaml`。

也就是说：

- 你可以把模型完整配置写进 YAML；
- 但只有被 `mutation_schema.yaml` 覆盖到的参数，才会在 Task1 中被“主动变异”；
- 其余参数只是作为静态背景配置存在。

### 步骤 4：新增 PTA 模板脚本

在 `scripts/templates/pretrain_example/` 下新增：

```text
scripts/templates/pretrain_example/pretrain_mutated_<model>.sh
```

这个文件同样必须与 `MODEL_NAME` 严格同名。  
例如 `MODEL_NAME=mygpt`，则模板脚本必须是：

```text
scripts/templates/pretrain_example/pretrain_mutated_mygpt.sh
```

`Task1` 在 `generate_pta_script()` 中会直接读取这个模板。

### 步骤 5：模板脚本至少要满足这些要求

建议直接参考现有模板：

- `scripts/templates/pretrain_example/pretrain_mutated_qwen2.sh`
- `scripts/templates/pretrain_example/pretrain_mutated_qwen3.sh`
- `scripts/templates/pretrain_example/pretrain_mutated_deepseekv3.sh`

模板至少要满足：

1. 能以 `pretrain_gpt.py` 为主入口启动训练；
2. 暴露 `CKPT_SAVE_DIR`、`CKPT_LOAD_DIR`、`DATA_PATH` 等基础变量；
3. 模型核心参数以 `*_ARGS` 的形式组织，便于后续自动改写；
4. 明确带上 tokenizer 参数。至少需要以下之一：
   - `--tokenizer-type` + `--tokenizer-name-or-path`
   - `--tokenizer-type` + `--tokenizer-model`
5. 如果模型依赖 MoE / MLA / RoPE 特定参数，最好在模板里显式写出对应参数位。

这是因为 `parallel_mutate` 在把 `mutating-<iter>.json` 回填到模板脚本时，不是“无条件追加所有字段”，而是带有一定的保守逻辑。  
例如当前实现里，下面这些参数如果原始模板中完全没有出现，更新逻辑可能会跳过它们：

- `--moe-grouped-gemm`
- `--moe-aux-loss-coeff`
- `--moe-router-topk`
- `--expert-model-parallel-size`
- `--num-experts`

所以对于 MoE/MLA 模型，不要只在 YAML 里写参数，也要在模板脚本里把对应参数位预留出来。

### 步骤 6：准备 tokenizer 资源

模板脚本里的 tokenizer 路径必须真实可用。当前仓库已有一些示例资源放在：

- `assets/runtime/tokenizers/llama2/`
- `assets/runtime/tokenizers/qwen2/`
- `assets/runtime/tokenizers/baichuan2/`

新增模型时可以有两种做法：

1. 把 tokenizer 资源纳入仓库，并放到 `assets/runtime/tokenizers/<model>/`；
2. 在模板脚本中直接写外部绝对路径。

无论采用哪种方式，都要保证：

- PTA 模板可直接使用；
- 后续 MSA 脚本转换时，`convert_pretrain_script.py` 能从 PTA 脚本中提取到 tokenizer 参数。

### 步骤 7：如果模型有新参数需要被“变异”，更新 mutation schema

如果新增模型只是复用现有参数集合，例如 `num_layers`、`hidden_size`、`num_attention_heads`、`num_query_groups`，通常不需要改 schema。

如果你希望新增模型的专属参数也参与整网变异，则需要更新：

```text
assets/runtime/configs/mutation_schema.yaml
```

重点看两部分：

- `model_structure_args`
- `mutable_params`

接入规则是：

1. 把新参数加到 `model_structure_args`，表示它会被归入“结构相关参数池”；
2. 在 `mutable_params` 中为它定义取值范围或枚举集合；
3. 约束要贴近真实模型边界，避免生成明显非法配置。

这里要特别注意一句话：

- `mutation_schema.yaml` 只决定“参数有没有资格参与变异”；
- 不代表“参数一加进去，这一轮就一定会被实际改动”。

在当前 Task1 实现里，一个参数要真正进入常规变异，至少还要同时满足：

1. 该参数出现在本轮送入变异器的 `base_config` 中；
2. 该参数存在于 `mutable_params` 中；
3. 该参数被当前分支选入候选池；
4. 本轮采样出的新值与原值不同；
5. 后续约束修正没有把它改回去或联动调整掉。

因此：

- 如果这里只改 schema，但参数根本不在 Task1 当前使用的 `base_config` 中，那么它仍然不会被实际变异；
- 如果参数在 `base_config` 中，但采样后没有产生新值，或被后置约束修正，也可能表现为“本轮没变”；
- 只有当 schema、`base_config`、候选池选择、采样结果和约束修正都对上时，参数才会在最终 `mutating-<iter>.json` 里真正体现为一次有效变异。

当前 Task1 中，`base_config` 不是整份模型 YAML，而是从模型 YAML 中抽取并清洗后的 Transformer 侧配置。因此很多只存在于 `extra_config` 的字段，单靠把它们加进 `mutation_schema.yaml`，并不能保证会被 Task1 真实变异。

### 步骤 8：必要时补充 `parallel_mutate` 的参数合并逻辑

新增模型如果带来的是“现有 schema 中没有的新字段”，仅改 YAML 和模板还不一定够，还要检查：

```text
utils/runtime/mutate_and_forward/parallel_mutate/main.py
```

这里至少有两段逻辑可能需要补：

1. `_merge_mutation_sections()`  
   决定 `mutating-<iter>.json` 中的 `after` 配置如何回填到解析后的中间配置。
2. `update_script_parameters()`  
   决定参数最终如何写回生成的 PTA 脚本。

判断标准很简单：

- 如果一轮变异后 `mutating-<iter>.json` 里已经有变化；
- 但生成出来的 PTA 脚本里没有对应参数变化；

那就优先检查这里，而不是只改模板脚本。

---

## 3. 让新模型进入 `pta_msa` 支持列表

`Task1` 在 `pta_msa` 模式下，会通过 `utils/runtime/model_support.py` 的 `list_task1_template_supported_models()` 动态扫描：

```text
scripts/templates/pretrain_example/pretrain_mutated_*.sh
```

因此，`pta_msa` 的模型准入规则非常直接：

- 只要模板脚本存在；
- 且名字满足 `pretrain_mutated_<model>.sh`；

它就会被视为 `pta_msa` 模式可支持模型。

也就是说，`pta_msa` 通常不需要额外维护白名单，核心是把：

- `assets/runtime/model_config/<model>.yaml`
- `scripts/templates/pretrain_example/pretrain_mutated_<model>.sh`

这两份文件补齐，并保证名字一致。

---

## 4. 如果要支持 `pta_mf`，还需要补哪些东西

`pta_mf` 比 `pta_msa` 多两层要求：

1. 要能生成 MF YAML；
2. 要能把 PTA 权重转成 MF 可加载格式。

因此，新增模型要接入 `pta_mf`，至少还要做下面几步。

### 步骤 1：新增 MF 模板

在 `assets/runtime/mf_templates/` 下新增：

```text
assets/runtime/mf_templates/<model>.yaml
```

`Task1` 在 `generate_mf_script()` 中会直接按模型名查这个文件。

最稳妥的方式是：

- 先复制一个最接近的新模型模板；
- 再把 `model.model_config`、`parallel_config`、`context`、`train_dataset` 等关键字段改成目标模型版本。

### 步骤 2：检查 `mf_converter.py` 的参数映射是否够用

MF YAML 不是直接拿模板跑，而是要经过：

```text
utils/runtime/mf_converter.py
```

它会把 PTA 脚本中的命令行参数映射到 MF YAML。  
映射表集中在 `ARG_MAPPING`。

因此，如果你的新模型依赖新的 PTA 参数，而这些参数没有出现在 `ARG_MAPPING` 中，就要同步扩展这里。  
否则会出现下面这种情况：

- PTA 脚本里参数是对的；
- 生成的 MF YAML 里却没有对应字段；
- 最终表现为 MF 跑不起来，或者跑起来但配置不一致。

### 步骤 3：把模型加入 `Task1` 的权重转换支持范围

`pta_mf` 模式下，Task1 会先调用：

```text
utils/runtime/model_support.py
```

做支持性检查。当前真正控制权重转换支持范围的是：

- `TASK1_WEIGHT_CONVERT_SUPPORTED_MODELS`
- `TASK1_WEIGHT_CONVERT_MODEL_ALIASES`

如果新增模型要支持 `pta_mf`，至少要把它加入 `TASK1_WEIGHT_CONVERT_SUPPORTED_MODELS`。  
如果工具内的模型名与上游转换器接受的模型名不一致，还要在 `TASK1_WEIGHT_CONVERT_MODEL_ALIASES` 里补映射。

当前代码中的一个现成例子是：

```python
TASK1_WEIGHT_CONVERT_SUPPORTED_MODELS = ("qwen3", "deepseekv3")
TASK1_WEIGHT_CONVERT_MODEL_ALIASES = {
    "deepseekv3": "deepseek3",
}
```

这表示：

- `deepseekv3` 是 LMSV 内部名字；
- 真正传给转换器时会映射成 `deepseek3`。

### 步骤 4：确认上游 checkpoint 转换器真的支持这个模型

这里只改本仓库的白名单还不够。  
`convert_pta_checkpoint_for_mf()` 最终会调用：

- `scripts/runtime/convert.sh`
- `utils/runtime/convert_ckpt.py`
- 以及上游 `mindspeed_llm.tasks.checkpoint.convert_mg2hf`

所以还必须确认上游转换逻辑本身支持该模型。  
如果上游不支持，那么：

- 本仓库就算放开了白名单；
- 运行时仍然会在 checkpoint 转换阶段失败。

一句话概括：  
`pta_mf` 的准入条件不是“有 MF 模板”这么简单，而是“MF 配置可生成 + 权重可转换 + MF 可加载”三者都成立。

### 步骤 5：注意“有 MF 模板”不等于“Task1 已支持 pta_mf”

当前仓库里，`assets/runtime/mf_templates/` 下已经有 `qwen2.yaml`，但 Task1 的 `pta_mf` 白名单并没有放开 `qwen2`。  
这就是一个典型提醒：

- 模板存在，只说明“配置层”可能已准备；
- 不代表“Task1 已正式允许该模型走完整的 pta_mf 链路”。

所以新增模型时，要把“模板准备好”和“Task1 正式放行”当成两件事来做。

---

## 5. 推荐的新增模型接入顺序

为了减少排障成本，建议按下面顺序推进，而不是一步到位直接冲 `pta_mf`：

1. 先补 `assets/runtime/model_config/<model>.yaml`；
2. 再补 `scripts/templates/pretrain_example/pretrain_mutated_<model>.sh`；
3. 先跑通 `pta_msa`；
4. 再看是否需要扩 `mutation_schema.yaml`；
5. 确认 `parallel_mutate` 生成出来的 PTA 脚本参数是正确的；
6. 最后再补 `assets/runtime/mf_templates/<model>.yaml` 和 `model_support.py`；
7. 单独验证 `pta_mf` 的 MF YAML 生成与 checkpoint 转换。

---

## 6. 建议的验收方式

### 验收 1：只测变异产物是否生成

优先确认是否能产出：

```text
res/<model>/mutating-1.json
```

如果连这个都没有，优先检查：

- `assets/runtime/model_config/<model>.yaml` 是否存在；
- YAML 字段是否符合当前解析约定；
- `mutation_schema.yaml` 是否引入了明显非法约束。

### 验收 2：测 `pta_msa` 最小闭环

建议把 Task1 配成最小规模，例如：

- `TOTAL_ITER=1`
- `MUTNM=1`
- `SAVE_STEPS=1`
- `LOAD_STEPS=1`
- `COMPARE_MODE=pta_msa`

跑通后重点检查：

- `res/<model>/mutating-1.json`
- `pta/<model>/pretrain_mutated_<model>-1.sh`
- `ms/<model>/pretrain_mutated_<model>-1.sh`
- `output/<timestamp>/iters/iter_1/` 下的归档产物

### 验收 3：再测 `pta_mf`

只有在 `pta_msa` 稳定后，再切到：

```text
COMPARE_MODE=pta_mf
```

重点检查：

- `mf/<model>/pretrain_mutated_<model>-1.yaml` 是否生成；
- 转换后的 MF checkpoint 目录是否非空；
- MF 日志与 loss 对齐逻辑是否正常。

---

## 7. 常见问题

### 1. 名字不一致

下面这些名字必须统一：

- `tasks["1"]["MODEL_NAME"]`
- `assets/runtime/model_config/<model>.yaml`
- `scripts/templates/pretrain_example/pretrain_mutated_<model>.sh`
- `assets/runtime/mf_templates/<model>.yaml`
- `utils/runtime/model_support.py` 里的模型名或别名

任何一处不一致，都会导致 Task1 在不同阶段报“找不到模型”。

### 2. 只加了 YAML，没有加模板脚本

这时 `run_mutate()` 可能能过，但 `generate_pta_script()` 会失败，因为 Task1 找不到：

```text
scripts/templates/pretrain_example/pretrain_mutated_<model>.sh
```

### 3. 只加了模板脚本，没有把新参数接到 schema

这时模型能跑，但很多你关心的参数其实不会被变异，结果看起来像“接入成功了，但变异没生效”。

### 4. `mutating-<iter>.json` 变了，但 PTA 脚本没变

优先检查：

- `utils/runtime/mutate_and_forward/parallel_mutate/main.py`

尤其是：

- `_merge_mutation_sections()`
- `update_script_parameters()`

### 5. 有 MF 模板，但 `pta_mf` 还是不支持

优先检查：

- `utils/runtime/model_support.py` 是否放开；
- `convert_ckpt.py` 对应模型名是否可接受；
- 上游 checkpoint 转换器是否真的支持。

---

## 8. 总结

在本工具里新增一个语言模型到 Task1，最少要补齐两份文件：

- `assets/runtime/model_config/<model>.yaml`
- `scripts/templates/pretrain_example/pretrain_mutated_<model>.sh`

如果要让它真正成为“可维护的整网变异模型”，还要继续检查：

- `mutation_schema.yaml` 是否覆盖了需要变异的参数；
- `parallel_mutate` 是否把这些参数正确写回 PTA 脚本；
- 如需 `pta_mf`，`assets/runtime/mf_templates/<model>.yaml`、`utils/runtime/mf_converter.py`、`utils/runtime/model_support.py` 和上游 checkpoint 转换器是否全部同步支持。
