# Task1 按模型/参数添加变异约束指南

本文说明在 Task1 的 PTA 脚本生成流程中，如何按模型特性与参数联动关系添加“变异约束”。

我们以一个扩展模型phi为例，如何给他添加特化的约束（例如固定 cp=1，且 topk 与 num_experts 相关联）。示例中会给出具体的代码修改建议，确保约束能在 mutate 流程中生效，并且不影响其他模型的变异空间。

---

## 1. Task1 PTA 脚本生成链路（约束生效位置）

Task1 中 PTA 脚本生成主入口：

- `utils/task/task1.py` -> `generate_pta_script(...)`

该函数会调用：

- `python -m utils.runtime.mutate_and_forward.parallel_mutate`

并进入：

- `utils/runtime/mutate_and_forward/parallel_mutate/main.py`

在 `main.py` 中关键顺序是：

1. `_merge_mutation_sections(...)`：把 `mutating-i.json` 的 `after` 合并到模板配置。
2. `ParallelParameterMutator.mutate_parallel_parameters()`：变异并行参数。
3. `EnhancedMegatronConfigValidator.validate_and_fix()`：统一校验并自动修复。
4. `YamlToBashConverter.convert()`：落地为最终 PTA bash 脚本。

结论：

- 与“合法性/一致性”相关的约束，优先放在 `config_validator_moe.py`。
- 与“参数采样空间”相关的约束，放在 `ParallelParameterMutator.py`。
- 与“字段从 mutation 记录映射到配置”相关的约束，放在 `main.py` 的 `_merge_mutation_sections(...)`。

---

## 2. 约束分层建议

推荐按两层加：

1. 通用参数约束（全模型生效）
- 位置：`config_validator_moe.py`。
- 例如：`topk<=num_experts`、`tp/cp` 乘积合法、`MoE+TP` 时启用 `sequence_parallel`。

2. 模型特定约束（仅某模型生效）
- 位置：同样放在 `config_validator_moe.py`，但使用 `model_name` 分支。
- 例如：`phi` 固定 `cp=1`。

这样做的好处是：

- 不会污染其他模型。
- 同时覆盖“变异后非法组合”的兜底修复。

---

## 3. `phi` 示例：添加两个约束

下面给出一套可直接落地的实现方式。

### 3.1 第一步：把 model_name 传给校验器

目前 `main.py` 已经接收了 `model_name` 参数（来自 Task1 的 `--model_name`）。

在 `utils/runtime/mutate_and_forward/parallel_mutate/main.py` 中，把：

```python
validator = EnhancedMegatronConfigValidator(mutated_config)
```

改为：

```python
validator = EnhancedMegatronConfigValidator(mutated_config, model_name=model_name)
```

并在 `config_validator_moe.py` 的构造函数接收该参数：

```python
class EnhancedMegatronConfigValidator:
    def __init__(self, config: Dict[str, Any], model_name: str | None = None):
        self.config = config
        self.model_name = (model_name or "").strip().lower()
        ...
```

### 3.2 第二步：在 validate_and_fix 中增加模型专属检查

在 `validate_and_fix()` 中通用检查后加一行：

```python
self._check_model_specific_constraints()
```

新增方法示例：

```python
def _check_model_specific_constraints(self) -> None:
    if self.model_name == "phi":
        self._check_phi_constraints()
```

### 3.3 第三步：实现 `phi` 的两个约束

```python
def _check_phi_constraints(self) -> None:
    parallel_cfg = self.config.setdefault("parallel", {})
    moe_cfg = self.config.setdefault("moe", {})

    # 约束1: phi 固定 CP=1
    cp = parallel_cfg.get("context_parallel_size", 1)
    if cp != 1:
        self._apply_fix(
            "parallel.context_parallel_size",
            cp,
            1,
            "phi 模型约束: context_parallel_size 必须为 1"
        )

    # 约束2: phi 的 MoE topk 与其他参数匹配
    # 策略: 1 <= topk <= num_experts 且 topk 必须整除 num_experts
    num_experts = int(moe_cfg.get("num_experts", 0) or 0)
    topk = int(self._get_moe_value(moe_cfg, "moe_router_topk", default=1) or 1)

    if num_experts > 0:
        # 先做范围约束
        bounded_topk = min(max(1, topk), num_experts)
        if bounded_topk != topk:
            self._apply_fix(
                "moe.moe_router_topk",
                topk,
                bounded_topk,
                f"phi 模型约束: moe_router_topk 必须在 [1, num_experts={num_experts}]"
            )
            topk = bounded_topk

        # 再做联动约束: topk 整除 num_experts
        if num_experts % topk != 0:
            import math
            fixed_topk = math.gcd(num_experts, topk)
            fixed_topk = max(1, fixed_topk)
            self._apply_fix(
                "moe.moe_router_topk",
                topk,
                fixed_topk,
                f"phi 模型约束: 为匹配 num_experts={num_experts}, topk 调整为其公约数"
            )

    # 复用已有规则: topk=1 时需要 pre_softmax=True
```

---

## 4. 与现有逻辑的关系

`config_validator_moe.py` 已有不少通用规则（例如 `topk<=num_experts`、`topk=1` 时 `pre_softmax=True`）。

`phi` 约束建议作为“模型专属补充规则”，不要把 `phi` 特性写进通用规则里。否则会影响其他模型的变异空间。

---

## 5. 最小改动清单

如果只实现本需求，最小改动是：

1. `parallel_mutate/main.py`
- 将 `model_name` 传给 `EnhancedMegatronConfigValidator`。

2. `parallel_mutate/config_validator_moe.py`
- 构造函数增加 `model_name`。
- `validate_and_fix()` 增加模型专属约束调用。
- 新增 `_check_model_specific_constraints()` 与 `_check_phi_constraints()`。

这样即可在 Task1 中实现：

- `phi` 模型 `cp=1` 强约束。
- `phi` 模型 `moe_router_topk` 与其他参数匹配约束。
