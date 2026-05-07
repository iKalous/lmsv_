# Task1 PTA 流程全解（含 Mutate 细化）

本文聚焦 Task1 在 PTA 主链路下的执行过程，尤其详细拆解 mutate 流程。
文档目标是支持三类工作：

- 快速理解 Task1 的实际执行顺序。
- 出现失败时，按固定顺序排查。
- 扩展模型或参数空间时，明确应该改哪里。

---

## 1. Task1 在系统中的位置

Task1 的调度入口是 [do.py](../do.py)，任务主体在 [utils/task/task1.py](../utils/task/task1.py)。

整体上，Task1 每轮都遵循同一主线：

1. mutate 生成本轮变异产物。
2. 基于产物生成 PTA 训练脚本。
3. PTA-SAVE 训练并产出权重。
4. PTA-LOAD 回灌训练并产出当前轮日志。
5. 进入对比分支（MSA 或 MF）。
6. 写入状态、归档产物、刷新分析。

说明：本文以 PTA 侧为主，MSA/MF 仅在必要处提及。

---

## 2. 执行前约束（会影响 mutate 是否真正开始）

Task1 在进入 mutate 前，会先做模式和模型合法性检查：

- 检查位置： [utils/runtime/model_support.py](../utils/runtime/model_support.py)
- 调用位置： [utils/task/task1.py](../utils/task/task1.py)

规则为：

1. COMPARE_MODE=pta_msa
- 模型支持集合来自模板目录动态扫描：
  [scripts/templates/pretrain_example](../scripts/templates/pretrain_example)

2. COMPARE_MODE=pta_mf
- 模型受 TASK1_WEIGHT_CONVERT_SUPPORTED_MODELS 限制。
- 当前值见 [utils/runtime/model_support.py](../utils/runtime/model_support.py)。

这一步失败会直接终止任务，不会进入 mutate。

---

## 3. Task1 每轮 PTA 主流程

以下步骤对应 [utils/task/task1.py](../utils/task/task1.py) 的单轮执行逻辑。

1. 迭代初始化
- 清理 ms/pta/mf/msrun_log 临时目录。
- 杀残留训练进程，避免上一轮污染。

2. 执行 mutate
- 调用 run_mutate。
- 生成并校验 mutating-i.json（或失败记录）。

3. 生成 PTA 脚本
- 调用 generate_pta_script。
- 对脚本做入口替换与兼容化清洗。

4. 生成 SAVE 版本脚本并执行 PTA-SAVE
- 修改 train-iters/save，删除 load。
- 训练后检查权重产物非空。

5. 生成 LOAD 版本脚本并执行 PTA-LOAD
- 恢复原始脚本，再改 train-iters/load，删除 save。
- PTA-LOAD 允许“返回码失败但 csv 有效”判成功。

6. 处理 PTA 日志并归档
- 拷贝 training_log-i.csv 到精度日志目录。
- 归档 runtime log、脚本快照、状态文件。

---

## 4. Mutate 流程细化（重点）

本节将 mutate 拆为四层：调度层、启动层、图变异层、脚本落地层。

### 4.1 调度层：Task1 如何触发 mutate

入口函数在 [utils/task/task1.py](../utils/task/task1.py) 的 run_mutate。

核心动作：

1. 组装 MUTATE_ARGS
- 包含 model_config 目录、轮次、每轮变异参数数量、目标模型 yaml、schema 路径。

2. 注入环境变量
- 关键变量包括 MUTATE_ROUND、MUTATE_ARGS、PTAPATH。

3. 调用脚本
- 触发 [scripts/mutation/mutate-auto.sh](../scripts/mutation/mutate-auto.sh)。

### 4.2 启动层：mutate-auto.sh 做了什么

脚本位置： [scripts/mutation/mutate-auto.sh](../scripts/mutation/mutate-auto.sh)

关键行为：

1. 设置分布式参数（nproc_per_node/master_addr/master_port 等）。
2. 自动处理 master_port 占用，避免端口冲突。
3. 通过 torchrun 启动变异入口：
- [utils/runtime/mutate_and_forward/mutate_graph-auto.py](../utils/runtime/mutate_and_forward/mutate_graph-auto.py)

结论：Task1 的 mutate 不是单进程脚本，而是带分布式上下文的启动链。

### 4.3 图变异层：mutate_graph-auto.py 如何产出 mutating-i.json

核心文件： [utils/runtime/mutate_and_forward/mutate_graph-auto.py](../utils/runtime/mutate_and_forward/mutate_graph-auto.py)

流程摘要：

1. 初始化 Megatron 运行时。
2. 按 BASE_SEED + MUTATE_ROUND 重播种，保证同轮可复现。
3. 从模型 yaml 抽取图变异所需 Transformer 关键参数。
4. 构建 graph/node，选定 decoder 节点做变异。
5. 调用 ConfigMutator 生成 before/after/diff。
6. 按轮次编号写出 mutating-i.json。

重要说明：

- 变异参数空间由 schema 约束，不是任意字段都可变。
- schema 路径默认为 [assets/runtime/configs/mutation_schema.yaml](../assets/runtime/configs/mutation_schema.yaml)。

### 4.4 脚本落地层：parallel_mutate 二次收敛为可执行 PTA 脚本

mutating-i.json 不是最终训练输入。Task1 会继续调用 parallel_mutate，把变异记录落到 bash 脚本。

入口文件： [utils/runtime/mutate_and_forward/parallel_mutate/main.py](../utils/runtime/mutate_and_forward/parallel_mutate/main.py)

主要步骤：

1. BashToYaml：模板脚本转中间 yaml。
2. _merge_mutation_sections：把 after 中的 TransformerConfig/extra_config/spec 合并进配置。
3. InfoParser：配置归一化。
4. ParallelParameterMutator：并行参数变异。
5. EnhancedMegatronConfigValidator：一致性修复。
6. YamlToBash：输出最终 PTA 可执行脚本。

该流程的专门文档见：

- [docs/PTA_PARALLEL_MUTATION.md](PTA_PARALLEL_MUTATION.md)

---

## 5. Mutate 成功判定与失败判定

Task1 对 mutate 采用两段式判定：

1. mutate 脚本执行必须成功。
2. 产物必须可加载。

可加载的含义（在 [utils/task/task1.py](../utils/task/task1.py)）：

- 文件存在且非空。
- JSON 可解析。
- 根节点是 dict。
- 包含可加载的数字节点配置。

若不满足，当前轮会被标记为 MUTATION_FAILED，并跳到下一轮。

---

## 6. 与 mutation_schema 的关系

schema 文件： [assets/runtime/configs/mutation_schema.yaml](../assets/runtime/configs/mutation_schema.yaml)

你可以把它理解为“变异候选空间定义”：

1. model_structure_args
- 定义哪些参数属于结构关键参数。

2. mutable_params
- 定义参数取值方式：
  - 区间型（min_factor/max_factor/min_val/max_val）
  - 枚举型（enums）

注意：

- 参数在 schema 中可变，不代表每轮都会被改到。
- 最终是否落到训练脚本，还受 parallel_mutate 合并与校验修复影响。

---

## 7. 常见误区

1. 误区：mutating-i.json 生成了就等于本轮 mutate 成功。
- 实际：还必须通过可加载校验。

2. 误区：mutating-i.json 的 after 与最终脚本必须完全一致。
- 实际：并行参数与一致性修复会做收敛改写，这是预期行为。

3. 误区：PTA-LOAD 返回码非 0 必定失败。
- 实际：若当前轮 step csv 已有有效指标，可按成功处理。

---

## 8. 建议排障顺序（mutate 相关）

1. 先看 mutate 运行日志
- 迭代目录下 runtime_logs 的 pta_mutate_iter{i}.log。

2. 检查产物文件
- 是否存在 mutating-i.json 或 mutating-i-err.json。

3. 检查 JSON 可加载性
- 是否为空、是否可解析、是否含数字节点配置。

4. 检查 parallel_mutate 中间产物
- 查看 validated_config.yaml 与最终脚本是否一致。

5. 若最终脚本与预期不一致
- 优先核查并行约束和 validator 自动修复，而非直接判定 mutate_graph 逻辑错误。

---

## 9. 相关文档索引

- Task1 模型接入： [docs/TASK1_MODEL_EXTENSION.md](TASK1_MODEL_EXTENSION.md)
- Task1 变异约束扩展： [docs/TASK1_MUTATION_CONSTRAINT_GUIDE.md](TASK1_MUTATION_CONSTRAINT_GUIDE.md)
- Task1 结果分析架构： [docs/TASK1_RESULT_ANALYZER_ARCHITECTURE.md](TASK1_RESULT_ANALYZER_ARCHITECTURE.md)
- Task2/Task3 对齐补充： [docs/TASK23_PTA_MF_ALIGNMENT.md](TASK23_PTA_MF_ALIGNMENT.md)
- 并行变异细节： [docs/PTA_PARALLEL_MUTATION.md](PTA_PARALLEL_MUTATION.md)
- 全局架构： [docs/ARCHITECTURE.md](ARCHITECTURE.md)

---

## 10. 一句话总结

Task1 的 mutate 本质是“分布式图变异 + schema 约束 + 并行一致性修复 + 脚本落地”的组合流水线，不是单脚本随机改参数。