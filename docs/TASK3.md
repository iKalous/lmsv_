# Task3 执行全流程与排障说明

本文档说明 `task3` 在 `pta_msa` / `pta_mf` 两种模式下的执行链路、关键产物、常见失败点，以及推荐排障方式。

## 1. Task3 定位

- 入口: `do.py` -> `utils.control.protect.task(3, params)` -> `utils.task.task3.main`
- 目标: 对多模型进行模块间组合变异，并在不同后端链路上完成可比对验证。
- 模式:
  - `COMPARE_MODE=pta_msa`: PTA 与 MSA 对比
  - `COMPARE_MODE=pta_mf`: PTA 与 MindFormers 对比

## 2. 迭代执行主链路

以单轮 `iter=i` 为例，`utils/task/task3.py` 的执行顺序如下。

1. PTA mutate
- 函数: `run_pta_mutate`
- 产物: `res/<model>/mutating-i.json` 等

2. PTA-SAVE
- 函数: `run_pta_verify_stage(..., mode="save")`
- 目的: 生成共享权重 `iter{i}.pth`
- 关键环境变量:
  - `LMSV_ENABLE_SUBMODULE_SHARED_WEIGHT_PATCH=1`
  - `LMSV_SHARED_WEIGHT_PATH=<...>/iter{i}.pth`
  - `LMSV_SHARED_WEIGHT_MODE=save`

   
3. PTA->MF 权重转换（仅 `pta_mf`）
- 函数: `convert_shared_weight_for_mf`
- 开关: `ENABLE_MF_WEIGHT_LOAD`
- 说明:
  - `utils/task/task3.py` 运行时已支持从参数读取该开关
  - 当前是否在 WebUI / `genconf.py` 中提供单独表单项，需要以界面实现为准
- 步骤: `pth -> npz -> ckpt`
- 产物: `iter{i}.ckpt`

4. PTA-LOAD
- 函数: `run_pta_verify_stage(..., mode="load")`
- 目的: 使用共享权重完成 PTA 回灌验证

5. 对端验证
- `pta_msa`: `run_msa_verify_load`
- `pta_mf`: `run_mf_verify` 调用 `utils/runtime/mf_mutate_and_forward/load_and_forward_graph.py`

6. 结果校验
- 校验: PTA 与 MSA/MF 的同轮结果记录是否存在
- 对比: PTA 与 MSA/MF 的同轮 loss 结果

## 3. 关键产物与日志位置

- 每轮日志: `output/<timestamp>/iters/iter_i/runtime_logs/`
  - `pta_mutate_iter{i}.log`
  - `pta_save_iter{i}.log`
  - `pta_load_iter{i}.log`
  - `msa_load_iter{i}.log`（若启用）
  - `mf_iter{i}.log`（若启用）
  - `convert_iter{i}.log`（若启用 `pta_mf`）

- 每轮脚本快照: `output/<timestamp>/iters/iter_i/scripts/`

- MSA worker 日志目录: `output/<timestamp>/iters/iter_i/msrun_log/`
  - `task3` 每轮只归档 `msrun_log/` 整目录
  - 不再额外在 `iter_i/` 根目录单独复制 `worker_x.log`

- 共享权重临时目录:
  - 默认来自 `SHARED_WEIGHT_TMP_ROOT`
  - 按 run 隔离到 `tmp/task3/shared_weight/task3_<run>_<pid>/`

## 3.1 多卡默认行为

- `task3` 默认按当前进程可见卡数自动启用多卡
- 自动探测优先读取:
  - `ASCEND_RT_VISIBLE_DEVICES`
  - `ASCEND_VISIBLE_DEVICES`
  - `NPU_VISIBLE_DEVICES`
  - `CUDA_VISIBLE_DEVICES`
  - `LOCAL_WORLD_SIZE` / `WORLD_SIZE`
- 仅当显式传入 `TARGET_TENSOR_PARALLEL_SIZE` / `TARGET_PIPELINE_PARALLEL_SIZE` / `TARGET_EXPERT_PARALLEL_SIZE` / `TARGET_NPUS_PER_NODE` 时，才覆盖自动策略

## 4. `pta_mf` 模式的配置对齐机制

`pta_mf` 模式中，PTA 侧会把运行时对齐信息写回变异 JSON，MF 侧读取后进行关键字段对齐。

- PTA 写入:
  - attention 对齐: `__pta_attention_align__`
  - decoder 对齐: `__pta_decoder_align__`

- MF 读取:
  - 文件: `utils/runtime/mf_mutate_and_forward/graph.py`
  - 默认会读取多项关键字段

## 5. 常见问题: `linear_qkv.bias` 未加载导致失败

### 5.1 典型日志

在 `mf_iter{i}.log` 出现:

- `attention未加载参数样例: node_block_*.self_attention.linear_qkv.bias`
- `attention参数未完全加载，严格模式下判定失败（LMSV_STRICT_ATTN_PARAM_LOAD=1）`
- `RuntimeError: 共享ckpt加载失败`

### 5.2 根因

常见根因是 PTA 与 MF 在 `add_qkv_bias` 上不一致:

- PTA: `add_qkv_bias=True`，共享 ckpt 含 QKV bias 语义
- MF: 未对齐该字段，构图期 `add_qkv_bias=False` 或不一致

结果是 `load_param_into_net` 后 attention 参数出现未加载，严格模式直接判失败。

### 5.3 当前默认修复策略

Task3 的 `run_mf_verify` 已默认导出:

- `LMSV_ALIGN_ADD_QKV_BIAS=${LMSV_ALIGN_ADD_QKV_BIAS:-1}`

含义:

- 默认开启 PTA->MF 的 `add_qkv_bias` 对齐，避免 `linear_qkv.bias` 丢载。
- 若需要回退旧行为，可显式设置 `LMSV_ALIGN_ADD_QKV_BIAS=0` 覆盖。

### 5.4 排障检查顺序

1. 检查 `mf_iter{i}.log` 是否出现 `decoder字段 add_qkv_bias 默认不对齐`。
2. 检查同日志中 `add_qkv_bias: PTA=... | MF=...` 是否不一致。
3. 检查 `attention参数加载统计` 的 `unloaded` 是否集中在 `linear_qkv.bias`。
4. 确认执行脚本中已导出 `LMSV_ALIGN_ADD_QKV_BIAS`。
5. 若仍失败，再检查共享 ckpt 路径、转换日志和字段命名映射。

## 6. 扩展流程文档（Task2/Task3 的 MF 部分）

关于“静态图下扩展模块内变异/模块间变异的新模块到工具”的完整接入流程，已单独整理为文档：

- `docs/TASK23_MF_EXTENSION.md`

## 7. 建议运行命令

从仓库根目录:

```bash
./lmsv conf
./lmsv do
./lmsv analyze --latest
```

若仅复核最新一次结果，可先看:

```bash
tail -n 200 output/<latest>/iters/iter_1/runtime_logs/mf_iter1.log
```
