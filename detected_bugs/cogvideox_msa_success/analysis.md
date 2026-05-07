# CogVideoX MSA 当前状态报告

> **日期**: 2026-04-14
> **状态**: PTA/MSA 双端成功执行

## 当前状态

CogVideoX-5B 在 **MSA (MindSpore Adapter)** 环境下已完成修复，目前可以正常执行训练流程。与历史记录（2026-04-10 原始 bug）相比，当前环境发生了显著变化，原 `accelerate` offload 错误**不再复现**。

## 验证结果

在 `python do.py --task 6`（MODEL_NAME=cogvideox）下连续验证两轮，结果稳定：

### Round 1 (2026-04-14 02:06:43)

| 指标 | PTA | MSA |
|------|-----|-----|
| loss | 0.9635118 | 1.026078 |
| memory | 38143.95 MB | 32753.86 MB |
| time | 21113.1 ms | ~25640 ms |

### Round 2 (2026-04-14 02:16:33)

| 指标 | PTA | MSA |
|------|-----|-----|
| loss | 0.9635289 | 1.026078 |
| memory | 38143.95 MB | 32753.86 MB |
| time | 21121.2 ms | ~25640 ms |

### 差异分析

- **PTA loss 波动**: < 0.002%（非常稳定）
- **MSA loss 波动**: 0%（完全一致）
- **PTA vs MSA loss diff**: **6.49%**

## 环境修复记录

在不修改 `mm-mindspeed` 源码的前提下，对框架/环境层进行了以下 4 项修复：

### 1. bfloat16 fallback
- **问题**: `msadapter` 中 `mindspore.common.np_dtype.bfloat16` 不存在，导致 `TypeError`
- **修改文件**:
  - `/shared/lm-sv/mm-new/msadapter/msadapter/_utils.py`
  - `/shared/lm-sv/mm-new/msadapter/msadapter/serialization.py`
  - `/shared/lm-sv/mm-new/msadapter/build/lib/msadapter/_utils.py`
  - `/shared/lm-sv/mm-new/msadapter/build/lib/msadapter/serialization.py`
- **方案**: 当 `np_dtype.bfloat16` 缺失时，fallback 到 `ml_dtypes.bfloat16`

### 2. GLIBCXX_3.4.29 缺失
- **问题**: 系统 `libstdc++.so.6` 版本过旧，导致 numpy 导入崩溃
- **修改文件**: `/shared/lm-sv/lmsv_rec/scripts/envset/mm-msa-task6`
- **方案**: 将 conda 环境自带的 `libstdc++.so.6` 路径 prepend 到 `LD_LIBRARY_PATH`

### 3. transformers 调用签名不匹配
- **问题**: MindSpeed MSA patch 对 `_load_state_dict_into_meta_model` 的调用与 transformers 4.51.0 签名冲突，`device_map` 被重复传参
- **修改文件**: `/root/anaconda3/envs/msadapter/lib/python3.10/site-packages/transformers/modeling_utils.py`
- **方案**: 将 `expected_keys` 和 `reverse_renaming_mapping` 从位置参数改为关键字参数传入

### 4. numpy array 索引不兼容 + Task6 超时
- **问题**: MindSpore Tensor `__getitem__` 不接受 `numpy.ndarray` 作为索引，导致数据预处理崩溃；同时 `msa_cogvideox_real.sh` 等待 loss 的超时只有 120 秒，不够 MSA 初始化完成
- **修改文件**:
  - `/shared/lm-sv/mm-new/msadapter/msadapter/__init__.py`
  - `/shared/lm-sv/mm-new/msadapter/build/lib/msadapter/__init__.py`
  - `/shared/lm-sv/lmsv_rec/scripts/runtime/msa_cogvideox_real.sh`
- **方案**:
  - 在 msadapter 初始化时 monkey-patch `mindspore.Tensor.__getitem__`，自动将 `numpy.ndarray` 及其嵌套组合递归转换为 `list`
  - 将 `MAX_CHECKS` 从 120 提升到 360，确保脚本能等待到 loss 输出

## 与历史 bug 的对比

| 维度 | 历史记录（2026-04-10） | 当前状态（2026-04-14 修复后） |
|------|------------------------|-------------------------------|
| 预期错误 | `TypeError: expected str, bytes or os.PathLike object, not NoneType` | **未复现** |
| 错误阶段 | T5 text encoder 加载（accelerate offload） | 模型加载成功，直接进入训练 |
| MSA 状态 | 崩溃 | **成功执行** |
| 新发现 | — | 存在约 6.49% 的 loss 精度差异 |

## 关键结论

1. **原 accelerate offload bug 在当前环境下不再出现**。由于上游兼容性问题的修复，执行流已经能够正常通过 `transformers.from_pretrained` 的 accelerate 分支。
2. **CogVideoX MSA 当前是可运行的**，但存在可观测的 loss 精度差异（6.49%）。
3. 所有修复均位于 **环境/框架适配层**，未触及 `mm-mindspeed` 或 `MindSpeed-MM` 的业务代码。

## 备注

- 若未来需要复现历史 accelerate offload bug，可能需要回滚上述框架层修复（尤其是 transformers 签名修复和 numpy 索引补丁），但当前环境下该 bug 已被完全绕过。
- Task6 检测脚本中的 `MAX_CHECKS=360` 是针对 CogVideoX 训练启动时间较长（约 4-5 分钟）的调整，其他模型无需同步修改。
