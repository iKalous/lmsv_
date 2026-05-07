# 检测到的缺陷汇总

> **作者**: 邹英龙
> **更新日期**: 2026-04-14

本文档汇总 LMSV (Large Model Selection and Verification) 测试过程中检测到的框架级缺陷。

## 缺陷列表

### 1. CogVideoX MSA 成功执行（成功案例）

**文件夹**: `cogvideox_msa_success/`

**描述**: 经 msadapter / transformers 环境层修复后，CogVideoX 在 MSA 环境下已可正常执行训练。

**关键修复**:
- `msadapter` bfloat16 fallback
- `LD_LIBRARY_PATH` 优先加载 conda 的 `libstdc++.so.6`
- `transformers` 调用签名关键字参数修复
- `mindspore.Tensor.__getitem__` 对 `numpy.ndarray` 索引的自动转换
- `msa_cogvideox_real.sh` 超时等待时间延长

**验证结果**:
- PTA: 成功（loss≈1.03）
- MSA: 成功（loss≈1.10，差异约 6.49%）

**状态**: ✅ 成功案例，已记录

---

### 2. OpenSora MSA safetensors 模型加载兼容性问题

**报告文件**: [bug_report_opensora_msa_huawei.md](./bug_report_opensora_msa_huawei.md)

**描述**: 在 MSA (MindSpore Adapter) 环境中运行 OpenSora1.2 时，加载 safetensors 格式的模型权重发生 `TypeError: 'UntypedStorage' object is not callable` 错误。

**根因分析**:
- msadapter 对 PyTorch 的 `torch.UntypedStorage` 进行了 patch
- 导致 safetensors 库底层调用的存储分配逻辑发生变化
- `safe_open()` 返回 `UntypedStorage` 对象而非预期的文件句柄

**影响范围**:
- 模型: OpenSora1.2（可能波及其他使用 safetensors 权重的模型）
- 环境: MSA (msadapter) 环境

**对比状态**:
- PTA: ✅ 正常执行
- MSA: ❌ 模型加载失败

**错误信息**:
```
TypeError: 'UntypedStorage' object is not callable
```

**状态**: 已记录，待修复

---

### 3. QwenVL MSA InnerInplaceIndexPut shape mismatch

**文件夹**: `qwenvl_msa_inner_inplace_indexput/`

**描述**: 在 MSA 环境下运行 QwenVL2.5 推理时，`vlm_model.py:548` 的 image token 替换操作因 shape mismatch 崩溃。

**根因分析**:
- MindSpore `InnerInplaceIndexPut` 的 shape 广播规则比 PyTorch `index_put_` 更严格
- MSA 环境下 TP=4 导致 `vit_embeds` 的 batch 维度被放大 4 倍，与索引结果 shape 不一致

**影响范围**:
- 模型: QwenVL2.5 / Qwen2.5-VL
- 环境: MSA (msadapter) 环境

**对比状态**:
- PTA: ✅ 正常执行
- MSA: ❌ `ValueError: For 'InnerInplaceIndexPut', shape mismatch...`

**错误信息**:
```
ValueError: For 'InnerInplaceIndexPut', shape mismatch: value tensor of shape [14308, 1280] cannot be broadcast to indexing result of shape [3577, 3584].
```

**状态**: 已记录，待修复

---

### 4. QwenVL MSA NPU Tensor 维度限制错误（历史 Bug）

**报告文件**: [bug_report_qwen_msa_tensor_dims_huawei.md](./bug_report_qwen_msa_tensor_dims_huawei.md)

**文件夹**: `qwen_msa_tensor_dims/`

**描述**: 在 MSA 环境下运行 QwenVL2.5 时，图像处理阶段 `image_grid_thw.prod()` 触发 `AclNN_Parameter_Error(EZ1001)`（tensor 维度 > 8）。

**状态**: 已记录。该历史 Bug 在 2026-04-14 复测时已被新的 `InnerInplaceIndexPut shape mismatch` 错误提前拦截，未再走到此路径。

---

### 5. InternVL3 PTA/MSA 精度差异（成功案例）

**文件夹**: `internvl3_pta_msa_precision_diff/`

**描述**: 成功的 LMSV 测试案例，PTA 和 MSA 环境均成功执行并完成训练，但检测到约 20.18% 的 loss 精度差异。

**执行结果**:
| 指标 | PTA | MSA | 差异 |
|------|-----|-----|------|
| Loss | 10.14411 | 12.19148 | 20.18% |
| Memory | — MB | 32805.31 MB | — |
| Time | 192 ms | 128 ms | 64 ms |

**分析**:
- 两个环境都成功完成了训练迭代
- 无异常崩溃
- 差异可能源于 PyTorch Ascend 和 MindSpore Adapter 的后端实现差异
- 属于框架级精度差异，而非功能缺陷

**状态**: ✅ 成功案例，已记录

---

## 汇总对比

| 模型 | PTA | MSA | 差异类型 |
|------|-----|-----|----------|
| CogVideoX | ✅ 正常 | ✅ 正常 | — |
| OpenSora | ✅ 正常 | ❌ safetensors加载失败 | 存储机制兼容性 |
| QwenVL | ✅ 正常 | ❌ InnerInplaceIndexPut shape mismatch | 算子语义兼容性 |
| QwenVL (历史) | ✅ 正常 | ❌ tensor维度限制(>8维) | NPU算子限制 |
| InternVL3 | ✅ 正常 | ✅ 正常(20.18%差异) | 精度差异 |

---

*最后更新: 2026-04-14*
