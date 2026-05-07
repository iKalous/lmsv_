# OpenSora MSA 模型加载错误分析

> **状态**: 已知稳定复现 bug
> **日期**: 2026-04-14

## 问题描述

在 MSA (MindSpore Adapter) 环境下运行 OpenSora1.2 推理时，模型权重加载阶段崩溃，PTA 环境正常执行。

## 错误信息

```
TypeError: 'UntypedStorage' object is not callable
```

### 完整堆栈

```
[rank0]: Traceback (most recent call last):
[rank0]:   File "/root/anaconda3/envs/msa-m/lib/python3.10/site-packages/diffusers/models/model_loading_utils.py", line 105, in load_state_dict
[rank0]:     return safetensors.torch.load_file(checkpoint_file, device="cpu")
[rank0]:   File "/root/anaconda3/envs/msa-m/lib/python3.10/site-packages/safetensors/torch.py", line 336, in load_file
[rank0]:     with safe_open(filename, framework="pt", device=device) as f:
[rank0]: TypeError: 'UntypedStorage' object is not callable

[rank0]:   File "/shared/lm-sv/mm-new/MindSpeed-MM/inference_sora.py", line 111, in main]
[rank0]:     sora_pipeline = prepare_pipeline(args, device)
[rank0]: OSError: Unable to load weights from checkpoint file for '/data2/dataset/opensora1.2/sd-vae-ft-ema/diffusion_pytorch_model.safetensors'
```

## 根因分析

1. **msadapter 对 `torch.UntypedStorage` 的 patch**：msadapter 为了适配 MindSpore 的内存管理机制，修改了 PyTorch 底层存储类。
2. **safetensors 兼容性破坏**：`safetensors.torch.load_file()` 在 MSA 环境下调用被 patch 后的 `torch.UntypedStorage` 时，返回了对象本身而非预期的文件句柄，导致 `safe_open` 调用失败。
3. **触发路径**：diffusers 加载 `sd-vae-ft-ema/diffusion_pytorch_model.safetensors` 时必然走到 `safetensors.torch.load_file()`。

## 环境信息

- **模型**: OpenSora1.2 (推理模式)
- **PTA环境**: mindspeed (正常)
- **MSA环境**: msa-m (崩溃)
- **NPU**: 8 卡 Ascend 910B1

## 复现步骤

```bash
cd /shared/lm-sv/lmsv_rec
# 设置 config.json: tasks["6"].MODEL_NAME = "opensora"
python do.py --task 6
```

## 影响评估

- **严重程度**: 高 — 阻止 MSA 环境下 OpenSora 的执行
- **影响范围**: OpenSora1.2 (确认)；其他使用 safetensors 格式权重的模型可能同样受影响

## 对比

| 环境 | 状态 |
|------|------|
| PTA | 正常 |
| MSA | 失败 (`TypeError: 'UntypedStorage' object is not callable`) |

## 结论

该 bug 在当前环境下**稳定复现**，与 `detected_bugs` 历史记录一致。在不修改 MindSpeed-MM 源码的前提下，根因是 msadapter 与 safetensors 的底层兼容性问题，无法通过 dtype 格式等上层配置规避。
