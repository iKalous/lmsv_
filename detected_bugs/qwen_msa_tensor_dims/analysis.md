# QwenVL2.5 MSA NPU Tensor维度限制错误 - 详细分析

> **作者**: 邹英龙
> **报告日期**: 2026-04-10

## 问题概述

在MSA (MindSpore Adapter) 环境中运行 QwenVL2.5 模型时，图像处理阶段出现 NPU tensor 维度限制错误。

## 技术细节

### 错误堆栈

```
[rank0]: Traceback (most recent call last):
[rank0]:   File "/shared/lm-sv/mm-new/MindSpeed-MM/inference_vlm.py", line 37, in <module>
[rank0]:     main()
[rank0]:   File "/shared/lm-sv/mm-new/MindSpeed-MM/inference_vlm.py", line 33, in main
[rank0]:     vlm_pipeline_dict[inference_config.pipeline_class](inference_config)()
[rank0]:   File "/shared/lm-sv/mm-new/MindSpeed-MM/mindspeed_mm/tasks/inference/pipeline/qwen2vl_pipeline.py", line 52, in __call__
[rank0]:     inputs = self.prepare_inputs(prompt=prompt, images=image, videos=video)
[rank0]:   File "/shared/lm-sv/mm-new/MindSpeed-MM/mindspeed_mm/tasks/inference/pipeline/qwen2vl_pipeline.py", line 102, in prepare_inputs
[rank0]:     inputs = self.image_processor(
[rank0]:   File "/root/anaconda3/envs/msa-m/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py", line 177, in __call__
[rank0]:     num_image_tokens = image_grid_thw[index].prod() // merge_length
[rank0]: RuntimeError: aclnnInplaceCopyGetWorkspaceSize call failed, please check!

[rank0]: ----------------------------------------------------
[rank0]: - Ascend Error Message:
[rank0]: ----------------------------------------------------
[rank0]: [PID: 2817452] 2026-04-10-17:11:48.281.315 AclNN_Parameter_Error(EZ1001): The self tensor cannot be larger than 8 dimensions.[THREAD:2820278]

[rank0]: (Please search "CANN Common Error Analysis" at https://www.mindspore.cn/en for error code description)

[rank0]: ----------------------------------------------------
[rank0]: - C++ Call Stack: (For framework developers)
[rank0]: ----------------------------------------------------
[rank0]: mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/copy.cc:44 operator()
```

### 根因分析

**错误类型**: `RuntimeError: aclnnInplaceCopyGetWorkspaceSize call failed`
**Ascend错误码**: `AclNN_Parameter_Error(EZ1001)`

**问题位置**: transformers库 Qwen2VLImageProcessor 处理图像token时失败

**关键信息**:
- 错误发生在 `image_grid_thw[index].prod()` 计算时
- NPU后端限制：tensor 维度不能超过8维
- MSA环境下MindSpore的aclnn操作触发维度限制检查

**可能原因**:
1. MSA (MindSpore Adapter) 环境下NPU tensor维度限制更严格（最大8维）
2. PyTorch Ascend (PTA) 环境下可能使用不同的kernel实现，不受此限制
3. transformers库的Qwen2VLImageProcessor在处理图像时生成了高维tensor

### 环境信息

- **模型**: QwenVL2.5 / Qwen2.5-VL-7B-Instruct (推理模式)
- **PTA环境**: mindspeed (正常执行)
- **MSA环境**: msa-m (图像处理阶段失败)
- **NPU**: 8卡Ascend 910B1
- **CANN版本**: 8.3
- **MindSpore版本**: 2.7.1
- **transformers版本**: 4.39.0+

### 复现步骤

1. 设置环境变量:
```bash
export TASK6_MODEL_NAME=qwenvl
export TASK6_TOTAL_ITER=1
export TASK6_TRAIN_ITERS=2
export PTA_PATH=/shared/lm-sv/mm-new
export MSA_PATH=/shared/lm-sv/mm-new
```

2. 执行Task6:
```bash
cd /shared/lm-sv/lmsv_rec
python -m utils.task.task6
```

3. PTA执行成功后，MSA执行时在图像处理阶段触发错误

### 相关日志

- **PTA日志**: `/shared/lm-sv/lmsv_rec/pta_logs/inference_*.log` (正常)
- **MSA日志**: `/shared/lm-sv/lmsv_rec/msrun_log/train_*.log`
- **迭代记录**: `detected_bugs/qwen_msa_tensor_dims/iters/iter_2/`
- **关键错误**: `The self tensor cannot be larger than 8 dimensions`

## 影响评估

### 受影响模型
- QwenVL2.5 / Qwen2.5-VL (确认)
- 可能受影响: 其他使用Qwen2VLImageProcessor的模型

### 受影响环境
- MSA (msadapter) 环境
- PTA环境正常

### 严重程度
- **高**: 导致MSA环境无法执行QwenVL模型推理

## 对比分析

| 环境 | 状态 | 说明 |
|------|------|------|
| PTA | ✅ 正常 | 成功执行推理，时间 81375ms |
| MSA | ❌ 失败 | 图像处理阶段tensor维度超过8维限制 |

**差异类型**: 框架底层NPU限制差异 (MindSpore/MSA tensor维度限制)

## 修复建议

### 方案1: 修改transformers库
在 `qwen2_5_vl/processing_qwen2_5_vl.py` 中增加维度检查，避免生成超过8维的tensor:
```python
# 在计算num_image_tokens前检查维度
if image_grid_thw.dim() > 8:
    # 降维处理或使用其他计算方式
    image_grid_thw = image_grid_thw.reshape(...)  # 降低维度
```

### 方案2: 使用PTA环境执行
对于QwenVL模型，暂时使用PTA环境进行推理，避开MSA的维度限制。

### 方案3: 修改MindSpeed-MM pipeline
在 `qwen2vl_pipeline.py` 的 `prepare_inputs` 方法中，预处理图像数据以避免高维tensor:
```python
# 在调用image_processor前进行预处理
images = self._preprocess_images_for_msa(images)
```

### 方案4: 升级CANN/MindSpore版本
检查新版本的CANN/MindSpore是否已修复此维度限制问题。

## 测试记录

### 2026-04-10 (首次发现)
**测试模型**: QwenVL2.5
**PTA结果**: ✅ 成功 (time=81375ms)
**MSA结果**: ❌ 失败 (RuntimeError: tensor cannot be larger than 8 dimensions)
**差异类型**: 框架底层NPU限制差异

### 2026-04-13 (复现确认)
**测试模型**: QwenVL2.5
**PTA结果**: ✅ 成功 (time=72763ms)
**MSA结果**: ❌ 失败 (RuntimeError: aclnnInplaceCopyGetWorkspaceSize call failed, AclNN_Parameter_Error(EZ1001): The self tensor cannot be larger than 8 dimensions)
**差异类型**: 框架底层NPU限制差异
**复现状态**: 与首次发现完全一致，确认为已知稳定复现bug
**相关日志**:
- PTA: `pta_logs/inference_20260413_182617.log`
- MSA: `msrun_log/train_20260413_182729.log`
- 迭代记录: `detected_bugs/qwen_msa_tensor_dims/iters/iter_2/`

---

*报告更新时间: 2026-04-13*
