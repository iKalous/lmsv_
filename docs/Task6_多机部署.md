# Task6 多机部署文档

> **目标**: 将 Task6 从单机部署拓展到多机器部署
> **原则**: 最小必要修改，所有路径通过 `config.json` 配置动态推导

---

## 1. 机器信息示例

| 属性 | Master | Worker |
|------|--------|--------|
| 内网 IP | 192.168.0.170 | 192.168.0.203 |
| NPU | 8x Ascend 910B | 8x Ascend 910B |
| 角色 | 协调+变异+执行 | 执行 |

---

## 2. 新机器最小配置

### 2.1 SSH 免密登录

Master 节点必须能免密 SSH 到所有 Worker 节点：

```bash
# Master 上执行
ssh-copy-id root@192.168.0.203
ssh root@192.168.0.203 echo "OK"  # 验证无需密码
```

### 2.2 代码与数据路径

Worker 节点上 `lmsv_rec` 代码仓库的位置**必须与 Master 不同或相同均可**，通过 `config.json` 的 `MULTI_NODE.OTHER_NODES[].LMSV_PATH` 指定。

**推荐方案（NFS 共享）**:
- Master: `/data2/lm-sv/lmsv_rec`
- Worker: `/zyl/lm-sv/lmsv_rec`（NFS 挂载或 symlink）

**独立部署方案**:
- Worker 独立 clone 代码仓库到任意路径，在 `config.json` 中配置对应 `LMSV_PATH`

### 2.3 Conda 环境

Worker 节点需要与 Master 相同的 conda 环境：
- `PTA_NAME`: 对应 Master 的 `PTA_NAME`（如 `mindspeed`）
- `MSA_NAME`: 对应 Master 的 `MSA_NAME`（如 `msadapter`）

环境安装包位于 `task6_conda_envs_export/`，Worker 节点可独立安装：

```bash
cd /path/to/task6_conda_envs_export/standard_env
# 创建裸环境
conda env create -f mindspeed_bare.yml
conda env create -f msadapter_bare.yml
# 应用定制化补丁
cd ../automated_setup
bash setup_task6_envs.sh
```

### 2.4 MindSpeed-MM 与依赖仓库

Worker 节点需要以下目录（路径任意，通过 `config.json` 映射）：

| 目录 | 说明 |
|------|------|
| `Megatron-LM` | Megatron 框架 |
| `MindSpeed` | MindSpeed 加速库 |
| `MindSpeed-MM` | MindSpeed-MM 多模态框架 |
| `msadapter` | MSA 适配层（仅 MSA 环境需要） |

**推荐方案（NFS 共享）**:
```bash
# Worker 上创建 symlink 指向 NFS 挂载
ln -s /zyl/mindspeed-mm /zyl/mindspeed-mm
ln -s /zyl/lm-sv /zyl/lm-sv
```

### 2.5 数据集

Worker 节点需要能访问 `DATASET_ROOT` 指定的数据集目录。推荐通过 NFS 共享。

---

## 3. 唯一配置：config.json

多机部署**只需要修改 `config.json`**，不需要修改任何代码。

### 3.1 配置示例

```json
{
    "task_type": 6,
    "PTA_NAME": "mindspeed",
    "PTA_PATH": "/shared/lm-sv/mm-new",
    "MSA_NAME": "msadapter",
    "MSA_PATH": "/shared/lm-sv/mm-new",
    "MF_NAME": "mindf_py311",
    "MINDSPEED_MM_PATH": "/shared/mindspeed-mm",
    "DATASET_ROOT": "/data2/dataset",
    "SAVE_ABNORMAL_WEIGHTS": false,
    "MULTI_NODE": {
        "ENABLED": true,
        "MASTER_ADDR": "192.168.0.170",
        "OTHER_NODES": [
            {
                "HOST": "192.168.0.203",
                "LMSV_PATH": "/zyl/lm-sv/lmsv_rec",
                "PTA_NAME": "mindspeed",
                "MSA_NAME": "msadapter",
                "PTA_PATH": "/zyl/lm-sv/mm-new",
                "MSA_PATH": "/zyl/lm-sv/mm-new"
            }
        ]
    },
    "tasks": {
        "6": {
            "MODEL_NAME": "cogvideox",
            "TOTAL_ITER": 2,
            "MUTNM": 2,
            "COMPARE_MODE": "pta_msa",
            "TRAIN_ITER": 2,
            "BASE_SEED": 42,
            "PTA_MAX_RUNTIME": 1800,
            "MSA_MAX_RUNTIME": 1800
        }
    },
    "MODEL_NAME": "cogvideox",
    "TOTAL_ITER": 2,
    "TRAIN_ITER": 2
}
```

### 3.2 配置字段说明

#### 全局路径（Master 节点本地路径）

| 字段 | 说明 | 示例 |
|------|------|------|
| `PTA_PATH` | Master 上 PTA workspace 根目录 | `/shared/lm-sv/mm-new` |
| `MSA_PATH` | Master 上 MSA workspace 根目录 | `/shared/lm-sv/mm-new` |
| `MINDSPEED_MM_PATH` | Master 上 MindSpeed-MM 路径 | `/shared/mindspeed-mm` |
| `DATASET_ROOT` | Master 上数据集根目录 | `/data2/dataset` |

#### MULTI_NODE 配置

| 字段 | 说明 |
|------|------|
| `ENABLED` | 是否启用多机模式 |
| `MASTER_ADDR` | Master 节点 IP（用于 HCCL 和 torchrun rendezvous）|
| `OTHER_NODES[].HOST` | Worker 节点 IP |
| `OTHER_NODES[].LMSV_PATH` | Worker 上 `lmsv_rec` 代码路径 |
| `OTHER_NODES[].PTA_PATH` | Worker 上 PTA workspace 路径 |
| `OTHER_NODES[].MSA_PATH` | Worker 上 MSA workspace 路径 |
| `OTHER_NODES[].PTA_NAME` | Worker 上 PTA conda 环境名 |
| `OTHER_NODES[].MSA_NAME` | Worker 上 MSA conda 环境名 |

### 3.3 路径映射原理

Task6 通过 `config.json` 中 Master/Worker 的路径对**动态推导**映射关系，**无需硬编码**。

例如：
- Master `PTA_PATH` = `/shared/lm-sv/mm-new`
- Worker `PTA_PATH` = `/zyl/lm-sv/mm-new`
- 共同后缀：`/lm-sv/mm-new`
- 推导映射：`/shared` → `/zyl`

所有路径（数据集、workspace、代码仓库）均通过此机制自动映射。

---

## 4. 运行

配置完成后，直接执行：

```bash
cd /data2/lm-sv/lmsv_rec
python3 do.py
```

或通过交互式配置：

```bash
python3 do.py conf  # 选择 Task6，按提示配置
```

---

## 5. 常见问题

### 5.1 HCCL 初始化超时

**现象**: 多机训练 hang 在 HCCL communicator 初始化

**排查**:
1. 检查 `MASTER_ADDR` 是否为 Master 的内网 IP
2. 检查 Worker 能否 `ping MASTER_ADDR`
3. 检查 `HCCL_IF_BASE_PORT` 范围（默认 61000-61015）是否被占用
4. 检查防火墙是否放行了相关端口

### 5.2 路径映射失败

**现象**: 日志中出现 "路径不在项目目录内，无法同步到远端"

**排查**:
1. 确认 `config.json` 中 `OTHER_NODES[].LMSV_PATH` 指向 Worker 上实际存在的路径
2. 确认 `PTA_PATH`/`MSA_PATH` 在 Worker 上存在
3. 检查 NFS 挂载状态：`df -h` 查看 `/zyl` 是否挂载

### 5.3 远程 conda 环境找不到

**现象**: 远程执行时报 "conda activate mindspeed" 失败

**排查**:
1. 在 Worker 上手动执行 `conda activate mindspeed`，确认环境存在
2. 检查 `conda init` 是否已应用到当前 shell（`~/.bashrc` 中有 conda 初始化块）
3. 确认 `OTHER_NODES[].PTA_NAME` 与 Worker 上实际环境名一致

### 5.4 NFS 缓存不一致

**现象**: Worker 生成的日志 Master 看不到，或延迟看到

**现象**: 这是 NFS 客户端缓存问题。Task6 已通过 `scp` 主动同步 worker 日志到本地，无需手动处理。

---

## 6. 约束

- 不要同时运行多个 Task6 进程
- 不要修改 `task6.py` 中的路径硬编码（已全部改为动态推导）
- 所有路径配置集中在 `config.json`
