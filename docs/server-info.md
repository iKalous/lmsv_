# 服务器信息文档

> **本文档记录 LMSV 项目涉及的两台服务器的所有相关信息，包括网络配置、登录方式、目录映射、NPU 状态等。**
> **创建日期**: 2026-04-19

---

## 1. 服务器概览

| 属性 | Old Server | New Server |
|------|-----------|------------|
| **主机名** | `liteserver-c007-3-0` | `liteserver-c007-2-4` |
| **公网 IP** | `1.92.208.189` | 无（仅内网） |
| **内网 IP** | `192.168.0.170` | `192.168.0.203` |
| **角色** | NFS 服务端、代码仓库、数据集存储、Jump Host | Task6 执行节点、NPU 计算节点 |
| **NPU** | 无（当前不可用） | 2x Ascend 910B1 |
| **SSH 端口** | 22 | 22 |
| **操作系统** | Linux aarch64 | Linux aarch64 |

---

## 2. 网络与登录

### 2.1 Old Server（有公网 IP）

```bash
# 直接 SSH 登录（使用 root 密码或密钥）
ssh root@1.92.208.189

# 或从内网登录
ssh root@192.168.0.170
```

### 2.2 New Server（无公网 IP，仅内网）

New Server **没有公网 IP**，无法直接从外部 SSH。必须通过 Old Server 作为 Jump Host 中转。

#### 方式一：命令行直接跳转（推荐）

```bash
# 从本地机器通过 Old Server 跳转到 New Server
ssh -J root@1.92.208.189 root@192.168.0.203

# 或分步
ssh root@1.92.208.189
ssh root@192.168.0.203
```

#### 方式二：配置 ~/.ssh/config

```
Host new-server
    HostName 192.168.0.203
    User root
    ProxyJump root@1.92.208.189

Host old-server
    HostName 1.92.208.189
    User root
```

配置后直接使用：
```bash
ssh new-server
```

#### 方式三：PyCharm 远程开发（SSH 隧道法）

由于 PyCharm 某些版本的 Jump Host UI 有问题，推荐先在本地建立端口转发隧道：

```bash
# 在本地终端执行，保持运行
ssh -L 2222:192.168.0.203:22 root@1.92.208.189
```

然后在 PyCharm 中：
- **Host**: `localhost`
- **Port**: `2222`
- **User**: `root`

### 2.3 两台机器之间的内网互通

Old Server 和 New Server 位于同一内网段（`192.168.0.0/24`），可以直接互相 SSH：

```bash
# 在 Old Server 上直接 SSH 到 New Server
ssh root@192.168.0.203

# 在 New Server 上直接 SSH 到 Old Server
ssh root@192.168.0.170
```

---

## 3. 目录映射与 NFS 挂载

### 3.1 挂载架构

```
Old Server (192.168.0.170)                    New Server (192.168.0.203)
+------------------------+                   +------------------------+
| /data2                 | --nfs4-->         | /zyl/data2             |
| /shared/mindspeed-mm   | --nfs4-->         | /zyl/mindspeed-mm      |
| /shared/lm-sv          | --nfs4-->         | /zyl/lm-sv             |
+------------------------+                   +------------------------+
```

### 3.2 New Server 当前挂载状态

```bash
# 查看挂载
mount | grep nfs4

# 输出
192.168.0.170:/data2               on /zyl/data2
192.168.0.170:/shared/mindspeed-mm on /zyl/mindspeed-mm
192.168.0.170:/shared/lm-sv        on /zyl/lm-sv
```

### 3.3 关键路径对照表

| 用途 | Old Server 路径 | New Server 路径 | 说明 |
|------|----------------|-----------------|------|
| 数据集根目录 | `/data2` | `/zyl/data2` | NFS 挂载 |
| MindSpeed-MM | `/shared/mindspeed-mm` | `/zyl/mindspeed-mm` | NFS 挂载 |
| LMSV 项目 | `/shared/lm-sv` | `/zyl/lm-sv` | NFS 挂载 |
| Task6 代码 | `/shared/lm-sv/lmsv_rec` | `/zyl/lm-sv/lmsv_rec` | 同上 |
| 环境安装包 | `/shared/lm-sv/task6_conda_envs_export` | `/zyl/lm-sv/task6_conda_envs_export` | 同上 |
| Conda 环境 | `/root/anaconda3/envs` | `/root/miniconda3/envs` | 本地安装 |
| Task6 输出 | `/data2/lm-sv/output` | `/zyl/data2/lm-sv/output` | 大文件存储 |
| mm-new | `/shared/lm-sv/mm-new` -> `/data2/lm-sv/mm-new` | `/zyl/lm-sv/mm-new` | symlink |

### 3.4 重新挂载命令（如需要）

```bash
# 在 New Server 上执行

# 1. 先卸载旧挂载
umount -f /zyl/data2 2>/dev/null || true
umount -f /zyl/mindspeed-mm 2>/dev/null || true
umount -f /zyl/lm-sv 2>/dev/null || true

# 2. 创建挂载点
mkdir -p /zyl/data2 /zyl/mindspeed-mm /zyl/lm-sv

# 3. 重新挂载
mount -t nfs4 -o rw,relatime,vers=4.2 192.168.0.170:/data2 /zyl/data2
mount -t nfs4 -o rw,relatime,vers=4.2 192.168.0.170:/shared/mindspeed-mm /zyl/mindspeed-mm
mount -t nfs4 -o rw,relatime,vers=4.2 192.168.0.170:/shared/lm-sv /zyl/lm-sv
```

### 3.5 大文件存储迁移说明

由于 `/shared/lm-sv` 所在磁盘空间有限，大体积数据已迁移至 `/data2/lm-sv`（通过 NFS 可在 New Server 的 `/zyl/data2/lm-sv` 访问）。

| 数据项 | 新位置 | 旧位置 | 说明 |
|--------|--------|--------|------|
| Task6 输出 | `/data2/lm-sv/output` | `/shared/lm-sv/lmsv_rec/output` | 已移动 |
| mm-new | `/data2/lm-sv/mm-new` | `/shared/lm-sv/mm-new` | 已移动，原位置为 symlink |
| mm-new.tar.gz | `/data2/lm-sv/mm-new.tar.gz` | `/shared/lm-sv/mm-new.tar.gz` | 已移动 |

**Old Server symlink 情况：**
```bash
# /shared/lm-sv/mm-new -> /data2/lm-sv/mm-new
ls -la /shared/lm-sv/mm-new
```

**New Server 路径解析：**
New Server 上已创建 `/data2 -> /zyl/data2` symlink，因此 `/data2/lm-sv/...` 路径可直接访问。

---

## 4. 硬件信息

### 4.1 New Server NPU 状态

```
+------------------------------------------------------------------+
| npu-smi 23.0.6                   Version: 23.0.6                 |
+---------------------------+---------------+----------------------+
| NPU   Name                | Health        | Power(W)  Temp(C)    |
| 0     910B1               | OK            | ~89W      ~48C       |
| 1     910B1               | OK            | ~91W      ~48C       |
+------------------------------------------------------------------+
| HBM Memory: 3346 MB / 65536 MB (每卡)                            |
+------------------------------------------------------------------+
```

### 4.2 CANN 环境

```bash
# CANN 工具包路径（两台机器相同）
/usr/local/Ascend/ascend-toolkit/set_env.sh

# 加载方式
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

---

## 5. Conda 环境

### 5.1 New Server 已安装环境

```bash
# 查看已安装环境
conda env list

# Task6 相关环境
mindspeed    /root/miniconda3/envs/mindspeed
msadapter    /root/miniconda3/envs/msadapter
```

### 5.2 环境安装说明

环境安装包位于：`/zyl/lm-sv/task6_conda_envs_export/`

采用"裸环境 + 一键补丁"流程：

```bash
cd /zyl/lm-sv/task6_conda_envs_export/standard_env
conda env create -f mindspeed_bare.yml -n mindspeed
conda env create -f msadapter_bare.yml -n msadapter
cd ../automated_setup
bash setup_task6_envs.sh
```

---

## 6. Task6 执行命令

### 6.1 配置 config.json

```bash
# 在 New Server 上
cat > /zyl/lm-sv/lmsv_rec/config.json << 'EOF'
{
  "task_type": 6,
  "PTA_NAME": "mindspeed",
  "MSA_NAME": "msadapter",
  "MINDSPEED_MM_PATH": "/zyl/mindspeed-mm",
  "DATASET_ROOT": "/zyl/data2/dataset",
  "SAVE_ABNORMAL_WEIGHTS": true,
  "tasks": {
    "6": {
      "MODEL_NAME": "internvl3",
      "TOTAL_ITER": 1,
      "MUTNM": 2,
      "COMPARE_MODE": "pta_msa",
      "TRAIN_ITER": 2,
      "BASE_SEED": 43,
      "PTA_MAX_RUNTIME": 3000,
      "MSA_MAX_RUNTIME": 3000
    }
  }
}
EOF
```

### 6.2 执行

```bash
cd /zyl/lm-sv/lmsv_rec
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python do.py
```

### 6.3 紧急停止

```bash
# 在 New Server 上执行
pkill -9 -f 'torchrun'
pkill -9 -f 'msrun'
pkill -9 -f 'pretrain_vlm'
pkill -9 -f 'inference_vlm'
pkill -9 -f 'inference_sora'
pkill -9 -f 'pretrain_sora'
fuser -k 6000/tcp 6001/tcp 6002/tcp 29500/tcp 29501/tcp
```

---

## 7. 常见问题速查

### Q1: PyCharm 无法连接 New Server

**原因**: New Server 没有公网 IP，必须走 Jump Host。

**解决**:
1. 先在本地建立 SSH 隧道：`ssh -L 2222:192.168.0.203:22 root@1.92.208.189`
2. PyCharm 连接 `localhost:2222`

### Q2: NFS 挂载超时

**原因**: 网络不稳定或 Old Server NFS 服务异常。

**解决**:
```bash
# 在 New Server 上
systemctl restart rpcbind 2>/dev/null || true
umount -lf /zyl/data2 && mount -t nfs4 192.168.0.170:/data2 /zyl/data2
```

### Q2.1: 训练退出时出现 `.nfs... Device or resource busy`

**原因**: Python multiprocessing 退出时在清理临时目录，但临时文件所在目录是 NFS 挂载，且仍有进程句柄占用，NFS 会把被删文件改成 `.nfs*`，这时就可能报 `Device or resource busy`。

**解决**:
```bash
# 优先把任务临时目录放到本地磁盘
export LMSV_PROJECT_TMP_ROOT=/tmp/lmsv_rec_tmp
./lmsv do
```

如果已经跑起来了，再补充排查：
```bash
lsof | grep '.nfs' || true
ps -ef | grep -E 'msrun|pretrain_gpt|python' | grep -v grep
```

### Q3: 数据集路径找不到

**原因**: `config.json` 中的 `DATASET_ROOT` 与实际挂载路径不匹配。

**解决**: 确认 New Server 上的实际挂载路径，更新 `config.json`:
```json
{
  "DATASET_ROOT": "/zyl/data2/dataset"
}
```

### Q4: Conda 环境找不到

**原因**: 环境未注册到当前 conda。

**解决**:
```bash
# 确认 conda 路径
conda info --base

# 如果环境在 miniconda3 但系统找的是 anaconda3，需要手动指定
export PATH="/root/miniconda3/bin:$PATH"
```

---

## 8. 网络拓扑图

```
                           公网
                            |
                            v
                  +---------------------+
                  |   Old Server        |
                  |   1.92.208.189      |
                  |   (Jump Host)       |
                  +----------+----------+
                             |
                   内网 192.168.0.0/24
                             |
                  +----------+----------+
                  |   New Server        |
                  |   192.168.0.203     |
                  |   (NPU 计算节点)    |
                  +---------------------+
```

---

## 9. 文档维护

- **本文档位置**: `<lm-sv-root>/lmsv_rec/docs/server-info.md`
- **修改权限**: 任何人
- **更新时机**: 服务器配置变更、IP 变更、新增挂载、环境变更时
