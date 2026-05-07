# Task6 多机部署详细过程文档

> 创建时间: 2026-04-25
> 状态: 10轮跨机验证进行中（iter_1 diff=1.52%通过，iter_2-10运行中）
> 机器: old-server (192.168.0.170) + new-server (192.168.0.203)

---

## 一、当前整体进展

| 组件 | 单机 | 跨机(2节点) | 状态 |
|------|------|------------|------|
| CogVideoX PTA | 已验证 | **已通过** | 可正常执行10iter训练 |
| CogVideoX MSA | 已验证 | **已通过** | 跨机精度已修复，diff=1.54%与单机一致 |
| InternVL3 PTA | 未验证 | 待验证 | - |
| InternVL3 MSA | 未验证 | 待验证 | - |
| OpenSora PTA | 未验证 | 待验证 | - |
| QwenVL PTA | 未验证 | 待验证 | - |

**最近一次单轮验证结果 (2026-04-25 17:07-17:20)**:
- PTA: loss=0.70338, mem=34126.1MB, time=10849.1ms (成功)
- MSA: loss=0.69259, mem=31336.9MB, time=12189.5ms (成功，diff=1.54%)

---

## 二、架构设计

### 2.1 多机部署模式

参照Task4-5的多机架构，Task6采用 **SSH + rsync + ThreadPoolExecutor** 并发启动本地/远程任务:

```
old-server (192.168.0.170, MASTER, NODE_RANK=0)
  ├─ PTA: torchrun --nproc_per_node 8 --nnodes 2 --node_rank 0
  └─ MSA: msrun --worker_num 16 --local_worker_num 8 --node_rank 0
                --master_addr 192.168.0.170 --master_port 29505

new-server (192.168.0.203, NODE_RANK=1)
  ├─ PTA: torchrun --nproc_per_node 8 --nnodes 2 --node_rank 1
  └─ MSA: msrun --worker_num 16 --local_worker_num 8 --node_rank 1
                --master_addr 192.168.0.170 --master_port 29505
```

### 2.2 路径映射

远程节点通过 `/zyl` 软链接访问NFS共享的 `/data2`:

| 本地路径 | 远程映射路径 |
|---------|------------|
| `/data2/lm-sv/lmsv_rec` | `/zyl/lm-sv/lmsv_rec` |
| `/data2/dataset` | `/zyl/data2/dataset` |
| `/shared/lm-sv/mm-new` | `/shared/lm-sv/mm-new` (共享) |
| `/shared/mindspeed-mm/MindSpeed-MM` | `/shared/mindspeed-mm/MindSpeed-MM` (共享) |

### 2.3 核心修改文件

```
/data2/lm-sv/lmsv_rec/utils/task/task6.py          # 多机支持核心实现
/data2/lm-sv/lmsv_rec/scripts/runtime/mm_pta_cogvideox.sh
/data2/lm-sv/lmsv_rec/scripts/runtime/pta_cogvideox_real.sh
/data2/lm-sv/lmsv_rec/scripts/runtime/mm_msa_cogvideox.sh
/data2/lm-sv/lmsv_rec/scripts/runtime/msa_cogvideox_real.sh
/data2/lm-sv/lmsv_rec/scripts/envset/mm-pta-task6.sh
/data2/lm-sv/lmsv_rec/scripts/envset/mm-msa-task6.sh
```

---

## 三、已解决的问题和修复

### 3.1 HCCL_DETERMINISTIC 导致多机死锁 (ROOT CAUSE)

**问题**: `HCCL_DETERMINISTIC=true` 在多机模式下导致HCCL同步死锁

**修复** (`task6.py`):
```python
def main(params: Dict[str, Any]):
    # ...
    if not _init_config(params):
        return 1
    # 仅在单机设置硬件确定性标志
    if not Config.MULTI_NODE_ENABLED:
        os.environ["HCCL_DETERMINISTIC"] = "true"
        os.environ["NCCL_DETERMINISTIC"] = "1"
```

关键: 必须在 `_init_config()` 之后设置，因为 `Config.MULTI_NODE_ENABLED` 默认False

### 3.2 awk 除零错误

**问题**: `pta_cogvideox_real.sh` 中 STEP_TIME="N/A" 时 awk 除法报错

**修复**:
```bash
# BEFORE
SPS=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`

# AFTER
if [ "$STEP_TIME" = "N/A" ]; then
    SPS="N/A"
else
    SPS=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
fi
```

### 3.3 Simplejson ImportError (远程节点)

**问题**: 远程msadapter环境缺少simplejson，导致msrun集群初始化失败

**修复**:
```bash
ssh 192.168.0.203 'conda run -n msadapter pip install simplejson'
# 本地同步升级: pip install simplejson==4.1.1
```

### 3.4 多进程竞争

**问题**: 同时启动多个测试进程导致路径映射混乱

**修复**: 在 `task6.py:main()` 开头增加强制kill逻辑:
```python
def main(params):
    utils.control.clean.kill_pretraingpt()
    _kill_remote_processes()  # 新增
    # ...
```

### 3.5 PIPE缓冲导致hang

**问题**: `subprocess.Popen(stdout=PIPE)` 在多机长运行时可能阻塞

**修复**: 直接重定向到日志文件:
```python
with open(exec_log_file, 'w') as log_f:
    process = subprocess.Popen(
        cmd, shell=True, stdout=log_f, stderr=subprocess.STDOUT, ...
    )
```

### 3.6 HCCL端口冲突

**问题**: HCCL通信端口被占用导致error code 7

**修复**:
```bash
export HCCL_IF_BASE_PORT=61000
sysctl -w net.ipv4.ip_local_reserved_ports="61000-61015"
```

### 3.7 pta_memory_wrapper 僵尸进程

**问题**: torchrun启动的子进程未被清理

**修复**: `clean.py:kill_pretraingpt()` 增加对 `pta_memory_wrapper` 的kill

---

### 3.8 MSA多机日志选择bug导致diff虚高

**问题**: `msa_cogvideox_real.sh` 和 `msa_internvl3_8B_real.sh` 的日志等待循环在找到第一个包含loss的worker日志后立即break，导致选择了错误的worker日志文件。

在多机MSA模式下，megatron的 `print_rank_last` 将完整的训练loss轨迹输出到最后一个rank的日志中（如worker_15.log），而不是worker_0.log。脚本错误地选择了worker_0.log（只包含iteration 1-9的loss），导致task6.py提取了step 9的loss而不是step 10的loss。

**影响**: 
- PTA提取step 10 loss = 0.7034
- MSA错误提取step 9 loss = 0.7280
- 报告diff = 3.50%（被夸大）
- 实际diff（使用正确step 10 loss = 0.6925）= 1.54%

**修复**: 修改等待循环逻辑，在进程结束后遍历所有worker日志，选择包含最多loss行的那个：
```bash
# 选择包含最多loss行的worker日志（确保获取最后一个iteration的loss）
BEST_LOG=""
BEST_LOSS_COUNT=0
for log in ${MINDSPEED_MM_PATH}/msrun_log/worker_*.log; do
    if [ -f "$log" ]; then
        LOSS_COUNT=$(grep -c "loss:" "$log" 2>/dev/null || echo 0)
        if [ "$LOSS_COUNT" -gt "$BEST_LOSS_COUNT" ]; then
            BEST_LOSS_COUNT=$LOSS_COUNT
            BEST_LOG="$log"
        fi
    fi
done
if [ -n "$BEST_LOG" ]; then
    MSRUN_LOG="$BEST_LOG"
    echo "Selected worker log with most loss entries: $MSRUN_LOG ($BEST_LOSS_COUNT loss lines)"
fi
```

**修复文件**: 
- `scripts/runtime/msa_cogvideox_real.sh`
- `scripts/runtime/msa_internvl3_8B_real.sh`

**验证结果**: 修复后iter_1 diff = 1.54%，与单机基准1.72%-1.79%一致。


### 3.9 msrun_log 多节点竞争清除导致日志丢失

**问题**: 多机模式下两个节点同时执行 `rm -rf ${MINDSPEED_MM_PATH}/msrun_log`，由于NFS共享目录，后执行的节点会删除先执行节点已创建的日志文件，导致日志丢失或选中旧日志。

**修复** (`msa_cogvideox_real.sh`, `msa_internvl3_8B_real.sh`):
```bash
# 仅在master节点清空日志，避免NFS竞争
if [ "${NODE_RANK:-0}" -eq 0 ]; then
    rm -rf ${MINDSPEED_MM_PATH}/msrun_log
fi
mkdir -p ${MINDSPEED_MM_PATH}/msrun_log
```

### 3.10 MSA日志异步写入导致BEST_LOG选择过早

**问题**: 多机模式下远程worker（worker_15）的日志写入比本地进程检测延迟约10-30秒。`BEST_LOG` 选择逻辑在wait loop结束后立即执行，此时worker_15.log可能只写入9个iteration，导致选中缺失最后一轮loss的日志。

**影响**:
- iter_2: 选中9 loss行的日志，diff虚高至3.54%
- iter_1/3/4: 因task6.py fallback或时机巧合，diff正常（1.53%-1.54%）

**修复**: 在wait loop后增加稳定化等待，直到检测到 `TRAIN_ITERS` 个loss条目或60秒超时:
```bash
# 额外等待：确保远程worker日志完全刷新，且找到包含所有iteration的日志
echo "Stabilizing: waiting for all iterations to be written to logs..."
FINAL_WAIT_COUNT=0
MAX_FINAL_WAIT=60
TARGET_ITERATIONS=${TRAIN_ITERS:-10}
while [ "$FINAL_WAIT_COUNT" -lt "$MAX_FINAL_WAIT" ]; do
    FINAL_WAIT_COUNT=$((FINAL_WAIT_COUNT + 1))
    
    CURRENT_BEST_COUNT=0
    for log in ${MINDSPEED_MM_PATH}/msrun_log/worker_*.log; do
        if [ -f "$log" ]; then
            COUNT=$(grep -c "loss:" "$log" 2>/dev/null || echo 0)
            if [ "$COUNT" -gt "$CURRENT_BEST_COUNT" ]; then
                CURRENT_BEST_COUNT=$COUNT
            fi
        fi
    done
    
    if [ "$CURRENT_BEST_COUNT" -ge "$TARGET_ITERATIONS" ]; then
        echo "Found log with $CURRENT_BEST_COUNT/$TARGET_ITERATIONS loss entries"
        break
    fi
    
    sleep 1
done
```

**验证结果**: 修复后iter_4 diff = 1.53%，与单机基准一致。

## 四、MSA跨机失败详细分析

### 4.1 现象

- 所有16个worker (8本地+8远程) 成功注册到meta server
- 集群在1秒内完成初始化 (11:33:51)
- meta server持续报告"16 alive nodes"
- ~3分38秒后，node 0被声明timed out
- 报错: `worker 0 is the first one timed out`

### 4.2 进程分析

| PID | 角色 | 日志文件 | 声称Rank |
|-----|------|---------|---------|
| 2462900 | scheduler子进程 | scheduler.log | Rank 0 |
| 2462901 | worker_0 | worker_0.log | Rank 0 |
| 2462903 | worker_1 | worker_1.log | Rank 1 |
| ... | ... | ... | ... |
| 1576808 | worker_8(远程) | worker_8.log | Rank 8 |

**核心发现**: scheduler子进程和worker_0进程都声称是 Rank 0 | Local Rank 0

### 4.3 根因推断

MindSpore msrun的 `_MetaServerNode.run()` 实现中，scheduler子进程**始终会执行用户的Python脚本** (`pretrain_sora.py`)，并设置:
- `RANK_ID=0`
- `MS_ROLE=MS_SCHED`

而worker_0通过 `_get_node_id_and_log_path(local_rank=0)` 也获得:
- `RANK_ID=0` (node_rank * local_worker_num + index = 0*8+0)
- `MS_ROLE=MS_WORKER`

当两者都调用 `mindspore.communication.init()` 时:
- scheduler进程: backend被强制改为 `mccl`，调用 `init_cluster()`
- worker_0: backend为 `nccl`(经msadapter转换)，调用 `init_cluster()`

两者都通过 `init_cluster()` 尝试参与集群拓扑，但**共享同一个rank 0**，导致:
1. scheduler的 `init_cluster()` 等待所有worker汇报拓扑状态
2. worker_0的 `init_cluster()` 因rank冲突而无法正确完成
3. meta server的 `UpdateTopoState` 在超时后声明 node 0 timed out

### 4.4 为什么PTA能通但MSA不行

| 维度 | PTA | MSA |
|------|-----|-----|
| 启动器 | torchrun | msrun |
| 调度模式 | torchrun每个rank都是平等的worker进程 | msrun有scheduler+worker层级架构 |
| scheduler角色 | 无，torchrun父进程只负责spawn | msrun scheduler子进程**也执行用户脚本** |
| rank分配 | torchrun通过环境变量分配唯一rank | msrun的scheduler固定RANK_ID=0，与worker_0冲突 |
| 分布式后端 | torch_npu直接转HCCL | msadapter -> MindSpore communication -> init_cluster() |

### 4.5 待尝试的解决方案

**方案A: 修改 --join 参数**
```bash
# 当前: --join=False (msrun立即退出，靠pgrep检测worker)
# 建议: --join=True (msrun监控所有进程并返回结果)
```

**方案B: 清除msrun_log缓存标记**
当前脚本用 `.msrun_first_run_done` 标记控制是否清空日志目录，导致多次运行时旧日志残留。应改为每次运行前无条件清空。

**方案C: 使用 RANK_TABLE_FILE 替代 msrun动态组网**
MindSpore支持静态rank table文件方式启动，可避免scheduler进程参与计算:
```bash
# 生成rank_table.json
# 各节点直接运行python pretrain_sora.py，通过RANK_TABLE_FILE环境变量组网
```

**方案D: 使用 mpirun 替代 msrun**
```bash
mpirun -n 16 --hostfile hosts.txt python pretrain_sora.py ...
```
需要验证msadapter对mpirun的兼容性。

**方案E: 修改msrun启动方式，阻止scheduler执行训练脚本**
查阅MindSpore文档，寻找 `--join` 或其他参数控制scheduler是否执行用户脚本。当前 MindSpore 2.8.0 的 `_MetaServerNode.run()` 代码中**无条件执行用户脚本**。

---

## 五、环境配置备忘

### 5.1 本地节点 (192.168.0.170)

```bash
export HCCL_IF_IP=192.168.0.170
export HCCL_SOCKET_IFNAME=enp67s0f5
export GLOO_SOCKET_IFNAME=enp67s0f5
export HCCL_IF_BASE_PORT=61000
export HCCL_CONNECT_TIMEOUT=1200
export ENABLE_OVERLAP=""  # 多机必须关闭
```

### 5.2 远程节点 (192.168.0.203)

```bash
export HCCL_IF_IP=192.168.0.203
export HCCL_SOCKET_IFNAME=enp67s0f5
export GLOO_SOCKET_IFNAME=enp67s0f5
export HCCL_IF_BASE_PORT=61000
export HCCL_CONNECT_TIMEOUT=1200
export ENABLE_OVERLAP=""
```

### 5.3 端口预留

```bash
RESERVED_PORTS="61000-61015"
sysctl -w net.ipv4.ip_local_reserved_ports="${RESERVED_PORTS}"
```

### 5.4 远程环境同步检查项

- [x] simplejson 已安装
- [x] `/zyl` NFS挂载正常
- [x] `enp67s0f5` 网卡存在
- [x] CANN环境正常
- [x] msadapter conda环境可用

---

## 六、代码修改清单

### task6.py 关键修改点

1. `_init_config()` 后设置 `HCCL_DETERMINISTIC` (带多机判断)
2. `run_msa_verify()`: `stdout=log_f` 避免PIPE缓冲
3. `run_remote_msa_verify()`: 远程SSH命令构造，含 `HCCL_IF_IP=node["HOST"]`
4. `run_remote_pta_verify()`: 同理设置远程HCCL网络环境
5. `main()`: 每轮迭代开头调用 `kill_pretraingpt()` + `_kill_remote_processes()`

### MSA脚本关键参数

```bash
# msa_cogvideox_real.sh
DISTRIBUTED_ARGS="
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK \
    --worker_num $WORLD_SIZE \
    --local_worker_num $NPUS_PER_NODE \
    --log_dir=msrun_log \
    --join=False \
    --cluster_time_out=300 \
    --bind_core=True
"
```

---

## 七、下一步行动计划

1. **修复MSA跨机超时** (最高优先级)
   - 尝试方案A (--join=True) + 方案B (清空日志标记)
   - 如不成功，尝试方案C (RANK_TABLE_FILE)
   - 如仍不成功，尝试方案D (mpirun)

2. **验证cogvideox 10轮跨机**
   - 单轮通过后，运行 `/tmp/run_task6_10iter.py`

3. **验证internvl3 跨机**
   - 需确认internvl3的PTA/MSA脚本是否已适配多机参数

4. **验证opensora和qwenvl PTA跨机**
   - 只需验证PTA执行，无需MSA

5. **检查MSA bug与detected_bugs一致性**
   - 对比 `/data2/lm-sv/lmsv_rec/detected_bugs` 目录

6. **更新config.json**
   - 确认 `TOTAL_ITER=10` 已设置

---

## 八、关键日志路径

| 日志 | 路径 |
|-----|------|
| PTA运行日志 | `/data2/lm-sv/output/YYYYMMDD_HHMMSS/iters/iter_N/runtime_logs/pta_verify_iterN.log` |
| MSA运行日志 | `/data2/lm-sv/output/YYYYMMDD_HHMMSS/iters/iter_N/runtime_logs/msa_verify_iterN.log` |
| msrun worker日志 | `/shared/mindspeed-mm/MindSpeed-MM/msrun_log/worker_*.log` |
| msrun scheduler日志 | `/shared/mindspeed-mm/MindSpeed-MM/msrun_log/scheduler.log` |
| 任务汇总报告 | `/data2/lm-sv/output/YYYYMMDD_HHMMSS/analysis/summary.md` |

---

## 九、注意事项

1. **不要同时运行多个Task6进程**，会导致路径映射混乱和资源竞争
2. **每次启动前必须kill残留进程** (已实现到代码中)
3. **多机模式下必须关闭 ENABLE_OVERLAP**，否则HCCL死锁
4. **不要设置 ASCEND_LAUNCH_BLOCKING=1**，会导致多机分布式hang
5. **远程节点的环境变更必须同步** (conda包、sysctl配置等)
6. **MSA的msrun_log目录需每次清空**，避免旧日志干扰


## 十轮验证实时记录

> 更新时间: 2026-04-25 19:17

| 轮次 | PTA loss | MSA loss | diff | 状态 |
|------|----------|----------|------|------|
| iter_1 | 0.70337 | 0.69265 | 1.52% | ✅ 通过 |
| iter_2 | 0.70331 | 0.69264 | 1.52% | ✅ 通过 |
| iter_3 | 0.70331 | 0.69264 | 1.52% | ✅ 通过 |
| iter_4 | - | - | - | 🔄 进行中 |
| iter_5-10 | - | - | - | ⏳ 待执行 |
