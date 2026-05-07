# NFS IO 问题分析与规避方案

## 背景

当前容器中的项目目录位于 NFS 挂载上：

```text
/mine/hec/lm-sv/lmsv_rec -> 192.168.0.170:/data2/hec/lm-sv/lmsv_rec, nfs4
```

运行任务时，训练脚本、变异流程、日志采集、checkpoint、中间配置和 rank 日志会产生大量小文件和频繁元数据操作。如果这些热路径直接写到 NFS，就容易出现 IO wait 高、日志卡顿、进程长时间等待、清理目录慢等问题。

## 当前容器文件系统情况

检查结果如下：

```text
/tmp                               overlay  可写，本地 Docker 写层，约 28G 可用
/mine/hec/lm-sv/lmsv_rec            nfs4     可写，约 4.6T 可用
/data                              ext4     只读，和 Docker 宿主盘同源，约 28G 可用
/shared                            ext4     只读，和 Docker 宿主盘同源，约 28G 可用
/dev/shm                           tmpfs    可写，内存盘，显示容量大但不适合长期保存
```

结论：

- 当前容器里没有比 `/tmp` 更大的可写本地磁盘。
- `/dev/shm` 是内存盘，可以用于短时间 IO 对照实验，不适合作为大规模 checkpoint 或长期 output 存储。
- NFS 容量充足，但更适合作为归档盘，不适合作为训练过程中的热写入路径。

## NFS 与本地文件系统的区别

NFS 是 Network File System，即网络文件系统。访问 NFS 路径时，程序看起来是在读写本地目录，但实际文件在远端服务器上。每次 `open`、`stat`、`mkdir`、`rename`、`unlink`、`write`、`fsync` 都可能涉及网络请求和远端 NFS server 处理。

本地文件系统，例如 ext4、xfs、Docker overlay：

- 文件在本机磁盘或容器本地写层上。
- 小文件创建、日志追加、目录遍历、临时文件清理延迟低。
- 更适合训练和变异流程里的临时文件、日志、checkpoint 中间态。

NFS：

- 文件在远端服务器上。
- 单个大文件顺序读写通常可以接受。
- 大量小文件、频繁 `stat`、频繁 `mkdir/rm/rename`、多进程并发日志写入会明显变慢。
- 网络抖动、NFS server 繁忙、锁和缓存一致性开销都会放大等待时间。

tmpfs，例如 `/dev/shm`：

- 数据在内存中，速度很快。
- 容器退出或系统回收后数据不可依赖。
- 写太多会占用内存，可能导致 OOM。
- 适合验证是否为磁盘 IO 问题，不适合长期保存结果。

## 产生问题的主要原因

### 1. 大量小文件放大网络延迟

训练框架和变异流程会创建很多临时文件、配置快照、rank 日志、marker 文件、checkpoint 分片。每个文件操作在本地可能很快，但在 NFS 上会变成网络往返和远端 server 操作。

### 2. 元数据操作比数据写更容易成为瓶颈

`stat`、`exists`、`glob`、`find`、`mkdir`、`rename`、`unlink` 这类操作在 NFS 上开销很明显。很多 Python 逻辑会频繁检查日志是否存在、目录是否生成、文件是否稳定，这会持续打 NFS 元数据服务。

### 3. 日志追加与 flush 频繁

多进程训练时，各 rank 会持续写日志。如果日志文件位于 NFS，很多小写入和 flush 会排队到远端服务器，表现为日志延迟、训练进程等待或整体吞吐下降。

### 4. checkpoint 并发写入压力大

checkpoint 通常包含大文件和多个分片。多个 rank 并发写入、临时文件落盘、最终 `rename` 原子替换，在 NFS 上会放大 IO 和元数据压力。

### 5. NFS hard mount 会让故障表现为卡住

当前挂载包含类似参数：

```text
hard,timeo=600,retrans=2,local_lock=none
```

`hard` 挂载能保护数据一致性，但当 NFS server 或网络短时间不可用时，客户端会持续等待，而不是快速失败。因此症状常常是命令卡住、IO wait 高、进程没有明显报错。

## 推荐策略

采用“热路径本地写，冷结果批量归档到 NFS”的方式：

1. 任务运行期间，把 `output`、`tmp`、checkpoint、日志、中间配置写到本地 `/tmp/lmsv_rec`。
2. 任务结束后，使用 `rsync` 将完整 run 目录批量转存到 NFS。
3. 确认转存成功后，删除本地 run 目录，释放 `/tmp` 空间。

当前代码已经支持 NFS 环境下默认使用本地 scratch：

```bash
./lmsv do
```

默认本地路径：

```text
/tmp/lmsv_rec/output
/tmp/lmsv_rec/tmp
```

也可以显式指定：

```bash
export LMSV_LOCAL_SCRATCH_ROOT=/tmp/lmsv_rec
./lmsv do
```

如果宿主机后续挂载了更大的本地盘，例如 `/scratch`：

```bash
export LMSV_LOCAL_SCRATCH_ROOT=/scratch/lmsv_rec
./lmsv do
```

## 归档流程建议

任务结束后，将本地 output 转存到 NFS：

```bash
RUN_ID=<run目录名>
rsync -a --info=progress2 "/tmp/lmsv_rec/output/${RUN_ID}/" "/mine/hec/lm-sv/lmsv_rec/output/${RUN_ID}/"
touch "/mine/hec/lm-sv/lmsv_rec/output/${RUN_ID}/.ARCHIVED_OK"
rm -rf "/tmp/lmsv_rec/output/${RUN_ID}"
```

删除本地目录前应满足：

- `rsync` 返回码为 `0`。
- NFS 目标目录存在且文件结构完整。
- 已写入 `.ARCHIVED_OK` 标记。

更稳妥的做法是后续增加一个命令，例如：

```bash
./lmsv archive-latest
```

该命令可以自动完成：

- 查找 `/tmp/lmsv_rec/output` 下最新 run。
- `rsync` 到仓库 `output/`。
- 写入 `.ARCHIVED_OK`。
- 成功后删除本地 run。

## 排查命令

查看当前路径所在文件系统：

```bash
findmnt -T . -o TARGET,SOURCE,FSTYPE,OPTIONS
```

查看容量和文件系统类型：

```bash
df -hT . /tmp /dev/shm /data /shared
```

查看某个路径是否可写：

```bash
test -w /tmp && echo writable
test -w /data && echo writable
```

查看本地 scratch 占用：

```bash
du -sh /tmp/lmsv_rec
```

## 结论

当前问题不是单纯“磁盘容量不够”，而是 NFS 不适合承载训练过程中的高频小 IO 热路径。NFS 适合作为大容量结果归档位置；任务执行期间应尽量使用本地 scratch，任务结束后再批量转存到 NFS。
