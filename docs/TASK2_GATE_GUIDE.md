# Task1/2/3 门禁测试开发与使用说明

## 1. 目标与背景

为降低合代码前引入回归风险，LMSV 增加了一个可复用的门禁入口：

- 命令入口：`./lmsv test`
- 当前作用：批量执行 Task1/Task2/Task3 预合入门禁用例（逐条跑）
- 设计目标：
  - 一条命令触发有限回归
  - 用例可开关、可增删
  - 配置风格尽量贴近原始 `config.json`
  - 执行后自动恢复原配置，减少人工失误

## 2. 当前实现范围

当前 `test` 子命令调用统一门禁执行器：

- 脚本：`tests/gate/task2_gate.py`
- 默认用例：
  - Task1：`tests/gate/task1_cases.json`
  - Task2：`tests/gate/task2_cases.json`
  - Task3：`tests/gate/task3_cases.json`
- 调度：每条用例都会临时写入 `config.json` 后执行 `python do.py`（可通过 conda 环境名切换为 `conda run -n <env> python do.py`）
- 结果：失败即停止（fail-fast），最后输出汇总

## 3. 门禁相关文件

- 入口脚本：`lmsv`
- 执行器：`tests/gate/task2_gate.py`
- 用例清单：`tests/gate/task1_cases.json`、`tests/gate/task2_cases.json`、`tests/gate/task3_cases.json`

## 4. 运行机制（开发视角）

`tests/gate/task2_gate.py` 的核心流程：

1. 读取并校验用例文件
2. 根据 `--task` 选择默认用例文件（或使用 `--cases-file` 自定义）
3. 选择待执行用例（支持 `enabled`、`--cases`、`--limit`）
4. 决定运行环境（优先级：`--env-name` > `case.env_name` > `common.env_name` > 当前环境）
5. 备份当前 `config.json`
6. 对每个用例执行：
   - `base_config_patch` 与 `case.config_patch` 深度合并
   - 临时覆盖 `config.json`
   - 运行 `python do.py`
   - 记录 PASS/FAIL 与耗时
7. 执行完成或中途失败后，恢复原 `config.json`
8. 打印 summary 并返回退出码（全通过返回 0）

## 5. 用例文件格式（贴近 config.json）

用例文件采用“公共配置 + 公共补丁 + 单条补丁”结构：

```json
{
  "common": {
    "env_name": "mindformers"
  },
  "base_config_patch": {
    "task_type": 2,
    "tasks": {
      "2": {
        "COMPARE_MODE": "pta_msa",
        "TOTAL_ITER": 1,
        "MUTNM": 2
      }
    }
  },
  "cases": [
    {
      "id": "qwen3-main",
      "enabled": true,
      "config_patch": {
        "tasks": {
          "2": {
            "MODELS": ["qwen3", "qwen3", "qwen3"],
            "SUBMODULES": [2, 3, 9]
          }
        }
      }
    }
  ]
}
```

字段说明：

- `common.env_name`：门禁默认 conda 环境名（可被 case/env-name 覆盖）
- `base_config_patch`：全部用例共享的默认配置（建议放 task_type、Task2 公共参数）
- `cases[*].id`：用例唯一标识，用于 `--cases` 精确选择
- `cases[*].enabled`：开关位，`false` 表示默认不执行
- `cases[*].config_patch`：该用例的覆盖配置（通常只覆盖 `MODELS`、`SUBMODULES`）

当前默认：

- `COMPARE_MODE=pta_msa`
- `ENABLE_MF_WEIGHT_LOAD=false`

## 6. 常用命令

在 `lmsv_rec` 目录下执行：

```bash
./lmsv test --task 2 --list
./lmsv test --task 1 --dry-run
./lmsv test --task 3 --dry-run
./lmsv test --task 2 --cases qwen3-main,deepseekv3-main
./lmsv test --task 2 --env-name mindformers
```

参数说明：

- `--task {1|2|3}`：选择任务类型（默认 `2`）
- `--list`：只列出全部用例，不执行
- `--dry-run`：仅展示本次选中的用例，不执行
- `--limit N`：只执行选中集合前 N 条
- `--cases a,b,c`：按 id 精确挑选要执行的用例
- `--cases-file <path>`：指定替代用例文件
- `--env-name <name>`：覆盖所有用例的 conda 环境名

## 7. 明天测试建议流程

1. 先看三类任务待跑集合

```bash
./lmsv test --task 1 --list
./lmsv test --task 2 --list
./lmsv test --task 3 --list
```

2. 先做轻量验证（不执行训练）

```bash
./lmsv test --task 2 --dry-run --limit 3
```

3. 空闲资源时先跑建议优先集（例如 Task2 的模型聚合子集）

```bash
./lmsv test --task 2 --cases qwen3-main,mixtral-main
```

4. 再跑全部 enabled 用例

```bash
./lmsv test --task 2
```

## 8. 失败排查建议

任一用例失败后会立即停止；优先查看最近一次 output 目录中的以下信息：

- `output/<timestamp>/log.txt`
- `output/<timestamp>/iters/iter_*/`
- `output/<timestamp>/iters/iter_*/res/`
- `output/<timestamp>/iters/iter_*/msrun_log/`

常见排查方向：

- 环境与路径：`PTA_PATH`、`MSA_PATH`、conda 环境名
- 资源：NPU 卡可见性、任务并发、运行超时
- 配置：`MODELS` 与 `SUBMODULES` 是否对应、`COMPARE_MODE` 是否符合预期

## 9. 维护建议

- 新增用例：复制现有 case，改 `id` 与 `config_patch`
- 临时关闭：把 `enabled` 设为 `false`
- 调整门禁强度：通过 `--limit` 或维护两份 `cases-file`
- 环境统一管理：优先在 `common.env_name` 维护默认环境
- 建议提交前固定跑一套“小而稳”的 smoke 用例，再跑全量

## 10. 备注

该门禁实现的关键安全点是“临时改写 + 自动恢复 `config.json`”。

即使某条用例失败，执行器也会在 `finally` 中恢复原配置，避免把测试态配置残留到后续手工任务中。
