#!/usr/bin/env python3
import json
import signal
from pathlib import Path


TASK_DESC = {
    0: "子节点模式（仅配置环境与 slave 监听）",
    1: "整网泛化变异测试",
    2: "模块内组件泛化测试",
    3: "模块间泛化组合变异测试",
    4: "【多模态模型】模块间泛化组合变异测试",
    5: "【多模态模型】模块内组件泛化测试",
    6: "【多模态模型】整网泛化变异测试"
}


def _handle_sigint(_signum, _frame) -> None:
    print("\n[genconf] 已取消。", flush=True)
    raise SystemExit(130)


signal.signal(signal.SIGINT, _handle_sigint)


def _default_int(value, fallback):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return fallback


def _default_bool(value, fallback):
    return value if isinstance(value, bool) else fallback


def _default_str(value, fallback):
    return value if isinstance(value, str) and value else fallback


def _default_str_list(value, fallback):
    if isinstance(value, list) and value and all(isinstance(x, str) and x for x in value):
        return value
    return fallback


def _default_int_list(value, fallback):
    if isinstance(value, list) and value and all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        return value
    return fallback


def load_existing_config(path=Path("config.json")):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def ask_with_default(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip()
    return default if value == "" else value


def ask_int(prompt, default, min_value=None, max_value=None):
    while True:
        raw = ask_with_default(prompt, str(default))
        try:
            value = int(raw)
        except ValueError:
            print("请输入整数。")
            continue
        if min_value is not None and value < min_value:
            print(f"请输入大于等于 {min_value} 的整数。")
            continue
        if max_value is not None and value > max_value:
            print(f"请输入小于等于 {max_value} 的整数。")
            continue
        return value


def ask_bool(prompt, default):
    default_text = "y" if default else "n"
    while True:
        raw = ask_with_default(f"{prompt}（y/n）", default_text).lower()
        if raw in ("是", "y", "Y", "yes", "true", "1"):
            return True
        if raw in ("否", "n", "N", "no", "false", "0"):
            return False
        print("请输入 是/否（或 y/n, true/false, 1/0）。")


def ask_model_name(prompt, default):
    while True:
        value = ask_with_default(prompt, default).strip()
        if value:
            return value
        print("模型名不能为空。")


def ask_required_str(prompt, default=""):
    while True:
        value = ask_with_default(prompt, default).strip()
        if value:
            return value
        print("该项不能为空。")


def ask_model_list(prompt, default):
    default_text = ",".join(default)
    while True:
        raw = ask_with_default(f"{prompt}（逗号分隔）", default_text)
        values = [x.strip() for x in raw.split(",") if x.strip()]
        if values:
            return values
        print("请至少输入一个模型名。")


def ask_int_list(prompt, default, min_value=None, max_value=None):
    default_text = ",".join(str(x) for x in default)
    while True:
        raw = ask_with_default(f"{prompt}（逗号分隔）", default_text)
        try:
            values = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            print("列表中只能包含整数。")
            continue
        if not values:
            print("请至少输入一个整数。")
            continue
        if min_value is not None and any(v < min_value for v in values):
            print(f"列表中的整数必须大于等于 {min_value}。")
            continue
        if max_value is not None and any(v > max_value for v in values):
            print(f"列表中的整数必须小于等于 {max_value}。")
            continue
        return values


def ask_choice(prompt, options, default):
    options_text = "/".join(options)
    default_value = default if default in options else options[0]
    while True:
        value = ask_with_default(f"{prompt}（{options_text}）", default_value).strip()
        if value in options:
            return value
        print(f"请输入以下选项之一：{options_text}")


def _default_node_list(value, fallback):
    if isinstance(value, list):
        normalized = [x for x in value if isinstance(x, dict)]
        if normalized:
            return normalized
    return fallback


def ask_task45_multinode(task_defaults):
    defaults = task_defaults.get("MULTI_NODE") if isinstance(task_defaults.get("MULTI_NODE"), dict) else {}
    enabled_default = _default_bool(defaults.get("ENABLED"), False)
    enabled = ask_bool("是否使用多机启动", enabled_default)
    if not enabled:
        return {"MULTI_NODE": {"ENABLED": False}}

    node_count = ask_int("NNODES（节点总数）", _default_int(defaults.get("NNODES"), 2), min_value=2)
    master_addr = ask_required_str("MASTER_ADDR（主节点地址）", _default_str(defaults.get("MASTER_ADDR"), "127.0.0.1"))
    default_nodes = _default_node_list(defaults.get("OTHER_NODES"), [])

    other_nodes = []
    for index in range(1, node_count):
        print(f"\n请输入第 {index + 1} 个节点信息：")
        node_default = default_nodes[index - 1] if index - 1 < len(default_nodes) else {}
        host = ask_required_str("  地址（HOST）", _default_str(node_default.get("HOST"), ""))
        ssh_port = ask_int("  SSH_PORT（SSH端口）", _default_int(node_default.get("SSH_PORT"), 22), min_value=1)
        lmsv_path = ask_required_str("  LMSV_PATH（远端lmsv路径）", _default_str(node_default.get("LMSV_PATH"), ""))
        pta_name = ask_required_str("  PTA_NAME（PTA conda环境名）", _default_str(node_default.get("PTA_NAME"), ""))
        msa_name = ask_required_str("  MSA_NAME（MSA conda环境名）", _default_str(node_default.get("MSA_NAME"), ""))
        pta_path = ask_required_str("  PTA_PATH（远端PTA代码路径）", _default_str(node_default.get("PTA_PATH"), ""))
        msa_path = ask_required_str("  MSA_PATH（远端MSA代码路径）", _default_str(node_default.get("MSA_PATH"), ""))
        has_container_default = _default_bool(
            node_default.get("HAS_CONTAINER"),
            bool(_default_str(node_default.get("CONTAINER_NAME"), "")),
        )
        has_container = ask_bool("  是否在容器内运行（y/n）", has_container_default)

        node_info = {
            "HOST": host,
            "SSH_PORT": ssh_port,
            "LMSV_PATH": lmsv_path,
            "PTA_NAME": pta_name,
            "MSA_NAME": msa_name,
            "PTA_PATH": pta_path,
            "MSA_PATH": msa_path,
            "HAS_CONTAINER": has_container,
        }
        if has_container:
            node_info["CONTAINER_NAME"] = ask_required_str(
                "  CONTAINER_NAME（容器名）",
                _default_str(node_default.get("CONTAINER_NAME"), ""),
            )
        other_nodes.append(node_info)

    return {
        "MULTI_NODE": {
            "ENABLED": True,
            "MASTER_ADDR": master_addr,
            "NNODES": node_count,
            "OTHER_NODES": other_nodes,
        }
    }


def build_cluster_config(global_defaults=None):
    global_defaults = global_defaults if isinstance(global_defaults, dict) else {}
    defaults = global_defaults.get("CLUSTER") if isinstance(global_defaults.get("CLUSTER"), dict) else {}
    enabled = ask_bool(
        "是否启用 Task1/2/3 主从多机模式",
        _default_bool(defaults.get("ENABLED"), False),
    )
    if not enabled:
        return {"CLUSTER": {"ENABLED": False}}

    raw_slaves = defaults.get("SLAVES")
    slave_defaults = []
    if isinstance(raw_slaves, list):
        for item in raw_slaves:
            if isinstance(item, str) and item.strip():
                slave_defaults.append(item.strip())
            elif isinstance(item, dict):
                endpoint = _default_str(item.get("ENDPOINT"), "")
                if endpoint:
                    slave_defaults.append(endpoint)
    slave_text = ask_with_default(
        "SLAVES（其它节点地址，逗号分隔，格式 IP:port；从机机器可留空）",
        ",".join(slave_defaults or ["192.168.0.203:19001"]),
    )
    slaves = [item.strip() for item in slave_text.split(",") if item.strip()]
    return {
        "CLUSTER": {
            "ENABLED": True,
            "MASTER_ADDR": ask_required_str(
                "CLUSTER.MASTER_ADDR（由核心节点下发给所有节点的训练地址）",
                _default_str(defaults.get("MASTER_ADDR"), "192.168.0.170"),
            ),
            "MASTER_PORT": ask_int(
                "CLUSTER.MASTER_PORT（训练 master_port）",
                _default_int(defaults.get("MASTER_PORT"), 8118),
                min_value=1,
            ),
            "LOCAL_NPUS_PER_NODE": ask_int(
                "CLUSTER.LOCAL_NPUS_PER_NODE（当前机器本地 worker 数，0 表示自动探测）",
                _default_int(defaults.get("LOCAL_NPUS_PER_NODE"), 0),
                min_value=0,
            ),
            "NODE_RANK": 0,
            "LISTEN_HOST": _default_str(defaults.get("LISTEN_HOST"), "0.0.0.0"),
            "LISTEN_PORT": _default_int(defaults.get("LISTEN_PORT"), 19001),
            "REQUEST_TIMEOUT": _default_int(defaults.get("REQUEST_TIMEOUT"), 30),
            "SESSION_TIMEOUT": _default_int(defaults.get("SESSION_TIMEOUT"), 7200),
            "SLAVES": slaves,
        }
    }


def build_slave_cluster_config(global_defaults=None):
    global_defaults = global_defaults if isinstance(global_defaults, dict) else {}
    defaults = global_defaults.get("CLUSTER") if isinstance(global_defaults.get("CLUSTER"), dict) else {}
    return {
        "CLUSTER": {
            "ENABLED": True,
            "MASTER_ADDR": _default_str(defaults.get("MASTER_ADDR"), "192.168.0.170"),
            "MASTER_PORT": _default_int(defaults.get("MASTER_PORT"), 8118),
            "NODE_RANK": 0,
            "LISTEN_HOST": ask_with_default(
                "CLUSTER.LISTEN_HOST（slave 监听地址）",
                _default_str(defaults.get("LISTEN_HOST"), "0.0.0.0"),
            ),
            "LISTEN_PORT": ask_int(
                "CLUSTER.LISTEN_PORT（slave 监听端口）",
                _default_int(defaults.get("LISTEN_PORT"), 19001),
                min_value=1,
            ),
            "REQUEST_TIMEOUT": _default_int(defaults.get("REQUEST_TIMEOUT"), 30),
            "SESSION_TIMEOUT": _default_int(defaults.get("SESSION_TIMEOUT"), 7200),
            "LOCAL_NPUS_PER_NODE": _default_int(defaults.get("LOCAL_NPUS_PER_NODE"), 0),
            "SLAVES": [],
        }
    }


def ask_task2_model_submodule_pairs(task_defaults):
    default_models = _default_str_list(task_defaults.get("MODELS"), ["qwen2", "qwen2", "qwen2"])
    default_submodules = _default_int_list(task_defaults.get("SUBMODULES"), [3, 4, 5])

    while True:
        models = ask_model_list("MODELS", default_models)
        submodules = ask_int_list(
            "SUBMODULES（取值范围 0~10）",
            default_submodules,
            min_value=0,
            max_value=10,
        )
        if len(models) == len(submodules):
            return models, submodules
        print("MODELS 和 SUBMODULES 数量必须一一对应，请重新输入。")


def ask_task_type(default=1):
    default = _default_int(default, 1)
    if default not in (0, 1, 2, 3, 4, 5, 6):
        default = 1
    print("请选择任务类型：")
    print("0. 旧版子节点模式（仅兼容 CLUSTER/slave 监听配置）")
    print("1. 整网泛化变异测试")
    print("2. 模块内组件泛化测试")
    print("3. 模块间泛化组合变异测试")
    print("4. 【多模态模型】模块间泛化组合变异测试")
    print("5. 【多模态模型】模块内组件泛化测试")
    print("6. 【多模态模型】整网泛化变异测试")
    while True:
        raw = ask_with_default("输入任务类型编号", str(default))
        if raw in ("0", "1", "2", "3", "4", "5", "6"):
            return int(raw)
        print("请输入 0、1、2、3、4、5 或 6。")


def build_task_config(task_type, task_defaults=None):
    task_defaults = task_defaults if isinstance(task_defaults, dict) else {}
    if task_type == 0:
        return {}
    if task_type == 1:
        return {
            "MODEL_NAME": ask_model_name("MODEL_NAME", _default_str(task_defaults.get("MODEL_NAME"), "qwen2")),
            "TOTAL_ITER": ask_int("TOTAL_ITER（总迭代数）", _default_int(task_defaults.get("TOTAL_ITER"), 10), min_value=1),
            "COMPARE_MODE": ask_choice(
                "COMPARE_MODE（Task1对比模式）",
                ["pta_msa", "pta_mf"],
                _default_str(task_defaults.get("COMPARE_MODE"), "pta_msa"),
            ),
            "ENABLE_MF_WEIGHT_LOAD": ask_bool(
                "ENABLE_MF_WEIGHT_LOAD（Task1是否加载MF权重）",
                _default_bool(task_defaults.get("ENABLE_MF_WEIGHT_LOAD"), True),
            ),
            "BASE_SEED": ask_int("BASE_SEED（基础随机种子）", _default_int(task_defaults.get("BASE_SEED"), 43), min_value=0),
            "MUTNM": ask_int("MUTNM（每轮变异参数数量）", _default_int(task_defaults.get("MUTNM"), 2), min_value=1),
            "LOAD_STEPS": ask_int("LOAD_STEPS（LOAD模式训练轮数）", _default_int(task_defaults.get("LOAD_STEPS"), 30), min_value=1),
        }
    if task_type == 2:
        models, submodules = ask_task2_model_submodule_pairs(task_defaults)
        return {
            "MODELS": models,
            "TOTAL_ITER": ask_int("TOTAL_ITER（总迭代数）", _default_int(task_defaults.get("TOTAL_ITER"), 100), min_value=1),
            "BASE_SEED": ask_int("BASE_SEED（基础随机种子）", _default_int(task_defaults.get("BASE_SEED"), 43), min_value=0),
            "SUBMODULES": submodules,
            "MUTNM": ask_int("MUTNM（每轮变异参数数量）", _default_int(task_defaults.get("MUTNM"), 2), min_value=1),
            "LOAD_STEPS": ask_int("LOAD_STEPS（LOAD模式训练步数）", _default_int(task_defaults.get("LOAD_STEPS"), 15), min_value=1),
            "COMPARE_MODE": ask_choice(
                "COMPARE_MODE（Task2对比模式）",
                ["pta_msa", "pta_mf"],
                _default_str(task_defaults.get("COMPARE_MODE"), "pta_msa"),
            ),
            "MF_ARGS_PATH": ask_with_default(
                "MF_ARGS_PATH（MF参数模板路径）",
                _default_str(task_defaults.get("MF_ARGS_PATH"), "assets/runtime/mf_templates/basic.yaml"),
            ),
            "ENABLE_MF_WEIGHT_LOAD": ask_bool(
                "ENABLE_MF_WEIGHT_LOAD（Task2是否加载MF权重）",
                _default_bool(task_defaults.get("ENABLE_MF_WEIGHT_LOAD"), False),
            ),
        }
    if task_type == 3:
        return {
            "MODELS": ask_model_list("MODELS", _default_str_list(task_defaults.get("MODELS"), ["qwen2", "glm4"])),
            "TOTAL_ITER": ask_int("TOTAL_ITER（变异轮次）", _default_int(task_defaults.get("TOTAL_ITER"), 100), min_value=1),
            "BASE_SEED": ask_int("BASE_SEED（基础随机种子）", _default_int(task_defaults.get("BASE_SEED"), 43), min_value=0),
            "MUTNM": ask_int("MUTNM（每轮变异参数数量）", _default_int(task_defaults.get("MUTNM"), 2), min_value=1),
            "LOAD_STEPS": ask_int("LOAD_STEPS（LOAD模式训练步数）", _default_int(task_defaults.get("LOAD_STEPS"), 15), min_value=1),
            "COMPARE_MODE": ask_choice(
                "COMPARE_MODE（Task3对比模式）",
                ["pta_msa", "pta_mf"],
                _default_str(task_defaults.get("COMPARE_MODE"), "pta_msa"),
            ),
        }
    if task_type == 4:
        config = {
            "TOTAL_ITER": ask_int("TOTAL_ITER（变异轮次）", _default_int(task_defaults.get("TOTAL_ITER"), 5), min_value=1),
            "RUN_STEPS": ask_int("RUN_STEPS（RUN模式训练轮数）", _default_int(task_defaults.get("RUN_STEPS"), 20), min_value=1),
            "COMPARE_MODE": ask_choice(
                "COMPARE_MODE（Task4对比模式）",
                ["pta_msa"],
                _default_str(task_defaults.get("COMPARE_MODE"), "pta_msa"),
            ),
        }
        config.update(ask_task45_multinode(task_defaults))
        return config
    if task_type == 5:
        config = {
            "TOTAL_ITER": ask_int("TOTAL_ITER（变异轮次）", _default_int(task_defaults.get("TOTAL_ITER"), 5), min_value=1),
            "RUN_STEPS": ask_int("RUN_STEPS（RUN模式训练轮数）", _default_int(task_defaults.get("RUN_STEPS"), 20), min_value=1),
            "MUTATE_STEPS": ask_int("MUTATE_STEPS（变异步数）", _default_int(task_defaults.get("MUTATE_STEPS"), 10), min_value=1),
            "COMPARE_MODE": ask_choice(
                "COMPARE_MODE（Task5对比模式）",
                ["pta_msa"],
                _default_str(task_defaults.get("COMPARE_MODE"), "pta_msa"),
            ),
            "MODULE_TYPE": ask_choice(
                "MODULE_TYPE（模块类型）",
                ["all", "text_decoder", "image_encoder"],
                _default_str(task_defaults.get("MODULE_TYPE"), "all"),
            ),
        }
        config.update(ask_task45_multinode(task_defaults))
        return config
    if task_type == 6:
        return {
            "MODEL_NAME": ask_choice(
                "MODEL_NAME（多模态模型名称）",
                ["internvl3", "qwenvl", "opensora", "cogvideox"],
                _default_str(task_defaults.get("MODEL_NAME"), "internvl3"),
            ),
            "TOTAL_ITER": ask_int("TOTAL_ITER（总迭代数）", _default_int(task_defaults.get("TOTAL_ITER"), 10), min_value=1),
            "MUTNM": ask_int("MUTNM（每轮变异参数数量）", _default_int(task_defaults.get("MUTNM"), 2), min_value=1),
            "COMPARE_MODE": ask_choice(
                "COMPARE_MODE（Task6对比模式）",
                ["pta_msa"],
                _default_str(task_defaults.get("COMPARE_MODE"), "pta_msa"),
            ),
            "TRAIN_ITER": ask_int("TRAIN_ITER（每轮训练/推理步数）", _default_int(task_defaults.get("TRAIN_ITER", task_defaults.get("SAVE_STEPS", task_defaults.get("TRAIN_ITERS"))), 5), min_value=1),
            "BASE_SEED": ask_int("BASE_SEED（基础随机种子）", _default_int(task_defaults.get("BASE_SEED"), 43), min_value=0),
        }


def build_task_advanced_config(task_type, task_defaults=None):
    task_defaults = task_defaults if isinstance(task_defaults, dict) else {}
    if task_type == 0:
        return {}
    save_steps_fallback = 5 if task_type == 6 else 1
    hidden = {
        "PTA_MAX_RUNTIME": _default_int(task_defaults.get("PTA_MAX_RUNTIME"), 3000),
        "MSA_MAX_RUNTIME": _default_int(task_defaults.get("MSA_MAX_RUNTIME", task_defaults.get("MAX_VALIDATE_TIME")), 3000),
    }
    if task_type == 6:
        hidden["TRAIN_ITER"] = _default_int(task_defaults.get("TRAIN_ITER", task_defaults.get("SAVE_STEPS", task_defaults.get("TRAIN_ITERS"))), save_steps_fallback)
    else:
        hidden["SAVE_STEPS"] = _default_int(task_defaults.get("SAVE_STEPS", task_defaults.get("TRAIN_ITERS")), save_steps_fallback)
    if task_type != 6:
        hidden["LOG_INIT_WAIT"] = _default_int(task_defaults.get("LOG_INIT_WAIT"), 240)
        hidden["LOG_STABLE_THRESHOLD"] = _default_int(task_defaults.get("LOG_STABLE_THRESHOLD"), 150)
    if task_type == 3:
        hidden["MAX_MUTATION_WAIT"] = _default_int(task_defaults.get("MAX_MUTATION_WAIT"), 600)
    if not ask_bool("是否进行高级配置", False):
        return hidden

    config = {
        "PTA_MAX_RUNTIME": ask_int("PTA_MAX_RUNTIME（PTA最大运行时间，秒）", hidden["PTA_MAX_RUNTIME"], min_value=1),
        "MSA_MAX_RUNTIME": ask_int("MSA_MAX_TIME（MSA最大运行时间，秒）", hidden["MSA_MAX_RUNTIME"], min_value=1),
    }
    if task_type == 6:
        config["TRAIN_ITER"] = ask_int("TRAIN_ITER（每轮训练/推理步数）", hidden["TRAIN_ITER"], min_value=1)
    else:
        config["SAVE_STEPS"] = ask_int("SAVE_STEPS（SAVE模式训练步数）", hidden["SAVE_STEPS"], min_value=1)
    if task_type != 6:
        config["LOG_INIT_WAIT"] = ask_int("LOG_INIT_WAIT（MSA日志初始化等待秒数）", hidden["LOG_INIT_WAIT"], min_value=1)
        config["LOG_STABLE_THRESHOLD"] = ask_int("LOG_STABLE_THRESHOLD（MSA日志稳定阈值秒数）", hidden["LOG_STABLE_THRESHOLD"], min_value=1)
    if task_type == 3:
        config["MAX_MUTATION_WAIT"] = ask_int("MAX_MUTATION_WAIT（变异产物等待秒数）", hidden["MAX_MUTATION_WAIT"], min_value=1)
    return config


def build_global_config(task_type, task_config, global_defaults=None):
    global_defaults = global_defaults if isinstance(global_defaults, dict) else {}
    # Task6使用专用的多模态环境名称作为默认值
    pta_default = "mindspeed" if task_type == 6 else _default_str(global_defaults.get("PTA_NAME"), "mindspeed")
    config = {
        "PTA_NAME": ask_with_default(
            "PTA_NAME（PTA conda环境名称）",
            pta_default,
        ),
    }

    # Task6 统一使用 MINDSPEED_MM_PATH，其他任务仍用 PTA_PATH / MSA_PATH
    if task_type == 6:
        mm_default = _default_str(
            global_defaults.get("MINDSPEED_MM_PATH"),
            _default_str(global_defaults.get("PTA_PATH"), "../mm-new")
        )
        config["MINDSPEED_MM_PATH"] = ask_with_default(
            "MINDSPEED_MM_PATH（MindSpeed-MM 工作区根目录，如 ../mm-new 或 /zyl/mindspeed-mm，自动推导 MindSpeed-MM 子目录）",
            mm_default,
        )
    else:
        config["PTA_PATH"] = ask_with_default(
            "PTA_PATH（PTA代码路径）",
            _default_str(global_defaults.get("PTA_PATH"), ""),
        )

    # 所有任务均按模式互斥：先选模式，再按模式补齐对应环境。
    compare_mode = str((task_config or {}).get("COMPARE_MODE", "")).strip().lower()
    if task_type == 0:
        config["MSA_NAME"] = ask_with_default(
            "MSA_NAME（MSA conda环境名称）",
            _default_str(global_defaults.get("MSA_NAME"), "msadapter"),
        )
        config["MSA_PATH"] = ask_with_default(
            "MSA_PATH（MSA代码路径）",
            _default_str(global_defaults.get("MSA_PATH"), ""),
        )
        config["MF_NAME"] = ask_with_default(
            "MF_NAME（MF conda环境名称）",
            _default_str(global_defaults.get("MF_NAME"), "mindf_py311"),
        )
        config["SAVE_ABNORMAL_WEIGHTS"] = _default_bool(global_defaults.get("SAVE_ABNORMAL_WEIGHTS"), True)
        config.update(build_slave_cluster_config(global_defaults))
        return config

    if task_type in (1, 2, 3) and compare_mode == "pta_mf":
        config["MF_NAME"] = ask_with_default(
            "MF_NAME（MF conda环境名称）",
            _default_str(global_defaults.get("MF_NAME"), "mindf_py311"),
        )
        config["SAVE_ABNORMAL_WEIGHTS"] = ask_bool(
            "SAVE_ABNORMAL_WEIGHTS（是否保存异常迭代权重）",
            _default_bool(global_defaults.get("SAVE_ABNORMAL_WEIGHTS"), True),
        )
        if isinstance(global_defaults.get("CLUSTER"), dict):
            config["CLUSTER"] = global_defaults.get("CLUSTER")
        return config

    msa_default = "msadapter" if task_type == 6 else _default_str(global_defaults.get("MSA_NAME"), "msadapter")
    config["MSA_NAME"] = ask_with_default(
        "MSA_NAME（MSA conda环境名称）",
        msa_default,
    )
    if task_type != 6:
        config["MSA_PATH"] = ask_with_default(
            "MSA_PATH（MSA代码路径）",
            _default_str(global_defaults.get("MSA_PATH"), ""),
        )
    config["SAVE_ABNORMAL_WEIGHTS"] = ask_bool(
        "SAVE_ABNORMAL_WEIGHTS（是否保存异常迭代权重）",
        _default_bool(global_defaults.get("SAVE_ABNORMAL_WEIGHTS"), True),
    )
    if isinstance(global_defaults.get("CLUSTER"), dict):
        config["CLUSTER"] = global_defaults.get("CLUSTER")
    return config


def main():
    existing_config = load_existing_config()
    task_type = ask_task_type(existing_config.get("task_type"))
    task_defaults = {}
    tasks = existing_config.get("tasks")
    if task_type != 0 and isinstance(tasks, dict):
        raw_task_defaults = tasks.get(str(task_type))
        if isinstance(raw_task_defaults, dict):
            task_defaults = raw_task_defaults

    task_config = build_task_config(task_type, task_defaults)
    if task_type in (1, 2, 3):
        task_config.update(ask_task45_multinode(task_defaults))
    task_config.update(build_task_advanced_config(task_type, task_defaults))

    config = {
        "task_type": task_type,
        **build_global_config(task_type, task_config, existing_config),
        "tasks": ({str(task_type): task_config} if task_type != 0 else {}),
    }
    output_path = Path("config.json")
    output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已生成配置文件：{output_path.resolve()}")
    print(f"任务类型：{task_type}（{TASK_DESC[task_type]}）")


if __name__ == "__main__":
    main()

# TODO美化交互
