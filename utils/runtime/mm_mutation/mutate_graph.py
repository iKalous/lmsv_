#!/usr/bin/env python3
"""
多模态配置变异执行入口
基于 MindSpeed-MM net_mutation 迁移适配
用于Task6多模态整网变异和验证
"""

import copy
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Any

import numpy as np

from utils.runtime.mm_mutation.mm_mutator import MMConfigMutator


_MUTATION_LOG_PATH = "tmp/task6/mutation.log"


def _mut_log(msg: str) -> None:
    """将变异详细日志写入文件，避免污染控制台"""
    os.makedirs(os.path.dirname(_MUTATION_LOG_PATH), exist_ok=True)
    with open(_MUTATION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")


def mutate_json_all(mutnm: int, file_path: str, output_dir: str, model_name: str = "internvl3", seed: int = 42) -> str:
    """
    执行配置突变

    Args:
        mutnm: 每次突变修改的参数个数
        file_path: 原始配置文件路径
        output_dir: 突变输出目录
        model_name: 模型名称，用于适配不同模型的变异策略

    Returns:
        str: 生成的变异配置文件路径
    """
    random.seed(seed)
    np.random.seed(seed)

    mutator = MMConfigMutator(output_dir=output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 查找已有的最新变异文件
    max_idx = 0
    latest_path = None
    if os.path.exists(output_dir):
        for name in os.listdir(output_dir):
            m = re.match(r"mutation_gen(\d+)\.json$", name)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
                    latest_path = os.path.join(output_dir, name)

    # 加载基础配置
    if latest_path:
        _mut_log(f"基于上一轮变异结果: {latest_path}")
        with open(latest_path, 'r', encoding='utf-8') as f:
            base_config = json.load(f)
    else:
        _mut_log(f"基于基础配置文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            base_config = json.load(f)

    # 深拷贝配置
    current_configs = {model_key: copy.deepcopy(cfg)
                       for model_key, cfg in base_config.items()}

    mutation_rate = 1  # 变异率设为1，确保选中的参数一定变异
    gen = max_idx + 1  # 本轮代数

    # 对每个模型配置进行变异
    for model_key, cfg in current_configs.items():
        _mut_log(f"+++++++++++++++ 处理模型: {model_key} +++++++++++++++")
        # 跳过非字典类型的配置（如整数、浮点数、字符串、列表）
        if isinstance(cfg, (int, float, str, list)):
            _mut_log(f"跳过非字典配置: {model_key} = {cfg}")
            continue

        try:
            mutated_cfg = mutator.mutate_predict_config(
                base_config=cfg,
                mutation_rate=mutation_rate,
                model_type=model_key,
                model_num=mutnm,
                parent_model_type=model_name,
            )
            if mutated_cfg:
                current_configs[model_key] = mutated_cfg
                _mut_log(f"✓ {model_key} 变异完成")
            else:
                _mut_log(f"⚠ {model_key} 变异返回空配置，保持原配置")
        except Exception as e:
            _mut_log(f"✗ {model_key} 变异失败: {e}")
            import traceback
            _mut_log(traceback.format_exc())

    # 重组配置，保持原有顺序
    new_config = {}
    for model_key in base_config.keys():
        if model_key in current_configs:
            new_config[model_key] = current_configs[model_key]
        else:
            new_config[model_key] = base_config[model_key]

    # 检查配置是否真的发生了变化（多样性保障）
    config_changed = False
    for model_key in base_config.keys():
        if model_key in current_configs and current_configs[model_key] != base_config[model_key]:
            config_changed = True
            break
    if not config_changed:
        _mut_log("⚠ 警告: 新配置与基础配置完全相同，尝试强制突变一个参数")
        # 强制修改第一个可突变的字典配置中的一个参数
        for model_key, cfg in current_configs.items():
            if isinstance(cfg, dict) and cfg:
                first_key = list(cfg.keys())[0]
                if isinstance(cfg[first_key], bool):
                    cfg[first_key] = not cfg[first_key]
                elif isinstance(cfg[first_key], (int, float)) and not isinstance(cfg[first_key], bool):
                    cfg[first_key] = cfg[first_key] * 1.1 if cfg[first_key] != 0 else 1.0
                _mut_log(f"    强制突变 {model_key}.{first_key}: {base_config[model_key][first_key]} -> {cfg[first_key]}")
                config_changed = True
                break

    # 保存变异后的配置
    output_path = os.path.join(output_dir, f"mutation_gen{gen}.json")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(new_config, out_f, ensure_ascii=False, indent=2)

    _mut_log(f"✓ Generation {gen} saved to {output_path}")
    return output_path


def rollback_mutation(output_dir: str, gen: int = None) -> bool:
    """
    撤销指定轮次的变异（删除对应的变异文件）

    Args:
        output_dir: 变异输出目录
        gen: 要撤销的轮次，如果为None则撤销最新的一轮

    Returns:
        bool: 是否成功撤销
    """
    if gen is None:
        # 找到最新的一轮
        max_idx = 0
        for name in os.listdir(output_dir):
            m = re.match(r"mutation_gen(\d+)\.json$", name)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
        gen = max_idx

    mutation_file = os.path.join(output_dir, f"mutation_gen{gen}.json")

    if os.path.exists(mutation_file):
        try:
            os.remove(mutation_file)
            _mut_log(f"✓ 已撤销第 {gen} 轮变异，删除文件: {mutation_file}")
            return True
        except Exception as e:
            _mut_log(f"✗ 撤销失败: {e}")
            return False
    else:
        _mut_log(f"⚠ 文件不存在: {mutation_file}")
        return False


def get_latest_mutation(output_dir: str) -> str:
    """
    获取最新的变异配置文件路径

    Args:
        output_dir: 变异输出目录

    Returns:
        str: 最新的变异配置文件路径，如果没有则返回空字符串
    """
    max_idx = 0
    latest_path = ""

    if not os.path.exists(output_dir):
        return ""

    for name in os.listdir(output_dir):
        m = re.match(r"mutation_gen(\d+)\.json$", name)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
                latest_path = os.path.join(output_dir, name)

    return latest_path


def get_mutation_history(output_dir: str) -> list:
    """
    获取变异历史列表

    Args:
        output_dir: 变异输出目录

    Returns:
        list: 按顺序排列的变异文件路径列表
    """
    mutations = []

    if not os.path.exists(output_dir):
        return mutations

    for name in sorted(os.listdir(output_dir)):
        m = re.match(r"mutation_gen(\d+)\.json$", name)
        if m:
            mutations.append({
                'gen': int(m.group(1)),
                'path': os.path.join(output_dir, name)
            })

    return sorted(mutations, key=lambda x: x['gen'])


if __name__ == "__main__":
    # 测试用例
    import sys

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "internvl3"

    # 从环境变量获取PTA_PATH，构建模型配置映射
    pta_path = os.environ.get("PTA_PATH") or os.environ.get("PTAPATH")
    if not pta_path:
        # 尝试从lmsv_rec根目录推断（用于独立运行测试）
        script_dir = Path(__file__).resolve().parents[3]
        pta_path = str(script_dir)
        print(f"警告: PTA_PATH未设置，使用lmsv_rec根目录: {pta_path}")

    # 模型配置映射 - 使用环境变量或相对路径
    MODEL_CONFIGS = {
        "internvl3": {
            "base_config": f"{pta_path}/assets/mm_configs/model_8B.json",
            "output_dir": f"{pta_path}/tmp/task6/mutation_results/internvl3",
        },
        "qwenvl": {
            "base_config": f"{pta_path}/assets/mm_configs/inference_qwen2_5_vl_7b.json",
            "output_dir": f"{pta_path}/tmp/task6/mutation_results/qwenvl",
        },
        "opensora": {
            "base_config": f"{pta_path}/assets/mm_configs/inference_model_102x720x1280.json",
            "output_dir": f"{pta_path}/tmp/task6/mutation_results/opensora",
        },
        "cogvideox": {
            "base_config": f"{pta_path}/assets/mm_configs/model_cogvideox_i2v_1.5.json",
            "output_dir": f"{pta_path}/tmp/task6/mutation_results/cogvideox",
        },
    }

    if model_name not in MODEL_CONFIGS:
        print(f"不支持的模型: {model_name}")
        print(f"支持的模型: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    config = MODEL_CONFIGS[model_name]
    result = mutate_json_all(
        mutnm=2,
        file_path=config["base_config"],
        output_dir=config["output_dir"],
        model_name=model_name
    )
    print(f"\n生成的配置文件: {result}")
