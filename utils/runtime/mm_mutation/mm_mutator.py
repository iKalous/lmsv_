#!/usr/bin/env python3
"""
多模态配置变异器
基于 MindSpeed-MM net_mutation 迁移适配
用于Task6多模态整网变异和验证
"""

import copy
import json
import math
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# 设置随机种子
random.seed(42)
np.random.seed(42)

_MUTATION_LOG_PATH = "tmp/task6/mutation.log"


def _mut_log(msg: str) -> None:
    """将变异详细日志写入文件，避免污染控制台"""
    os.makedirs(os.path.dirname(_MUTATION_LOG_PATH), exist_ok=True)
    with open(_MUTATION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")


class MMConfigMutator:
    """多模态配置变异器"""

    def __init__(self, output_dir: str = "./mutated_mm_configs"):
        """
        初始化多模态配置变异器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 增量变异状态管理
        self.incremental_configs = {}  # 存储每个节点的增量变异配置 {node_id: config}
        self.mutation_history = {}  # 存储每个节点的变异历史 {node_id: [mutation_records]}
        self.current_round = 0  # 当前变异轮次

        # 加载变异参数池配置（从YAML文件）
        self.mutable_params_pool = self._load_mutable_params_pool()

        _mut_log(f"✓ MM配置变异器初始化完成")
        _mut_log(f"  输出目录: {output_dir}")
        _mut_log(f"  增量变异支持: 启用")
        _mut_log(f"  变异参数池: {len(self.mutable_params_pool)} 个参数")

    def reset_incremental_state(self):
        """重置增量变异状态，开始新的变异序列"""
        self.incremental_configs.clear()
        self.mutation_history.clear()
        self.current_round = 0
        _mut_log("✓ 增量变异状态已重置")

    def _load_mutable_params_pool(self) -> Dict[str, Any]:
        """
        从YAML文件加载变异参数池配置

        Returns:
            Dict[str, Any]: 变异参数池字典
        """
        # 默认配置文件路径（相对于lmsv_rec根目录）
        default_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'configs', 'mutable_params_pool.yaml'
        )

        # 尝试多个可能的路径
        config_paths = [
            os.environ.get('MUTABLE_PARAMS_POOL_PATH'),  # 环境变量指定
            default_config_path,  # 默认路径（向后兼容）
            './mutable_params_pool.yaml',  # 当前目录根路径
            '../mutable_params_pool.yaml',  # 上级目录
            './configs/mutable_params_pool.yaml',  # 当前目录相对路径（向后兼容）
            '../configs/mutable_params_pool.yaml',  # 上级目录（向后兼容）
            '../../configs/mutable_params_pool.yaml',  # 上两级目录（向后兼容）
        ]

        for config_path in config_paths:
            if config_path and os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        pool_config = yaml.safe_load(f)

                    if not pool_config:
                        continue

                    # 将YAML配置转换为内部格式
                    mutable_params = {}

                    # 处理数值型参数
                    numeric_params = pool_config.get('numeric_params', {})
                    for param_name, config in numeric_params.items():
                        mutable_params[param_name] = {
                            'min_val': config.get('min_val'),
                            'max_val': config.get('max_val'),
                            'min_factor': config.get('min_factor', 0.5),
                            'max_factor': config.get('max_factor', 2.0)
                        }

                    # 处理枚举型参数
                    enum_params = pool_config.get('enum_params', {})
                    for param_name, enum_values in enum_params.items():
                        mutable_params[param_name] = {'enum': enum_values}

                    _mut_log(f"✓ 已加载变异参数池: {config_path}")
                    _mut_log(f"  - 数值型参数: {len(numeric_params)} 个")
                    _mut_log(f"  - 枚举型参数: {len(enum_params)} 个")
                    return mutable_params

                except Exception as e:
                    _mut_log(f"⚠️  加载配置文件失败 {config_path}: {e}")
                    continue

        # 如果都失败了，使用默认配置
        _mut_log("⚠️  无法加载YAML配置文件，使用内置默认配置")
        return self._get_default_mutable_params_pool()

    def _get_default_mutable_params_pool(self) -> Dict[str, Any]:
        """
        获取默认的变异参数池配置（内置后备配置）

        Returns:
            Dict[str, Any]: 默认变异参数池
        """
        return {
            'mlp_ratio': {'min_val': 2.0, 'max_val': 8.0, 'min_factor': 0.7, 'max_factor': 1.5},
            'model_max_length': {'min_val': 32, 'max_val': 512, 'min_factor': 0.5, 'max_factor': 2.0},
            'class_dropout_prob': {'min_val': 0.0, 'max_val': 0.3, 'min_factor': 0.5, 'max_factor': 2.0},
            'drop_path': {'min_val': 0.0, 'max_val': 0.3, 'min_factor': 0.5, 'max_factor': 2.0},
            'space_scale': {'min_val': 0.0, 'max_val': 0.3, 'min_factor': 0.5, 'max_factor': 2.0},
            'time_scale': {'min_val': 0.0, 'max_val': 0.3, 'min_factor': 0.5, 'max_factor': 2.0},
            'norm_eps': {'min_val': 1e-06, 'max_val': 1e-04, 'min_factor': 0.5, 'max_factor': 2.0},
            'num_embeds_ada_norm': {'min_val': 1000, 'max_val': 2000, 'min_factor': 0.5, 'max_factor': 2.0},
            'attention_dropout': {'min_val': 0.0, 'max_val': 1.0, 'min_factor': 0.5, 'max_factor': 2.0},
            'hidden_dropout': {'min_val': 0.0, 'max_val': 1.0, 'min_factor': 0.5, 'max_factor': 2.0},
            'vocab_size': {'min_val': 10000, 'max_val': 11000, 'min_factor': 0.5, 'max_factor': 2.0},
            'batch_size': {'min_val': 4, 'max_val': 8, 'min_factor': 0.5, 'max_factor': 2.0},
            'seq_len': {'min_val': 10, 'max_val': 20, 'min_factor': 0.5, 'max_factor': 2.0},
            'max_sequence_length': {'min_val': 128, 'max_val': 228, 'min_factor': 0.5, 'max_factor': 2.0},
            'num_tokentypes': {'min_val': 2, 'max_val': 4, 'min_factor': 0.5, 'max_factor': 2.0},
            'dropout_prob': {'min_val': 0.1, 'max_val': 0.1, 'min_factor': 0.5, 'max_factor': 2.0},
            'image_size': {'min_val': 224, 'max_val': 224, 'min_factor': 0.5, 'max_factor': 2.0},
            'patch_size': {'min_val': 16, 'max_val': 16, 'min_factor': 0.5, 'max_factor': 2.0},
            'num_idx': {'min_val': 200, 'max_val': 1000, 'min_factor': 0.5, 'max_factor': 2.0},
            'shift_scale': {'min_val': 1.0, 'max_val': 20.0, 'min_factor': 0.5, 'max_factor': 2.0},
            'class_token_len': {'min_val': 1, 'max_val': 2, 'min_factor': 0.5, 'max_factor': 2.0},
            'learnable_pos_embed': {'enum': [True, False]},
            'norm_elementwise_affine': {'enum': [True, False]},
            'uniform_sampling': {'enum': [True, False]},
            'quantize_c_noise': {'enum': [True, False]},
            'ucg_rate': {'min_val': 0.0, 'max_val': 2.0, 'min_factor': 0.5, 'max_factor': 2.0},
            'concat_text_embed': {'enum': [True, False]},
            'conv_padding': {'enum': [0, 1]},
            'num_res_blocks': {'enum': [2, 3, 4]},
            'use_tiling': {'enum': [True, False]},
            'use_rope': {'enum': [True, False]},
            'cross_attention_dim': {'min_val': 1024, 'max_val': 4096, 'min_factor': 0.5, 'max_factor': 2.0},
            'attention_bias': {'enum': [True, False]},
            'parallel_output': {'enum': [True, False]},
            'intermediate_size': {'min_val': 855, 'max_val': 3420, 'min_factor': 0.5, 'max_factor': 2.0},
            'activation_func': {'enum': ['gelu', 'silu']},
            'disable_bias_linear': {'enum': [True, False]},
            'bias_activation_fusion': {'enum': [True, False]},
            'attention_softmax_in_fp32': {'enum': [True]},
            'normalization': {'enum': ['LayerNorm', 'RMSNorm']},
            'max_position_embeddings': {'min_val': 1000, 'max_val': 128000, 'min_factor': 0.5, 'max_factor': 2.0},
            'add_qkv_bias': {'enum': [True, False]},
            'post_layer_norm': {'enum': [True, False]},
            'group_query_attention': {'enum': [True, False]},
            'untie_embeddings_and_output_weights': {'enum': [True, False]},
            'tokens_per_second': {'min_val': 2, 'max_val': 64, 'min_factor': 0.5, 'max_factor': 2.0},
            'window_attn_size': {'min_val': 56, 'max_val': 448, 'min_factor': 0.5, 'max_factor': 2.0},
            'decoder_gather_norm': {'enum': [True, False]},
            'noised_image_all_concat': {'enum': [True, False]},
            'noised_image_input': {'enum': [True, False]},
            'noised_image_dropout': {'min_val': 0.0, 'max_val': 0.2, 'min_factor': 0.5, 'max_factor': 2.0},
            'low_cpu_mem_usage': {'enum': [True, False]},
            'use_attention_mask': {'enum': [True, False]},
            'init_method_std': {'enum': [0.01]},
            'use_fused_rotary_pos_emb': {'enum': [True, False]},
            'recompute_num_layers': {'min_val': 2, 'max_val': 64, 'min_factor': 0.5, 'max_factor': 2.0},
            'recompute_method': {'enum': ['block', 'uniform']},
            'persist_layer_norm': {'enum': [True, False]},
            'apply_query_key_layer_scaling': {'enum': [True, False]},
            'deallocate_pipeline_outputs': {'enum': [True, False]},
            'layernorm_zero_centered_gamma': {'enum': [True, False]},
            'bias_dropout_fusion': {'enum': [True, False]},
            'apply_rope_fusion': {'enum': [True, False]},
        }

    def set_base_config_for_node(self, node_id: int, base_config: Dict[str, Any]):
        """
        为指定节点设置基础配置（用于初始轮次）

        Args:
            node_id: 节点ID
            base_config: 基础配置
        """
        self.incremental_configs[node_id] = base_config.copy()
        if node_id not in self.mutation_history:
            self.mutation_history[node_id] = []
        _mut_log(f"✓ 为节点 {node_id} 设置基础配置")

    def get_incremental_config(self, node_id: int) -> Optional[Dict[str, Any]]:
        """
        获取指定节点的增量变异配置

        Args:
            node_id: 节点ID

        Returns:
            增量的配置，如果不存在则返回None
        """
        return self.incremental_configs.get(node_id)

    def get_mutation_history(self, node_id: int) -> List[Dict[str, Any]]:
        """
        获取指定节点的变异历史

        Args:
            node_id: 节点ID

        Returns:
            变异历史列表
        """
        return self.mutation_history.get(node_id, [])

    def mutate_predict_config(self, base_config: Dict[str, Any] = None,
                              mutation_rate: float = 1,
                              model_type: str = 'stdit3',
                              model_num: int = 2,
                              node_id: int = None,
                              use_accumulated: bool = True,
                              parent_model_type: str = None) -> Dict[str, Any]:
        """
        变异predict_config配置

        Args:
            base_config: 基础配置，如果为None则使用默认配置
            mutation_rate: 变异率
            model_type: 模型类型（组件类型，如text_decoder, image_encoder等）
            model_num: 变异参数数量
            node_id: 节点ID，用于增量变异管理
            use_accumulated: 是否使用增量变异（基于上一轮结果）
            parent_model_type: 父模型类型（如internvl3, qwenvl等），用于区分不同模型的验证规则

        Returns:
            Dict[str, Any]: 变异后的配置
        """
        _mut_log(f"对{model_type}模型进行变异，变异率: {mutation_rate}, 变异参数数: {model_num}")

        # 使用从YAML加载的变异参数池
        mutable_params_pool = self.mutable_params_pool

        # 确定要使用的基础配置
        if use_accumulated and node_id is not None and node_id in self.incremental_configs:
            working_config = self.incremental_configs[node_id].copy()
            _mut_log(f"💡 节点 {node_id}: 基于第 {self.current_round} 轮增量结果进行变异")
        elif base_config is not None:
            if isinstance(base_config, str):
                working_config = {}
            else:
                working_config = base_config.copy()
            if node_id is not None:
                self.incremental_configs[node_id] = working_config.copy()
            _mut_log(f"节点 {node_id}: 使用提供的基础配置进行变异")
        else:
            return None

        # 记录变异前的状态
        pre_mutation_config = working_config.copy()
        mutations_applied = {}

        # 只保留存在于当前配置中且可变的参数
        available_params = {
            k: v for k, v in mutable_params_pool.items()
            if k in working_config and working_config[k] is not None
            and not isinstance(working_config[k], (list, tuple, np.ndarray))
        }
        num = min(model_num, len(available_params))
        if num == 0:
            _mut_log(f"⚠ 没有可变异的参数")
            return working_config
        sampled_keys = random.sample(list(available_params.keys()), k=num)
        mutable_params = {k: available_params[k] for k in sampled_keys}

        for param_name, constraints in mutable_params.items():
            if param_name in working_config and working_config[param_name] is not None and random.random() < mutation_rate:
                _mut_log(f"  变异参数: {param_name}")
                original_value = working_config[param_name]

                # 跳过列表/元组/数组类型
                if isinstance(original_value, (list, tuple, np.ndarray)):
                    continue

                # 优先处理枚举类型参数
                if isinstance(constraints, dict) and 'enum' in constraints:
                    enum_values = constraints['enum']
                    if isinstance(original_value, bool) and all(isinstance(v, bool) for v in enum_values):
                        new_value = not original_value
                    else:
                        candidates = [v for v in enum_values if v != original_value]
                        new_value = random.choice(candidates) if candidates else original_value

                    _mut_log(f"    {param_name}: {original_value} -> {new_value}")
                    working_config[param_name] = new_value
                    mutations_applied[param_name] = {
                        'from': original_value,
                        'to': new_value,
                        'constraints': constraints
                    }
                    continue

                # 处理浮点数参数
                float_params = ['mlp_ratio', 'class_dropout_prob', 'drop_path', "space_scale", "time_scale",
                               "num_embeds_ada_norm", "attention_dropout", "hidden_dropout", "dropout_prob", "ucg_rate"]
                if param_name in float_params:
                    min_val = max(constraints['min_val'], original_value * constraints['min_factor'])
                    max_val = min(constraints['max_val'], original_value * constraints['max_factor'])
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val
                    new_value = random.uniform(min_val, max_val)
                    new_value = round(new_value, 3)
                elif param_name == "norm_eps":
                    min_val = max(constraints['min_val'], original_value * constraints['min_factor'])
                    max_val = min(constraints['max_val'], original_value * constraints['max_factor'])
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val
                    new_value = random.uniform(min_val, max_val)
                    new_value = round(new_value, 8)
                else:  # 整数参数
                    min_val = max(constraints['min_val'], int(original_value * constraints['min_factor']))
                    max_val = min(constraints['max_val'], int(original_value * constraints['max_factor']))
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val

                    # 特殊处理2的幂
                    if param_name in ['num_heads', 'num_attention_heads']:
                        possible_values = [2 ** i for i in range(3, 7) if 2 ** i >= min_val and 2 ** i <= max_val]
                        if possible_values:
                            new_value = random.choice(possible_values)
                        else:
                            new_value = max(8, min(32, original_value))
                    else:
                        if param_name in ['num_layers', 'num_heads', 'num_attention_heads']:
                            min_val = max(min_val, 1)
                        new_value = random.randint(min_val, max_val)

                    # 在整数参数变异后，若涉及 hidden_size，保证能被 num_heads 与 num_attention_heads 的最小公倍数整除
                    if param_name == 'hidden_size':
                        nh = working_config.get('num_heads', None)
                        nah = working_config.get('num_attention_heads', None)
                        divisors = [h for h in [nh, nah] if isinstance(h, int) and h > 0]
                        if divisors:
                            div = divisors[0]
                            for h in divisors[1:]:
                                div = div * h // math.gcd(div, h)
                            if new_value % div != 0:
                                remainder = new_value % div
                                lower = new_value - remainder
                                upper = lower + div
                                valid_candidates = []
                                min_i = max(min_val, 1)
                                max_i = max_val
                                if lower >= min_i:
                                    valid_candidates.append(lower)
                                if upper <= max_i:
                                    valid_candidates.append(upper)
                                if not valid_candidates:
                                    ceil_multiple = ((min_i + div - 1) // div) * div
                                    floor_multiple = (max_i // div) * div
                                    if ceil_multiple <= max_i:
                                        valid_candidates.append(ceil_multiple)
                                    if floor_multiple >= min_i and floor_multiple != ceil_multiple:
                                        valid_candidates.append(floor_multiple)
                                    if not valid_candidates:
                                        valid_candidates.append(max(min_i, div))
                                adjusted = min(valid_candidates, key=lambda v: abs(v - new_value))
                                new_value = adjusted
                                _mut_log(f"    调整 hidden_size 以整除: {original_value} -> {new_value} (div={div}, heads={divisors})")

                _mut_log(f"    {param_name}: {original_value} -> {new_value}")
                working_config[param_name] = new_value
                mutations_applied[param_name] = {
                    'from': original_value,
                    'to': new_value,
                    'constraints': constraints
                }

        # 对嵌套配置进行变异（如vision_encoder等）
        for nested_key in ['vision_encoder', 'vision_projector', 'text_decoder']:
            if nested_key in working_config and isinstance(working_config[nested_key], dict):
                nested_config = working_config[nested_key]
                for param_name, constraints in mutable_params.items():
                    if param_name in nested_config and nested_config[param_name] is not None and random.random() < mutation_rate:
                        if isinstance(nested_config[param_name], (list, tuple, np.ndarray)):
                            continue

                        original_value = nested_config[param_name]
                        key_with_prefix = f"{nested_key}.{param_name}"
                        _mut_log(f"  变异参数: {key_with_prefix}")

                        if isinstance(constraints, dict) and 'enum' in constraints:
                            enum_values = constraints['enum']
                            if isinstance(original_value, bool) and all(isinstance(v, bool) for v in enum_values):
                                new_value = not original_value
                            else:
                                candidates = [v for v in enum_values if v != original_value]
                                new_value = random.choice(candidates) if candidates else original_value
                        elif param_name in ['mlp_ratio', 'class_dropout_prob', 'drop_path', "space_scale", "time_scale",
                                           "num_embeds_ada_norm", "attention_dropout", "hidden_dropout", "dropout_prob", "ucg_rate"]:
                            min_val = max(constraints['min_val'], original_value * constraints['min_factor'])
                            max_val = min(constraints['max_val'], original_value * constraints['max_factor'])
                            if min_val > max_val:
                                min_val, max_val = max_val, min_val
                            new_value = round(random.uniform(min_val, max_val), 3)
                        elif param_name == "norm_eps":
                            min_val = max(constraints['min_val'], original_value * constraints['min_factor'])
                            max_val = min(constraints['max_val'], original_value * constraints['max_factor'])
                            if min_val > max_val:
                                min_val, max_val = max_val, min_val
                            new_value = round(random.uniform(min_val, max_val), 8)
                        else:
                            min_val = max(constraints['min_val'], int(original_value * constraints['min_factor']))
                            max_val = min(constraints['max_val'], int(original_value * constraints['max_factor']))
                            if min_val > max_val:
                                min_val, max_val = max_val, min_val
                            new_value = random.randint(min_val, max_val)

                        _mut_log(f"    {param_name}: {original_value} -> {new_value}")
                        nested_config[param_name] = new_value

        # TP兼容性验证与修复
        mutations_applied = self._fix_invalid_mutations(
            working_config, mutations_applied, available_params,
            model_num, mutation_rate, parent_model_type
        )

        # 更新增量配置和历史记录
        if node_id is not None:
            self.incremental_configs[node_id] = working_config.copy()

            mutation_record = {
                'round': self.current_round + 1,
                'timestamp': datetime.now().isoformat(),
                'pre_mutation': pre_mutation_config,
                'post_mutation': working_config,
                'mutations_applied': mutations_applied,
                'mutation_rate': mutation_rate,
                'model_type': model_type,
                'use_accumulated': use_accumulated
            }

            if node_id not in self.mutation_history:
                self.mutation_history[node_id] = []
            self.mutation_history[node_id].append(mutation_record)

            _mut_log(f"✓ 节点 {node_id} 第 {self.current_round + 1} 轮变异完成，应用了 {len(mutations_applied)} 个参数变异")

        # 特殊处理 pipeline_num_layers - 确保长度与 pipeline-model-parallel-size 匹配 (PP=4)
        pp_size = 4
        # 处理顶层配置中的 pipeline_num_layers
        for pp_key in ['pipeline_num_layers']:
            if pp_key in working_config:
                pp_value = working_config[pp_key]
                if isinstance(pp_value, (list, tuple)) and len(pp_value) != pp_size:
                    total_layers = sum(pp_value)
                    base = total_layers // pp_size
                    remainder = total_layers % pp_size
                    new_pp = [base + 1] * remainder + [base] * (pp_size - remainder)
                    working_config[pp_key] = new_pp
                    _mut_log(f"    调整 {pp_key} 长度: {pp_value} ({len(pp_value)}项) -> {new_pp} ({pp_size}项)")

        # 处理嵌套配置中的 pipeline_num_layers (如 vision_encoder, text_decoder)
        for nested_key in ['vision_encoder', 'text_decoder', 'image_encoder']:
            if nested_key in working_config and isinstance(working_config[nested_key], dict):
                nested = working_config[nested_key]
                if 'pipeline_num_layers' in nested:
                    pp_value = nested['pipeline_num_layers']
                    if isinstance(pp_value, (list, tuple)) and len(pp_value) != pp_size:
                        total_layers = sum(pp_value)
                        base = total_layers // pp_size
                        remainder = total_layers % pp_size
                        new_pp = [base + 1] * remainder + [base] * (pp_size - remainder)
                        nested['pipeline_num_layers'] = new_pp
                        _mut_log(f"    调整 {nested_key}.pipeline_num_layers 长度: {pp_value} ({len(pp_value)}项) -> {new_pp} ({pp_size}项)")

        return working_config

    def _compute_cogvideox_patch_tokens(self, working_config: Dict[str, Any]) -> Optional[int]:
        """
        计算CogVideoX的patch token数量。
        patch tokens = input_size各维度 / patch_size各维度的乘积
        """
        input_size = working_config.get('input_size')
        patch_size = working_config.get('patch_size')
        if not input_size or not patch_size or len(input_size) != len(patch_size):
            return None
        try:
            total = 1
            for i in range(len(input_size)):
                if patch_size[i] == 0:
                    return None
                total *= input_size[i] // patch_size[i]
            return total
        except (TypeError, ZeroDivisionError):
            return None

    def _mutate_single_param_value(self, param_name: str, original_value: Any,
                                    constraints: Dict[str, Any], working_config: Dict[str, Any]) -> Any:
        """
        对单个参数执行突变，返回新值。
        复用mutate_predict_config中的突变逻辑。
        """
        if isinstance(constraints, dict) and 'enum' in constraints:
            enum_values = constraints['enum']
            if isinstance(original_value, bool) and all(isinstance(v, bool) for v in enum_values):
                return not original_value
            else:
                candidates = [v for v in enum_values if v != original_value]
                return random.choice(candidates) if candidates else original_value

        # 浮点数参数
        float_params = ['mlp_ratio', 'class_dropout_prob', 'drop_path', "space_scale", "time_scale",
                       "num_embeds_ada_norm", "attention_dropout", "hidden_dropout", "dropout_prob", "ucg_rate"]
        if param_name in float_params:
            min_val = max(constraints['min_val'], original_value * constraints['min_factor'])
            max_val = min(constraints['max_val'], original_value * constraints['max_factor'])
            if min_val > max_val:
                min_val, max_val = max_val, min_val
            new_value = random.uniform(min_val, max_val)
            return round(new_value, 3)
        elif param_name == "norm_eps":
            min_val = max(constraints['min_val'], original_value * constraints['min_factor'])
            max_val = min(constraints['max_val'], original_value * constraints['max_factor'])
            if min_val > max_val:
                min_val, max_val = max_val, min_val
            new_value = random.uniform(min_val, max_val)
            return round(new_value, 8)
        else:  # 整数参数
            min_val = max(constraints['min_val'], int(original_value * constraints['min_factor']))
            max_val = min(constraints['max_val'], int(original_value * constraints['max_factor']))
            if min_val > max_val:
                min_val, max_val = max_val, min_val

            if param_name in ['num_heads', 'num_attention_heads']:
                possible_values = [2 ** i for i in range(3, 7) if 2 ** i >= min_val and 2 ** i <= max_val]
                if possible_values:
                    return random.choice(possible_values)
                else:
                    return max(8, min(32, original_value))
            else:
                if param_name in ['num_layers', 'num_heads', 'num_attention_heads']:
                    min_val = max(min_val, 1)
                new_value = random.randint(min_val, max_val)

                if param_name == 'hidden_size':
                    nh = working_config.get('num_heads', None)
                    nah = working_config.get('num_attention_heads', None)
                    divisors = [h for h in [nh, nah] if isinstance(h, int) and h > 0]
                    if divisors:
                        div = divisors[0]
                        for h in divisors[1:]:
                            div = div * h // math.gcd(div, h)
                        if new_value % div != 0:
                            remainder = new_value % div
                            lower = new_value - remainder
                            upper = lower + div
                            valid_candidates = []
                            min_i = max(min_val, 1)
                            max_i = max_val
                            if lower >= min_i:
                                valid_candidates.append(lower)
                            if upper <= max_i:
                                valid_candidates.append(upper)
                            if not valid_candidates:
                                ceil_multiple = ((min_i + div - 1) // div) * div
                                floor_multiple = (max_i // div) * div
                                if ceil_multiple <= max_i:
                                    valid_candidates.append(ceil_multiple)
                                if floor_multiple >= min_i and floor_multiple != ceil_multiple:
                                    valid_candidates.append(floor_multiple)
                                if not valid_candidates:
                                    valid_candidates.append(max(min_i, div))
                            adjusted = min(valid_candidates, key=lambda v: abs(v - new_value))
                            new_value = adjusted
                return new_value

        return original_value

    def _fix_invalid_mutations(self, working_config: Dict[str, Any], mutations_applied: Dict[str, Any],
                               available_params: Dict[str, Any], model_num: int,
                               mutation_rate: float, parent_model_type: Optional[str]) -> Dict[str, Any]:
        """
        验证并修复不合法的突变，确保TP兼容性。
        对于CogVideoX：concat_text_embed=false时，patch tokens数量必须能被TP size整除。
        如果不合法，回退该突变并从剩余参数中补选；若最终无突变，强制突变一个参数。
        """
        if parent_model_type != 'cogvideox':
            return mutations_applied

        # 检查concat_text_embed导致的TP不兼容
        if working_config.get('concat_text_embed') is False and 'concat_text_embed' in mutations_applied:
            patch_tokens = self._compute_cogvideox_patch_tokens(working_config)
            tp_size = 8  # 当前环境TP size（8卡）
            if patch_tokens is not None and patch_tokens % tp_size != 0:
                # 回退concat_text_embed
                original_value = mutations_applied['concat_text_embed']['from']
                working_config['concat_text_embed'] = original_value
                _mut_log(f"    [TP兼容性修正] concat_text_embed=false 导致patch tokens({patch_tokens})"
                        f"不能被TP size({tp_size})整除，回退到 {original_value}")
                del mutations_applied['concat_text_embed']

                # 从剩余可用参数中补选一个进行额外突变（排除concat_text_embed防止再次选中）
                remaining = [k for k in available_params if k not in mutations_applied and k != 'concat_text_embed']
                if remaining and len(mutations_applied) < model_num:
                    extra_param = random.choice(remaining)
                    if extra_param in working_config and working_config[extra_param] is not None:
                        orig_val = working_config[extra_param]
                        constraints = available_params[extra_param]
                        new_val = self._mutate_single_param_value(extra_param, orig_val, constraints, working_config)
                        if new_val != orig_val:
                            working_config[extra_param] = new_val
                            mutations_applied[extra_param] = {
                                'from': orig_val,
                                'to': new_val,
                                'constraints': constraints
                            }
                            _mut_log(f"    [补偿突变] {extra_param}: {orig_val} -> {new_val}")

        # 最终检查：确保至少有一个参数被突变（保障多样性）
        if len(mutations_applied) == 0 and available_params:
            # 排除已知会导致问题的参数
            exclude_params = set()
            if parent_model_type == 'cogvideox':
                # concat_text_embed=false可能导致TP不兼容，强制突变时避免它
                if working_config.get('concat_text_embed') is True:
                    exclude_params.add('concat_text_embed')

            force_candidates = [k for k in available_params if k not in exclude_params]
            if not force_candidates:
                force_candidates = list(available_params.keys())

            force_param = random.choice(force_candidates)
            if force_param in working_config and working_config[force_param] is not None:
                orig_val = working_config[force_param]
                constraints = available_params[force_param]
                new_val = self._mutate_single_param_value(force_param, orig_val, constraints, working_config)
                if new_val != orig_val:
                    working_config[force_param] = new_val
                    mutations_applied[force_param] = {
                        'from': orig_val,
                        'to': new_val,
                        'constraints': constraints
                    }
                    _mut_log(f"    [强制突变] {force_param}: {orig_val} -> {new_val}")

        return mutations_applied

    def advance_round(self):
        """推进到下一轮变异"""
        self.current_round += 1
        _mut_log(f"🔄 变异轮次推进到第 {self.current_round + 1} 轮")

    def get_incremental_changes_summary(self, node_id: int) -> Dict[str, Any]:
        """
        获取指定节点的增量变异变化汇总
        """
        if node_id not in self.mutation_history:
            return {
                'node_id': node_id,
                'total_rounds': 0,
                'current_round': self.current_round,
                'incremental_changes': {},
                'total_parameters_mutated': 0
            }

        history = self.mutation_history[node_id]
        parameter_changes = {}
        total_mutated_params = 0

        for round_record in history:
            mutations = round_record.get('mutations_applied', {})
            for param_name, mutation_info in mutations.items():
                if param_name not in parameter_changes:
                    parameter_changes[param_name] = {
                        'initial': mutation_info['from'],
                        'current': mutation_info['to'],
                        'total_rounds_mutated': 1
                    }
                    total_mutated_params += 1
                else:
                    parameter_changes[param_name]['current'] = mutation_info['to']
                    parameter_changes[param_name]['total_rounds_mutated'] += 1

        return {
            'node_id': node_id,
            'total_rounds': len(history),
            'current_round': self.current_round,
            'incremental_changes': parameter_changes,
            'total_parameters_mutated': total_mutated_params
        }
