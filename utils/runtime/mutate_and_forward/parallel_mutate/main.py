#!/usr/bin/env python3
"""
Megatron-LM 配置一键变异+校验修复主入口
顺序执行：
1. ParallelParameterMutator：随机生成满足 2 的幂次且乘积合法的并行参数；
2. EnhancedMegatronConfigValidator：检查并自动修正与并行策略冲突的其他维度。
"""

import argparse
from pathlib import Path
from .InfoParser import InfoParser                      # 第一段脚本，将输入非标准yaml转为标准yaml
from .ParallelParameterMutator import ParallelParameterMutator          # 第二段脚本，将并行参数变异
from .config_validator_moe import EnhancedMegatronConfigValidator  # 第三段脚本，校验并修复配置
from .deepseek_profile import apply_deepseekv3_unified_low_memory_profile
from .yamlToBash import YamlToBashConverter, save_bash_script      # 第四段脚本，将yaml转为bash脚本
import yaml
from .BashToYaml import bashtoyaml


def _normalize_key(key):
    return str(key).replace('-', '_').strip()


_SECTION_KEY_ALIASES = {
    "expert_num": "num_experts",
    "num_moe_experts": "num_experts",
    "per_token_num_experts_chosen": "moe_router_topk",
    "num_experts_chosen": "moe_router_topk",
    "moe_router_group_topk": "topk_group",
}


_SH_ARG_ALIASES = {
    "layernorm_epsilon": ("--norm-epsilon", False),
    "expert_num": ("--num-experts", False),
    "num_moe_experts": ("--num-experts", False),
    "num_experts": ("--num-experts", False),
    "per_token_num_experts_chosen": ("--moe-router-topk", False),
    "num_experts_chosen": ("--moe-router-topk", False),
    "moe_router_topk": ("--moe-router-topk", False),
    "moe_router_group_topk": ("--moe-router-group-topk", False),
    "topk_group": ("--moe-router-group-topk", False),
    "routed_scaling_factor": ("--moe-router-topk-scaling-factor", False),
    "add_bias_linear": ("--disable-bias-linear", True),
}


def _canonicalize_section_key(key):
    normalized_key = _normalize_key(key)
    return _SECTION_KEY_ALIASES.get(normalized_key, normalized_key)


def _canonicalize_sh_arg(key, value):
    normalized_key = _normalize_key(key)
    alias = _SH_ARG_ALIASES.get(normalized_key)
    if alias is None:
        return f"--{normalized_key.replace('_', '-')}", value

    arg_name, invert_bool = alias
    if invert_bool:
        if isinstance(value, bool):
            return arg_name, (not value)
        return arg_name, value
    return arg_name, value


def _merge_mutation_sections(config_data_res, mutation_record):
    """Merge all mutation sections into the parsed template config.

    The original implementation only applied TransformerConfig, which left
    extra_config/spec values stuck at template defaults. That made the final
    training script drift away from the effective mutated model config.
    """
    after = (mutation_record or {}).get('after', {}) or {}

    model_section = config_data_res.setdefault('model', {})
    training_section = config_data_res.setdefault('training', {})
    parallel_section = config_data_res.setdefault('parallel', {})
    moe_section = config_data_res.setdefault('moe', {})
    mla_section = config_data_res.setdefault('mla', {})

    transformer_cfg = after.get('TransformerConfig') or after.get('MLATransformerConfig') or {}
    model_key_overrides = {
        'layernorm_epsilon': 'norm_epsilon',
        'add_bias_linear': 'disable_bias_linear',
    }
    moe_keys = {
        'num_moe_experts': 'num_experts',
        'expert_num': 'num_experts',
        'moe_router_topk': 'moe_router_topk',
        'per_token_num_experts_chosen': 'moe_router_topk',
        'moe_router_load_balancing_type': 'moe_router_load_balancing_type',
        'moe_aux_loss_coeff': 'moe_aux_loss_coeff',
        'moe_router_pre_softmax': 'moe_router_pre_softmax',
    }
    for key, value in transformer_cfg.items():
        normalized_key = _canonicalize_section_key(key)
        if normalized_key in moe_keys:
            moe_section[moe_keys[normalized_key]] = value
            continue
        target_key = model_key_overrides.get(normalized_key, normalized_key)
        if normalized_key == 'add_bias_linear':
            try:
                model_section[target_key] = not value
            except Exception:
                model_section[target_key] = value
            continue
        if target_key == 'normalization' and 'normalization' in model_section:
            continue
        model_section[target_key] = value

    extra_cfg = after.get('extra_config', {}) or {}
    model_keys = {
        'seq_length',
        'position_embedding_type',
        'max_position_embeddings',
        'rotary_base',
        'vocab_size',
        'use_fused_rmsnorm',
        'use_fused_rotary_pos_emb',
        'use_fused_swiglu',
        'use_flash_attn',
    }
    training_keys = {
        'micro_batch_size',
        'global_batch_size',
        'lr',
        'min_lr',
        'weight_decay',
        'clip_grad',
        'lr_warmup_iters',
        'lr_decay_style',
        'train_iters',
        'use_distributed_optimizer',
    }
    parallel_keys = {
        'tensor_model_parallel_size',
        'pipeline_model_parallel_size',
        'expert_model_parallel_size',
        'context_parallel_size',
        'sequence_parallel',
        'num_layers_per_virtual_pipeline_stage',
    }
    for key, value in extra_cfg.items():
        normalized_key = _canonicalize_section_key(key)
        if normalized_key in model_keys:
            model_section[normalized_key] = value
        elif normalized_key in training_keys:
            training_section[normalized_key] = value
        elif normalized_key in parallel_keys:
            parallel_section[normalized_key] = value
        else:
            model_section[normalized_key] = value

    spec_cfg = after.get('get_gpt_layer_local_spec', {}) or {}
    for key, value in spec_cfg.items():
        normalized_key = _canonicalize_section_key(key)
        if normalized_key in {'num_experts', 'moe_grouped_gemm', 'moe_use_legacy_grouped_gemm'}:
            moe_section[normalized_key] = value
        elif normalized_key in {'qk_layernorm', 'qk_l2_norm', 'multi_latent_attention', 'fp8'}:
            mla_section[normalized_key] = value
        elif normalized_key == 'normalization' and 'normalization' not in model_section:
            model_section[normalized_key] = value
        else:
            model_section[normalized_key] = value

    if 'num_query_groups' in model_section and 'group_query_attention' not in model_section:
        try:
            model_section['group_query_attention'] = int(model_section['num_query_groups']) > 0
        except Exception:
            model_section['group_query_attention'] = True

    return config_data_res

def extract_parameters_to_sh_args(config):
    """
    提取配置中的所有二级参数并转换为sh脚本参数形式
    
    Args:
        config: 配置字典
        
    Returns:
        dict: 参数名到值的映射，格式为 {'--param-name': value}
    """
    sh_args = {}
    
    def process_section(section_data, prefix=""):
        """递归处理配置的各个部分"""
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                param_name, value = _canonicalize_sh_arg(key, value)
                
                if isinstance(value, bool):
                    # 布尔值：True时添加参数，False时不添加
                    if value:
                        sh_args[param_name] = None
                elif isinstance(value, (int, float, str)):
                    # 数值和字符串：添加参数和值
                    sh_args[param_name] = value
                elif isinstance(value, dict):
                    # 嵌套字典：递归处理
                    process_section(value, prefix)
                elif isinstance(value, list):
                    # 列表：转换为逗号分隔的字符串
                    sh_args[param_name] = ','.join(map(str, value))
    
    # 处理整个配置
    process_section(config)
    
    return sh_args

def update_script_parameters(script_path, sh_args):
    """
    更新脚本中的参数值
    
    Args:
        script_path: 脚本文件路径
        sh_args: 参数字典，格式为 {'--param-name': value}
    
    Returns:
        str: 更新后的脚本内容
    """
    import re
    
    # 读取脚本内容
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更新参数值
    updated_content = content
    updated_params = []
    new_params = []  # 存储需要添加的新参数

    # 互斥参数清理：Megatron 只允许 warmup fraction / warmup iters 二选一
    # 若本轮配置显式指定了其中一个，先从模板里删除另一个，避免最终脚本同时存在两者。
    def _remove_conflicting_arg(text, arg_name):
        import re
        pattern = rf'^[ \t]*{re.escape(arg_name)}(?:\s+[^\n\\]+)?\s*\\?\s*$\n?'
        return re.sub(pattern, '', text, flags=re.MULTILINE)

    if '--lr-warmup-iters' in sh_args:
        updated_content = _remove_conflicting_arg(updated_content, '--lr-warmup-fraction')
        updated_params.append('--lr-warmup-fraction (已移除，避免与 --lr-warmup-iters 冲突)')
    elif '--lr-warmup-fraction' in sh_args:
        updated_content = _remove_conflicting_arg(updated_content, '--lr-warmup-iters')
        updated_params.append('--lr-warmup-iters (已移除，避免与 --lr-warmup-fraction 冲突)')
    
    # 检查原脚本中是否包含特定参数
    has_attention_dropout = '--attention-dropout' in content
    has_use_fused_rmsnorm = '--use-fused-rmsnorm' in content
    has_moe_grouped_gemm = '--moe-grouped-gemm' in content
    has_moe_loss_coef = '--moe-aux-loss-coeff' in content
    has_moe_balancing_type = '--moe-router-load-balancing-type' in content
    has_moe_router_topk = '--moe-router-topk' in content
    has_ep = '--expert-model-parallel-size' in content
    has_experts = '--num-experts' in content
    for param, value in sh_args.items():
        if param == '--attention-dropout' and has_attention_dropout:
            continue
            
        # 特殊处理：如果原脚本包含--use-fused-rmsnorm，则不修改--normalization参数
        if param == '--normalization' and has_use_fused_rmsnorm:
            updated_params.append(f"{param} {value} (跳过修改，因为存在--use-fused-rmsnorm)")
            continue
        
        if param == '--moe-router-load-balancing-type' and not has_moe_balancing_type:
            continue            
            
        # 特殊处理：如果原脚本不包含--moe-grouped-gemm，则不添加该参数
        if param == '--moe-grouped-gemm' and not has_moe_grouped_gemm:
            updated_params.append(f"{param} (跳过添加，因为原脚本中没有--moe-grouped-gemm)")
            continue
        
        if param == '--moe-aux-loss-coeff' and not has_moe_loss_coef:
            updated_params.append(f"{param} (跳过添加，因为原脚本中没有--moe-aux-loss-coeff)")
            continue

        if param == '--moe-router-topk' and not has_moe_router_topk:
            updated_params.append(f"{param} (跳过添加，因为原脚本中没有--moe-router-topk)")
            continue

        if param == '--expert-model-parallel-size' and not has_ep:
            updated_params.append(f"{param} (跳过添加，因为原脚本中没有--moe-router-topk)")
            continue
            
        if param == '--num-experts' and not has_experts:
            updated_params.append(f"{param} (跳过添加，因为原脚本中没有--moe-router-topk)")
            continue
        
        
        
        if value is None:
            # 布尔参数，确保存在
            pattern = rf'(\s*)({re.escape(param)})(\s*\\?)'
            if re.search(pattern, updated_content):
                # 参数已存在，不需要修改
                updated_params.append(f"{param} (已存在)")
            else:
                # 参数不存在，记录需要添加
                new_params.append(f"    {param}")
                updated_params.append(f"{param} (待添加)")
        else:
            # 带值的参数，更新或添加
            # 匹配参数名和其后的值（可能跨行）
            pattern = rf'(\s*)({re.escape(param)})\s+[^\s\\\n]+'
            replacement = rf'\1\2 {value}'
            
            if re.search(pattern, updated_content):
                # 参数存在，更新值
                updated_content = re.sub(pattern, replacement, updated_content)
                updated_params.append(f"{param} {value} (已更新)")
            else:
                # 参数不存在，记录需要添加
                new_params.append(f"    {param} {value}")
                updated_params.append(f"{param} {value} (待添加)")
    
    # 如果有新参数需要添加，统一添加到GPT_ARGS中
    if new_params:
        # 找到GPT_ARGS的结束位置
        gpt_args_pattern = r'(GPT_ARGS="[^"]*?)(\s*")'
        match = re.search(gpt_args_pattern, updated_content, flags=re.DOTALL)
        
        if match:
            # 首先确保原始GPT_ARGS的最后一个参数后面有\和换行
            gpt_content = match.group(1)
            
            # 检查最后一个参数是否已经有\，如果没有则添加
            if not gpt_content.rstrip().endswith('\\'):
                # 找到最后一个非空行，在其后添加\
                lines = gpt_content.split('\n')
                last_non_empty_line_idx = -1
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip():
                        last_non_empty_line_idx = i
                        break
                
                if last_non_empty_line_idx >= 0:
                    # 在最后一个非空行后添加\
                    lines[last_non_empty_line_idx] = lines[last_non_empty_line_idx].rstrip() + ' \\'
                    gpt_content = '\n'.join(lines)
            
            # 构建新的参数列表，每行一个参数，除最后一行外都加\
            new_params_text = ""
            for i, param in enumerate(new_params):
                if i == len(new_params) - 1:
                    # 最后一行不加\
                    new_params_text += f"{param}\n"
                else:
                    # 其他行都加\
                    new_params_text += f"{param} \\\n"
            
            # 替换GPT_ARGS部分，在gpt_content和new_params_text之间添加换行
            replacement = f'{gpt_content}\n{new_params_text}\\2'
            updated_content = re.sub(gpt_args_pattern, replacement, updated_content, flags=re.DOTALL)
            
            # 更新状态信息
            for i, param_info in enumerate(updated_params):
                if "(待添加)" in param_info:
                    updated_params[i] = param_info.replace("(待添加)", "(已添加)")
    
    print("参数更新情况：")
    for param_info in updated_params:
        print(f"  {param_info}")
    
    return updated_content

def main(
    input_file,
    output_dir,
    input_bash_dir=None,
    output_bash_dir=None,
    data_path_override=None,
    train_iters=100,
    model_name=None,
    enable_deepseek_profile=False,
):
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    import json
    if not input_bash_dir:
        raise ValueError("输入原始脚本路径不能为空，请使用 -isc 参数指定输入脚本路径")
    if not output_bash_dir:
        raise ValueError("输出脚本路径不能为空，请使用 -osc 参数指定输出脚本路径")
    # 解析输入文件，生成标准的yaml配置文件
    parser = InfoParser()
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    bashtoyaml_path = output_path / "bashtoyaml.yaml"
    config_data_res = bashtoyaml(input_bash_dir, bashtoyaml_path)
    print("++++++++++++++++++++++++++++++++++++")
    print(config_data_res)
    print("++++++++++++++++++++++++++++++++++++")
    # 允许通过参数覆盖数据路径，便于快速切换
    if data_path_override:
        config_data_res['paths'] = config_data_res.get('paths', {})
        config_data_res['paths']['data_path'] = data_path_override
        config_data_res['paths']['DATA_PATH'] = data_path_override

    if train_iters is not None:
        config_data_res['train-iters']=train_iters
    config_data_res = _merge_mutation_sections(config_data_res, data.get('1', {}))

    config_data_path = output_path / "config_data.yaml"

    try:
        with open(config_data_path, 'w', encoding='utf-8') as file:
            yaml.dump(config_data_res, file)
            print(f"已保存合并后的配置到 {config_data_path}")
    except Exception as e:
        print(f"保存bashtoyaml 合并后配置时出错: {e}")
        return

    parsed_file = parser.parse_file(config_data_path)
    print("=====", parsed_file)
    
    
    
    standard_yaml_path = output_path / "standard_config.yaml"
    with open(standard_yaml_path, 'w') as f:
        yaml.dump(parsed_file, f)

    # 读取标准yaml配置文件
    with open(standard_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # 变异并行参数, 只进行一次变异
    mutator = ParallelParameterMutator(config)
    mutated_config, mutated_changes = mutator.mutate_parallel_parameters()
    print("变异的并行参数及其新值：", mutated_changes)
    mutated_yaml_path = output_path / "mutated_config.yaml"
    with open(mutated_yaml_path, 'w') as f:
        yaml.dump(mutated_config, f)
    print(f"变异后的配置已保存到 {mutated_yaml_path}")

    # mutated_config['parallel']['tensor_model_parallel_size'] = 1
    # mutated_config['parallel']['expert_model_parallel_size'] = 1
    
    # 校验并修复配置
    validator = EnhancedMegatronConfigValidator(mutated_config)
    validated_config, validated_issues, validated_warnings, validated_applied = validator.validate_and_fix()
    # 打印校验结果
    if validated_issues:
        print("校验发现的问题及其修复：", validated_issues)
    else:
        print("未发现严重问题。")
    if validated_warnings:
        print("校验发现的警告及其修复：", validated_warnings)
    else:
        print("未发现警告。")
    if validated_applied:
        print("自动修复的配置项及其新值：", validated_applied)
    else:
        print("未进行任何自动修复。")
    validated_yaml_path = output_path / "validated_config.yaml"
    with open(validated_yaml_path, 'w') as f:
        yaml.dump(validated_config, f)
    print(f"校验并修复后的配置已保存到 {validated_yaml_path}")

    # 提取所有二级参数并转换为sh脚本参数形式
    sh_args = extract_parameters_to_sh_args(validated_config)
    print("提取的sh脚本参数：")
    if "--layernorm-epsilon" in sh_args:
        sh_args["--norm-epsilon"] = sh_args["--layernorm-epsilon"]
        sh_args.pop("--layernorm-epsilon")
    if "--num-moe-experts" in sh_args:
        sh_args["--num-experts"] = sh_args["--num-moe-experts"]
        sh_args.pop("--num-moe-experts")    
    
    for param, value in sh_args.items():
            
        if value is None:
            print(f"  {param}")
        else:
            print(f"  {param} {value}")
    
    # 将参数保存到文件
    sh_args_path = output_path / "sh_arguments.txt"
    with open(sh_args_path, 'w') as f:
        for param, value in sh_args.items():
            if value is None:
                f.write(f"{param}\n")
            else:
                f.write(f"{param} {value}\n")
    print(f"sh脚本参数已保存到 {sh_args_path}")
    import os
    # # 更新脚本参数（如果提供了脚本路径）
    # script_path = args.input_bash_dir
    # if script_path and Path(script_path).exists():
    #     try:
    #         updated_script = update_script_parameters(script_path, sh_args)
    #         updated_script_path = args.output_bash_dir
    #         with open(updated_script_path, 'w', encoding='utf-8') as f:
    #             f.write(updated_script)
    #         print(f"更新后的脚本已保存到 {updated_script_path}")
    #     except Exception as e:
    #         print(f"更新脚本时出错: {e}")
    # elif script_path:
    #     print(f"脚本文件不存在: {script_path}")

    # 将最终配置转换为bash脚本
    bash_converter = YamlToBashConverter(validated_config)
    bash_script_path = output_bash_dir
    bash_str = bash_converter.convert()
    try:
        main_dir = os.path.dirname(bash_script_path)
        os.makedirs(main_dir, exist_ok=True)
        save_bash_script(bash_str, bash_script_path)
        if enable_deepseek_profile and str(model_name).strip().lower() == "deepseekv3":
            apply_deepseekv3_unified_low_memory_profile(bash_script_path)
        print(f"Bash脚本已保存到 {bash_script_path}")
    except Exception as e:
        print(f"yamlToBash 保存Bash脚本时出错: {e}")
        raise

def run_cli():
    parser = argparse.ArgumentParser(description="Megatron-LM 配置一键变异+校验修复工具")
    # 使用 -i, -o 作为输入输出参数的简写
    parser.add_argument("-i", "--input_file", type=str, required=False, help="输入配置文件路径")
    parser.add_argument("-o", "--output_dir", type=str, required=False, help="输出目录路径")
    parser.add_argument("-isc", "--input_bash_dir", type=str, required=False, help="输入原始脚本路径")
    parser.add_argument("-osc", "--output_bash_dir", type=str, required=False, help="输出脚本路径")
    parser.add_argument("--data_path", type=str, required=False, help="覆盖配置中的 data_path/DATA_PATH")
    parser.add_argument("--train_iters", type=int, required=False, help="覆盖配置中的 train_iters/TRAIN_ITERS")
    parser.add_argument("--model_name", type=str, required=False, help="模型名称，用于特定 profile 调整")
    parser.add_argument("--enable_deepseek_profile", action="store_true", help="启用 DeepSeekV3 低显存脚本调整")
    
    args = parser.parse_args()
    if not args.input_file:
        raise ValueError("输入文件路径不能为空，请使用 -i 参数指定输入文件路径")
    
    if not args.output_dir:
        output_dir = "./pretrain_mutated/general/pta"
        main(
            args.input_file,
            output_dir,
            args.input_bash_dir,
            args.output_bash_dir,
            args.data_path,
            args.train_iters,
            args.model_name,
            args.enable_deepseek_profile,
        )
    else:
        main(
            args.input_file,
            args.output_dir,
            args.input_bash_dir,
            args.output_bash_dir,
            args.data_path,
            args.train_iters,
            args.model_name,
            args.enable_deepseek_profile,
        )


if __name__ == "__main__":
    run_cli()
