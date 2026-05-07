import re
import yaml
from typing import Dict, Any, List, Union, Optional
from pathlib import Path


class InfoParser:
    """
    A robust parser to convert arguments.yaml format to ds_lite_problem.yaml format.
    Uses line-by-line parsing to handle non-standard YAML formats with command line arguments.
    """
    
    def __init__(self):
        self.warnings_list = []
        self.current_section = None
        self.result = {}
        self.flat_params = {}
        
        # 参数映射字典 - 完全基于 ds_lite_fix.yaml 的结构
        self.param_mapping = {
            # Environment variables
            'CUDA_DEVICE_MAX_CONNECTIONS': ('environment', 'CUDA_DEVICE_MAX_CONNECTIONS'),
            'PYTORCH_NPU_ALLOC_CONF': ('environment', 'PYTORCH_NPU_ALLOC_CONF'),
            
            # Distributed settings
            'NPUS_PER_NODE': ('distributed', 'npus_per_node'),
            'MASTER_ADDR': ('distributed', 'master_addr'),
            'MASTER_PORT': ('distributed', 'master_port'),
            'NNODES': ('distributed', 'num_nodes'),
            'NODE_RANK': ('distributed', 'node_rank'),
            'WORLD_SIZE': ('distributed', 'world_size'),
            
            
            # Parallel settings
            'tensor_model_parallel_size': ('parallel', 'tensor_model_parallel_size'),
            'pipeline_model_parallel_size': ('parallel', 'pipeline_model_parallel_size'),
            'expert_model_parallel_size': ('parallel', 'expert_model_parallel_size'),
            'context_parallel_size': ('parallel', 'context_parallel_size'),
            'sequence_parallel': ('parallel', 'sequence_parallel'),
            'num_layers_per_virtual_pipeline_stage': ('parallel', 'num_layers_per_virtual_pipeline_stage'),

            # Path settings
            'DATA_PATH': ('paths', 'DATA_PATH'),
            'VOCAB_FILE': ('paths', 'VOCAB_FILE'),
            'MERGE_FILE': ('paths', 'MERGE_FILE'),
            'CKPT_SAVE_DIR': ('paths', 'CKPT_SAVE_DIR'),
            'TOKENIZER_MODEL': ('paths', 'TOKENIZER_MODEL'),
            
            # Model settings
            'spec': ('model', 'spec'),
            'transformer_impl': ('model', 'transformer_impl'),
            'num_layers': ('model', 'num_layers'),
            'hidden_size': ('model', 'hidden_size'),
            'ffn_hidden_size': ('model', 'ffn_hidden_size'),
            'num_attention_heads': ('model', 'num_attention_heads'),
            'num_query_groups': ('model', 'num_query_groups'),
            'group_query_attention': ('model', 'group_query_attention'),
            'vocab_size': ('model', 'vocab_size'),
            'padded_vocab_size': ('model', 'padded_vocab_size'),
            'make_vocab_size_divisible_by': ('model', 'make_vocab_size_divisible_by'),
            'seq_length': ('model', 'seq_length'),
            'max_position_embeddings': ('model', 'max_position_embeddings'),
            'position_embedding_type': ('model', 'position_embedding_type'),
            'rotary_base': ('model', 'rotary_base'),
            'kv_channels': ('model', 'kv_channels'),
            'normalization': ('model', 'normalization'),
            'norm_epsilon': ('model', 'norm_epsilon'),
            'init_method_std': ('model', 'init_method_std'),
            'attention_dropout': ('model', 'attention_dropout'),
            'hidden_dropout': ('model', 'hidden_dropout'),
            'disable_bias_linear': ('model', 'disable_bias_linear'),
            'untie_embeddings_and_output_weights': ('model', 'untie_embeddings_and_output_weights'),
            'swiglu': ('model', 'swiglu'),
            'use_flash_attn': ('model', 'use_flash_attn'),
            'use_mcore_models': ('model', 'use_mcore_models'),
            'use_fused_rotary_pos_emb': ('model', 'use_fused_rotary_pos_emb'),
            'use_rotary_position_embeddings': ('model', 'use_rotary_position_embeddings'),
            'use_fused_swiglu': ('model', 'use_fused_swiglu'),
            'use_fused_rmsnorm': ('model', 'use_fused_rmsnorm'),
            'no_masked_softmax_fusion': ('model', 'no_masked_softmax_fusion'),
            'attention_softmax_in_fp32': ('model', 'attention_softmax_in_fp32'),
            'no_gradient_accumulation_fusion': ('model', 'no_gradient_accumulation_fusion'),
            'reuse_fp32_param': ('model', 'reuse_fp32_param'),
            
            # MLA settings
            'multi_head_latent_attention': ('mla', 'multi_head_latent_attention'),
            'q_lora_rank': ('mla', 'q_lora_rank'),
            'qk_rope_head_dim': ('mla', 'qk_rope_head_dim'),
            'qk_nope_head_dim': ('mla', 'qk_nope_head_dim'),
            'kv_lora_rank': ('mla', 'kv_lora_rank'),
            'v_head_dim': ('mla', 'v_head_dim'),
            'qk_layernorm': ('mla', 'qk_layernorm'),
            
            # MoE settings
            'moe_grouped_gemm': ('moe', 'moe_grouped_gemm'),
            'moe_permutation_async_comm': ('moe', 'moe_permutation_async_comm'),
            'moe_token_dispatcher_type': ('moe', 'moe_token_dispatcher_type'),
            'use_fused_moe_token_permute_and_unpermute': ('moe', 'use_fused_moe_token_permute_and_unpermute'),
            'first_k_dense_replace': ('moe', 'first_k_dense_replace'),
            'moe_layer_freq': ('moe', 'moe_layer_freq'),
            'n_shared_experts': ('moe', 'n_shared_experts'),
            'expert_num': ('moe', 'num_experts'),
            'num_moe_experts': ('moe', 'num_experts'),
            'num_experts': ('moe', 'num_experts'),
            'per_token_num_experts_chosen': ('moe', 'moe_router_topk'),
            'num_experts_chosen': ('moe', 'moe_router_topk'),
            'moe_router_topk': ('moe', 'moe_router_topk'),
            'moe_router_pre_softmax': ('moe', 'moe_router_pre_softmax'),
            'moe_router_group_topk': ('moe', 'moe_router_group_topk'),
            'moe_intermediate_size': ('moe', 'moe_intermediate_size'),
            'moe_router_load_balancing_type': ('moe', 'moe_router_load_balancing_type'),
            'moe_router_num_groups': ('moe', 'moe_router_num_groups'),
            'topk_group': ('moe', 'topk_group'),
            'moe_aux_loss_coeff': ('moe', 'moe_aux_loss_coeff'),
            'routed_scaling_factor': ('moe', 'routed_scaling_factor'),
            'seq_aux': ('moe', 'seq_aux'),
            
            # Rope settings
            'rope_scaling_type': ('rope', 'rope_scaling_type'),
            'rope_scaling_factor': ('rope', 'rope_scaling_factor'),
            'rope_scaling_beta_fast': ('rope', 'rope_scaling_beta_fast'),
            'rope_scaling_beta_slow': ('rope', 'rope_scaling_beta_slow'),
            'rope_scaling_mscale': ('rope', 'rope_scaling_mscale'),
            'rope_scaling_mscale_all_dim': ('rope', 'rope_scaling_mscale_all_dim'),
            'rope_scaling_original_max_position_embeddings': ('rope', 'rope_scaling_original_max_position_embeddings'),
            
            # Training settings
            'micro_batch_size': ('training', 'micro_batch_size'),
            'global_batch_size': ('training', 'global_batch_size'),
            'train_iters': ('training', 'train_iters'),
            'lr': ('training', 'lr'),
            'lr_decay_style': ('training', 'lr_decay_style'),
            'lr_decay_iters': ('training', 'lr_decay_iters'),
            'min_lr': ('training', 'min_lr'),
            'weight_decay': ('training', 'weight_decay'),
            'lr_warmup_iters': ('training', 'lr_warmup_iters'),
            'clip_grad': ('training', 'clip_grad'),
            'adam_beta1': ('training', 'adam_beta1'),
            'adam_beta2': ('training', 'adam_beta2'),
            'initial_loss_scale': ('training', 'initial_loss_scale'),
            'bf16': ('training', 'bf16'),
            'finetune': ('training', 'finetune'),
            'use_distributed_optimizer': ('training', 'use_distributed_optimizer'),
            'no_load_optim': ('training', 'no_load_optim'),
            'no_load_rng': ('training', 'no_load_rng'),
            'recompute_activations': ('training', 'recompute_activations'),
            'recompute_granularity': ('training', 'recompute_granularity'),
            'recompute_method': ('training', 'recompute_method'),
            'recompute_num_layers': ('training', 'recompute_num_layers'),
            'distribute_saved_activations': ('training', 'distribute_saved_activations'),
            
            # Data settings
            'split': ('data', 'split'),
            
            # Output settings
            'log_interval': ('output', 'log_interval'),
            'save_interval': ('output', 'save_interval'),
            'eval_interval': ('output', 'eval_interval'),
            'eval_iters': ('output', 'eval_iters'),
            'no_save_optim': ('output', 'no_save_optim'),
            'no_save_rng': ('output', 'no_save_rng'),
            'log_file': ('output', 'log_file'),
            
            # Launcher settings
            'distributed_backend': ('launcher', 'distributed_backend'),
            'nproc_per_node': ('launcher', 'nproc_per_node'),
            'nnodes': ('launcher', 'nnodes'),
            'node_rank': ('launcher', 'node_rank'),
            'master_addr': ('launcher', 'master_addr'),
            'master_port': ('launcher', 'master_port'),
        }
        
        # 基于关键词的分类映射 - 提高扩展性
        self.keyword_section_mapping = {
            'environment': ['CUDA_DEVICE', 'PYTORCH', 'OMP', 'NCCL', 'GLOO'],
            
            'distributed': ['master_addr', 'master_port', 'world_size', 'rank', 'node', 'nnodes', 
                          'npus_per_node', 'gpus_per_node', 'nproc_per_node'],
            
            'parallel': ['tensor_model_parallel', 'pipeline_model_parallel', 'expert_model_parallel', 
                        'context_parallel', 'data_parallel', 'num_layers_per_virtual_pipeline_stage'],
            
            'paths': ['path', 'dir', 'file', 'vocab', 'merge', 'tokenizer', 'checkpoint', 'save', 'load'],
            
            'model': ['spec', 'num_layers', 'hidden_size', 'ffn_hidden_size', 'num_attention_heads', 
                     'num_query_groups', 'group_query_attention', 'kv_channels',
                     'vocab_size', 'padded_vocab_size', 'make_vocab_size_divisible_by', 'seq_length',
                     'max_position_embeddings', 'position_embedding_type', 'rotary_base', 
                     'normalization', 'norm_epsilon', 'init_method_std', 'attention_dropout',
                     'hidden_dropout', 'disable_bias_linear', 'untie_embeddings_and_output_weights',
                     'swiglu', 'use_flash_attn', 'use_mcore_models', 'use_fused_rotary_pos_emb',
                     'use_rotary_position_embeddings', 'use_fused_swiglu', 'use_fused_rmsnorm',
                     'no_masked_softmax_fusion', 'attention_softmax_in_fp32', 
                     'no_gradient_accumulation_fusion', 'reuse_fp32_param', 'layer', 'embed', 
                     'norm', 'dropout', 'position', 'transformer', 'tokenizer', 'sequence', 
                     'flash', 'fused', 'mcore', 'bias'],
            
            'mla': ['multi_head_latent_attention', 'q_lora_rank', 'qk_rope_head_dim', 'qk_nope_head_dim', 
                   'kv_lora_rank', 'v_head_dim', 'qk_layernorm', 'multi-head-latent', 'qk-', 
                   'kv-', 'lora', 'nope'],
            
                 'moe': ['moe_grouped_gemm', 'moe_permutation_async_comm', 'moe_token_dispatcher_type',
                   'use_fused_moe_token_permute_and_unpermute', 'first_k_dense_replace', 
                     'moe_layer_freq', 'n_shared_experts', 'expert_num', 'num_moe_experts',
                     'num_experts', 'per_token_num_experts_chosen', 'num_experts_chosen', 'moe_router_topk',
                     'moe_router_pre_softmax',
                     'moe_router_group_topk', 'moe_intermediate_size', 'moe_router_load_balancing_type', 'topk_group',
                   'moe_aux_loss_coeff', 'routed_scaling_factor', 'seq_aux', 'moe', 'expert', 
                   'router', 'topk', 'aux', 'permute', 'dispatch', 'gemm', 'grouped', 'shared'],
            
            'rope': ['rope_scaling_type', 'rope_scaling_factor', 'rope_scaling_beta_fast',
                    'rope_scaling_beta_slow', 'rope_scaling_mscale', 'rope_scaling_mscale_all_dim',
                    'rope_scaling_original_max_position_embeddings', 'rope'],
            
            'training': ['micro_batch_size', 'global_batch_size', 'train_iters', 'lr', 
                        'lr_decay_style', 'lr_decay_iters', 'min_lr', 'weight_decay', 
                        'lr_warmup_iters', 'clip_grad', 'adam_beta1', 'adam_beta2', 
                        'initial_loss_scale', 'bf16', 'finetune', 'use_distributed_optimizer',
                        'no_load_optim', 'no_load_rng', 'recompute_activations',
                        'recompute_granularity', 'recompute_method', 'recompute_num_layers',
                        'distribute_saved_activations', 'batch', 'train', 'weight', 'decay', 
                        'warmup', 'clip', 'adam', 'loss', 'scale', 'optim', 'rng', 'seed'],
            
            'data': ['split', 'data_path', 'dataset'],
            
            'output': ['log_interval', 'save_interval', 'eval_interval', 'eval_iters', 
                      'no_save_optim', 'no_save_rng', 'log_file', 'log', 'save', 'eval', 
                      'interval', 'iters'],
            
            'launcher': ['distributed_backend', 'nccl', 'gloo']
        }
    
    def parse_file(self, input_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Parse input file line by line and convert to target format.
        
        Args:
            input_path: Path to input file
            output_path: Optional path to save converted YAML
            
        Returns:
            Converted configuration dictionary
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 第一阶段：解析所有参数到扁平字典
            self.flat_params = {}
            self.current_section = None
            
            for line_num, line in enumerate(lines, 1):
                self._parse_line_flat(line, line_num)
            
            # 第二阶段：通过映射字典转换为分层结构
            self.result = {}
            self._convert_flat_to_hierarchical()
            
            # Save if output path provided
            if output_path:
                self._save_config(self.result, output_path)
            
            # Print warnings
            self._print_warnings()
            
            return self.result
            
        except Exception as e:
            self._add_warning(f"Error parsing file {input_path}: {str(e)}")
            raise
    
    def _parse_line_flat(self, line: str, line_num: int):
        """第一阶段：解析单行到扁平字典"""
        original_line = line
        line = line.rstrip()
        
        # Skip empty lines
        if not line.strip():
            return
        
        # Skip comment lines
        if line.strip().startswith('#'):
            return
        
        # Detect section headers (记录当前section但不创建层级)
        if not line.startswith(' ') and not line.startswith('\t') and line.endswith(':'):
            section_name = line[:-1].strip()
            self.current_section = section_name
            return
        
        # Skip lines without a current section
        if self.current_section is None:
            self._add_warning(f"Line {line_num}: Found content outside of any section: {line.strip()}")
            return
        
        # Parse content lines to flat dictionary
        self._parse_content_line_flat(line, line_num)
    
    def _parse_content_line_flat(self, line: str, line_num: int):
        """解析内容行到扁平字典"""
        line = line.strip()
        
        if not line:
            return
        
        # Handle key: value format (标准YAML格式)
        if ':' in line and not line.startswith('--'):
            self._parse_key_value_colon(line, line_num)
            return
        
        # Handle command line arguments (start with --)
        if line.startswith('--'):
            self._parse_command_line_arg_flat(line, line_num)
            return
        
        # Handle key=value format
        if '=' in line:
            self._parse_key_value_equals_flat(line, line_num)
            return
        
        # Handle key value format (space separated)
        if ' ' in line:
            self._parse_key_value_space_flat(line, line_num)
            return
        
        # Handle standalone keys (boolean flags)
        self._parse_standalone_key_flat(line, line_num)
    
    def _parse_key_value_colon(self, line: str, line_num: int):
        """解析 key: value 格式"""
        if ':' not in line:
            return
        
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        
        # Handle special cases
        if value == 'None' or value == 'null':
            self._add_warning(f"Line {line_num}: Skipping None/null value for {key}")
            return
        
        # Remove quotes
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        
        # Convert and format value
        value = self._convert_and_format_value(value)
        
        self.flat_params[key] = value
    
    def _parse_command_line_arg_flat(self, line: str, line_num: int):
        """解析命令行参数到扁平字典"""
        # Remove trailing backslash if present
        if line.endswith('\\'):
            line = line[:-1].strip()
            self._add_warning(f"Line {line_num}: Removed trailing backslash")
        
        parts = line.split()
        
        if len(parts) == 1:
            # Boolean flag like --use-flash-attn
            key = parts[0]
            value = True
        elif len(parts) == 2:
            # Key-value pair like --num-layers 8
            key = parts[0]
            value = self._convert_and_format_value(parts[1])
        else:
            # Multiple values - join them
            key = parts[0]
            value = ' '.join(parts[1:])
            # Remove quotes if they wrap the entire value
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            value = self._convert_and_format_value(value)
        
        self.flat_params[key] = value
    
    def _parse_key_value_equals_flat(self, line: str, line_num: int):
        """解析 key=value 格式到扁平字典"""
        if '=' not in line:
            return
        
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # Handle special cases
        if value == 'None':
            self._add_warning(f"Line {line_num}: Skipping None value for {key}")
            return
        
        # Remove quotes
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        
        # Convert and format value
        value = self._convert_and_format_value(value)
        
        self.flat_params[key] = value
    
    def _parse_key_value_space_flat(self, line: str, line_num: int):
        """解析空格分隔的key value格式到扁平字典"""
        parts = line.split(None, 1)  # Split on first whitespace only
        
        if len(parts) != 2:
            self._add_warning(f"Line {line_num}: Could not parse space-separated key-value: {line}")
            return
        
        key, value = parts
        
        # Remove quotes
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        
        # Convert and format value
        value = self._convert_and_format_value(value)
        
        self.flat_params[key] = value
    
    def _parse_standalone_key_flat(self, line: str, line_num: int):
        """解析独立key到扁平字典"""
        key = line.strip()
        
        # For command line sections, treat standalone keys as boolean flags
        if self.current_section in ['model', 'mla', 'moe', 'rope', 'training', 'output']:
            # Add -- prefix if not present
            if not key.startswith('--'):
                key = f"--{key}"
            self.flat_params[key] = True
        else:
            self._add_warning(f"Line {line_num}: Unexpected standalone key in {self.current_section}: {key}")
    
    def _convert_and_format_value(self, value: str) -> Union[str, int, float, bool]:
        """转换值类型并根据规则手动添加引号"""
        if not isinstance(value, str):
            return value
        
        value = value.strip()
        
        # Handle boolean values (不加引号)
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Handle numeric values (整数，不加引号)
        if value.isdigit():
            return int(value)
        
        # Handle negative numbers
        if value.startswith('-') and value[1:].isdigit():
            return int(value)
        
        # Handle floating point numbers (浮点数，不加引号)
        try:
            if '.' in value or 'e' in value.lower():
                float_val = float(value)
                return float_val
        except ValueError:
            pass
        
        # Handle localhost (不加引号)
        if value == 'localhost':
            return value
        
        # Handle variable substitutions (不加引号)
        if value.startswith('$'):
            return value
        
        # Handle already quoted strings (移除外层引号，不再添加)
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            # 移除外层引号
            unquoted_value = value[1:-1]
            # 检查是否是特殊值
            if unquoted_value.lower() in ['true', 'false']:
                return unquoted_value.lower() == 'true'
            elif unquoted_value.isdigit():
                return int(unquoted_value)
            elif unquoted_value == 'localhost':
                return unquoted_value
            elif unquoted_value.startswith('$'):
                return unquoted_value
            else:
                # 其他情况返回不带引号的字符串，让后面的逻辑处理
                return unquoted_value
        
        # Handle multiple quotes (like '''megatron_cp_algo''')
        if value.startswith("'''") and value.endswith("'''"):
            unquoted_value = value[3:-3]
            # 对于这种情况，直接返回内部的字符串值
            return unquoted_value
        
        # 对于其他所有字符串值，手动添加双引号
        return f"{value}"
    
    def _convert_flat_to_hierarchical(self):
        """第二阶段：将扁平字典转换为分层结构"""
        # Initialize all sections
        sections = ['environment', 'distributed', 'parallel', 'paths', 'model', 
                    'mla', 'moe', 'rope', 'training', 'data', 'output', 'launcher']
        
        for section in sections:
            self.result[section] = {}
        
        # Convert known parameters using mapping
        for flat_key, value in self.flat_params.items():
            normalized_key = self._normalize_lookup_key(flat_key)
            if normalized_key in self.param_mapping:
                section, target_key = self.param_mapping[normalized_key]
                self.result[section][target_key] = value
            else:
                # Add a warning for unmapped parameters
                self._add_warning(f"Unmapped parameter: {flat_key} (value: {value})")
                # Try to categorize by keywords as fallback
                self._categorize_by_keywords(flat_key, value)
        
        # Remove empty sections
        self.result = {k: v for k, v in self.result.items() if v}

    def _categorize_by_keywords(self, key: str, value: Any):
        """使用关键词映射对参数进行分类"""
        search_key = self._normalize_lookup_key(key).lower()
        
        # Check each section's keywords
        for section, keywords in self.keyword_section_mapping.items():
            for keyword in keywords:
                if keyword.lower() in search_key:
                    # Convert key to snake_case format for consistency
                    target_key = self._convert_key_format(key)
                    self.result[section][target_key] = value
                    return
        
        # If no keyword match found, try the old categorization method as fallback
        self._categorize_unknown_param(key, value)

    def _convert_key_format(self, key: str) -> str:
        """将key转换为一致的格式"""
        # Remove -- prefix
        if key.startswith('--'):
            key = key[2:]
        
        # Convert kebab-case to snake_case
        key = key.replace('-', '_')

        return key

    def _normalize_lookup_key(self, key: str) -> str:
        """规范化键名，便于映射和关键词匹配。"""
        return self._convert_key_format(key).strip()

    def _categorize_unknown_param(self, key: str, value: Any):
        """对未知参数进行分类 - 作为关键词匹配的后备方案"""
        # Default to model section for other -- parameters
        if key.startswith('--'):
            target_key = self._convert_key_format(key)
            self.result['model'][target_key] = value
            self._add_warning(f"Categorized unknown parameter {key} as model parameter")
            return
        
        # Non -- parameters, try to categorize by name pattern
        if any(keyword in key.upper() for keyword in ['ADDR', 'PORT', 'RANK', 'NODE', 'WORLD']):
            self.result['distributed'][key] = value
            return
        
        if any(keyword in key.upper() for keyword in ['PATH', 'DIR', 'FILE']):
            self.result['paths'][key] = value
            return
        
        # Default warning for completely unknown parameters
        self._add_warning(f"Could not categorize parameter: {key}")

    def _add_warning(self, message: str):
        """Add a warning message."""
        self.warnings_list.append(message)
    
    def _print_warnings(self):
        """Print all collected warnings."""
        if self.warnings_list:
            print("\n⚠️  Conversion Warnings:")
            for i, warning in enumerate(self.warnings_list, 1):
                print(f"  {i}. {warning}")
            print()
    
    def _save_config(self, config: Dict[str, Any], output_path: str):
        """Save converted configuration to file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            print(f"✅ Converted configuration saved to: {output_path}")
        except Exception as e:
            self._add_warning(f"Error saving to {output_path}: {str(e)}")


def main():
    """Main function for command line usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python InfoParser.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not output_file:
        # Generate output filename
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_converted.yaml")
    
    parser = InfoParser()
    try:
        converted_config = parser.parse_file(input_file, output_file)
        print(f"✅ Successfully converted {input_file}")
        
        if parser.warnings_list:
            print(f"📊 Total warnings: {len(parser.warnings_list)}")
        else:
            print("🎉 No warnings - clean conversion!")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
