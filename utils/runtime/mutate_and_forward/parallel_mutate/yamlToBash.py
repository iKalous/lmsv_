#!/usr/bin/env python3
"""
YAML 到 Bash 转换器
将 Megatron-LM YAML 配置文件转换为对应的 Bash 启动脚本
"""

import yaml
import argparse
import os
from typing import Dict, Any


PTA_UNSUPPORTED_SCRIPT_FLAGS = (
    "--use-fused-swiglu",
    "--use-fused-rmsnorm",
    "--use-fused-rotary-pos-emb",
)


def strip_pta_unsupported_script_flags(script: str) -> str:
    """Delete unsupported flags when emitting PTA bash scripts."""
    cleaned_lines = []
    for line in script.splitlines():
        stripped = line.strip()
        if any(
            stripped == f"{flag} \\"
            or stripped == flag
            or stripped.startswith(f"{flag} ")
            for flag in PTA_UNSUPPORTED_SCRIPT_FLAGS
        ):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

class YamlToBashConverter:
    def __init__(self, yaml_config: Dict[str, Any]):
        self.config = yaml_config
        self.bash_lines = []

    @staticmethod
    def _normalize_positive_int(value: Any, default: int | None = None) -> int | None:
        if value in (None, "", "None", "null"):
            return default
        try:
            normalized = int(value)
            return normalized if normalized > 0 else default
        except (TypeError, ValueError):
            return default

    def _normalize_expert_count(self, value: Any) -> int:
        if value in (None, "", "None", "null"):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _split_csv_values(value: Any) -> list[str]:
        if value in (None, "", "None", "null"):
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if str(item).strip()]
        return [item.strip() for item in str(value).split(",") if item.strip()]

    def _infer_rope_factor_count(self) -> int | None:
        model = self.config.get('model', {})
        kv_channels = self._normalize_positive_int(model.get('kv_channels'))
        if kv_channels is None:
            kv_channels = self._normalize_positive_int(model.get('qk_rope_head_dim'))
        if kv_channels is None:
            kv_channels = self._normalize_positive_int(model.get('qk_pos_emb_head_dim'))
        if kv_channels is None:
            hidden_size = self._normalize_positive_int(model.get('hidden_size'))
            num_attention_heads = self._normalize_positive_int(model.get('num_attention_heads'))
            if hidden_size is not None and num_attention_heads is not None and num_attention_heads > 0:
                kv_channels = hidden_size // num_attention_heads if hidden_size % num_attention_heads == 0 else None
        if kv_channels is None or kv_channels <= 0:
            return None
        return (kv_channels + 1) // 2

    def _has_moe(self) -> bool:
        moe = self.config.get('moe', {})
        return self._normalize_expert_count(
            moe.get('num_experts', moe.get('num_moe_experts'))
        ) > 1
        
    def convert(self) -> str:
        """将 YAML 配置转换为 Bash 脚本"""
        self.bash_lines = [""]
        
        # 添加环境变量
        self._add_environment_vars()
        
        # 添加分布式参数
        self._add_distributed_params()
        
        # # 添加并行策略参数
        # self._add_parallel_params()
        
        # 添加路径参数
        self._add_path_params()
        
        # 添加模型参数
        self._add_model_params()
        
        # 添加 MLA 参数
        self._add_mla_params()
        
        # 添加 MOE 参数
        self._add_moe_params()
        
        # 添加 ROPE 参数
        self._add_rope_params()
        
        # # 添加训练参数
        # self._add_training_params()
        
        # 添加数据参数
        self._add_data_params()
        
        # 添加输出参数
        self._add_output_params()
        
        # 构建分布式参数变量
        self._build_final_command()
        
        return "\n".join(self.bash_lines)
    
    def _add_environment_vars(self) -> None:
        """添加环境变量"""
        env_vars = self.config.get('environment', {})
        for key, value in env_vars.items():
            self.bash_lines.append(f"export {key}={value}")
        self.bash_lines.append("")
    
    def _add_distributed_params(self) -> None:
        """添加分布式参数"""
        dist = self.config.get('distributed', {})
        
        npus_per_node = dist.get('npus_per_node', 8)
        num_nodes = dist.get('num_nodes', 1)
        master_addr = dist.get('master_addr', 'localhost')
        master_port = dist.get('master_port', 6000)
        node_rank = dist.get('node_rank', 0)
        
        self.bash_lines.append(f"NPUS_PER_NODE={npus_per_node}")
        self.bash_lines.append(f"MASTER_ADDR={master_addr}")
        self.bash_lines.append(f"MASTER_PORT={master_port}")
        self.bash_lines.append(f"NNODES={num_nodes}")
        self.bash_lines.append(f"NODE_RANK={node_rank}")
        self.bash_lines.append('WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))')
        self.bash_lines.append("")
    
    def _get_parallel_params(self) -> list[str]:
        """添加并行策略参数"""
        parallel = self.config.get('parallel', {})
        model = self.config.get('model', {})
        params_str = []
        has_moe = self._has_moe()
        position_type = str(model.get('position_embedding_type', '') or '').strip().lower()
        # 遍历所有 key value
        for key, value in parallel.items():
            if key == 'expert_model_parallel_size' and not has_moe:
                continue
            if key == 'context_parallel_size':
                cp = self._normalize_positive_int(value, 1)
                # 对默认 CP=1 不显式下发，避免上游 alibi feature 读取成字符串。
                if cp is None or cp <= 1:
                    continue
                if position_type == 'alibi':
                    continue
                value = cp
            elif key in {
                'tensor_model_parallel_size',
                'pipeline_model_parallel_size',
                'expert_model_parallel_size',
                'num_layers_per_virtual_pipeline_stage',
            }:
                normalized = self._normalize_positive_int(value, None)
                if normalized is not None:
                    value = normalized
            if value in (None, "", "None", "null"):
                continue
            param_key = key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    params_str.append(f'    --{param_key} \\')
            else:
                params_str.append(f'    --{param_key} {value} \\')
        return params_str
        # tp = parallel.get('tensor_model_parallel_size', 1)
        # pp = parallel.get('pipeline_model_parallel_size', 1)
        # ep = parallel.get('expert_model_parallel_size', 1)
        # cp = parallel.get('context_parallel_size', 1)
        
        # self.bash_lines.append(f"TP={tp}")
        # self.bash_lines.append(f"PP={pp}")
        # self.bash_lines.append(f"EP={ep}")
        # self.bash_lines.append(f"CP={cp}")
        # self.bash_lines.append("")

    
    def _add_path_params(self) -> None:
        """添加路径参数"""
        paths = self.config.get('paths', {})
        
        envs = []
        params_str = []
        # 遍历所有 key value
        for key, value in paths.items():
            if key.lower() == "no_create_attention_mask_in_dataloader":
                params_str.append(f'    --{key.replace("_", "-")} \\')
                continue
            # 如果 key 只包含大写字母和下划线, 添加为环境变量
            if key.isupper() and all(c.isupper() or c == '_' for c in key):
                envs.append(f"{key}={value}")
            else: 
                params_str.append(f'    --{key.replace("_", "-")} {value} \\')
        
        # 添加环境变量
        for env in envs:
            self.bash_lines.append(f"{env}")
        self.bash_lines.append("")

        # 添加路径参数
        self.bash_lines.append('PATH_ARGS="\\')
        for param in params_str:
            self.bash_lines.append(param)
        self.bash_lines.append('"')
        self.bash_lines.append("")

    def _add_model_params(self) -> None:
        """添加模型参数"""
        model = self.config.get('model', {})
        
        # 基础参数
        self.bash_lines.append('GPT_ARGS="\\')
        # 遍历所有 key value
        for key, value in model.items():
            # 如果只有大写字母和下划线, 则跳过
            if key.isupper() and all(c.isupper() or c == '_' for c in key):
                continue
            param_key = key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    self.bash_lines.append(f'    --{param_key} \\')
            else:
                self.bash_lines.append(f'    --{param_key} {value} \\')

        # self.bash_lines.append(f'    --spec {model.get("spec", "mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec")} \\')
        # self.bash_lines.append(f'    --num-layers {model.get("num_layers", 27)} \\')
        # self.bash_lines.append(f'    --hidden-size {model.get("hidden_size", 2048)} \\')
        # self.bash_lines.append(f'    --ffn-hidden-size {model.get("ffn_hidden_size", 10944)} \\')
        # self.bash_lines.append(f'    --num-attention-heads {model.get("num_attention_heads", 16)} \\')
        # self.bash_lines.append(f'    --vocab-size {model.get("vocab_size", 102400)} \\')
        # self.bash_lines.append(f'    --padded-vocab-size {model.get("padded_vocab_size", 102400)} \\')
        # self.bash_lines.append(f'    --make-vocab-size-divisible-by {model.get("make_vocab_size_divisible_by", 1)} \\')
        # self.bash_lines.append(f'    --seq-length {model.get("seq_length", 4096)} \\')
        # self.bash_lines.append(f'    --max-position-embeddings {model.get("max_position_embeddings", 163840)} \\')
        # self.bash_lines.append(f'    --position-embedding-type {model.get("position_embedding_type", "rope")} \\')
        # self.bash_lines.append(f'    --rotary-base {model.get("rotary_base", 10000)} \\')
        # self.bash_lines.append(f'    --normalization {model.get("normalization", "RMSNorm")} \\')
        # self.bash_lines.append(f'    --norm-epsilon {model.get("norm_epsilon", 1e-6)} \\')
        # self.bash_lines.append(f'    --init-method-std {model.get("init_method_std", 0.02)} \\')
        # self.bash_lines.append(f'    --attention-dropout {model.get("attention_dropout", 0.0)} \\')
        # self.bash_lines.append(f'    --hidden-dropout {model.get("hidden_dropout", 0.0)} \\')
        # self.bash_lines.append(f'    --tokenizer-type {model.get("tokenizer_type", "GPT2BPETokenizer")} \\')
        
        # # 布尔参数
        # if model.get('disable_bias_linear', True):
        #     self.bash_lines.append('    --disable-bias-linear \\')
        # if model.get('untie_embeddings_and_output_weights', True):
        #     self.bash_lines.append('    --untie-embeddings-and-output-weights \\')
        # if model.get('swiglu', True):
        #     self.bash_lines.append('    --swiglu \\')
        # if model.get('use_flash_attn', True):
        #     self.bash_lines.append('    --use-flash-attn \\')
        # if model.get('use_mcore_models', True):
        #     self.bash_lines.append('    --use-mcore-models \\')
        # if model.get('use_fused_rotary_pos_emb', True):
        #     self.bash_lines.append('    --use-fused-rotary-pos-emb \\')
        # if model.get('use_rotary_position_embeddings', True):
        #     self.bash_lines.append('    --use-rotary-position-embeddings \\')
        # if model.get('use_fused_swiglu', True):
        #     self.bash_lines.append('    --use-fused-swiglu \\')
        # if model.get('use_fused_rmsnorm', True):
        #     self.bash_lines.append('    --use-fused-rmsnorm \\')
        # if model.get('no_masked_softmax_fusion', True):
        #     self.bash_lines.append('    --no-masked-softmax-fusion \\')
        # if model.get('attention_softmax_in_fp32', True):
        #     self.bash_lines.append('    --attention-softmax-in-fp32 \\')
        # if model.get('no_gradient_accumulation_fusion', True):
        #     self.bash_lines.append('    --no-gradient-accumulation-fusion \\')
        # if model.get('reuse_fp32_param', True):
        #     self.bash_lines.append('    --reuse-fp32-param \\')
        
        # 获取 training 参数
        training_params = self._get_training_params()
        parallel_params = self._get_parallel_params()
        
        for param in training_params:
            self.bash_lines.append(param)
        for param in parallel_params:
            self.bash_lines.append(param)

        self.bash_lines.append('"')
        self.bash_lines.append("")
    
    def _add_mla_params(self) -> None:
        """添加 MLA 参数"""
        mla = self.config.get('mla', {})
        
        self.bash_lines.append('MLA_ARGS="\\')
        for key, value in mla.items():
            # 跳过大写字母和下划线的 key
            if key.isupper() and all(c.isupper() or c == '_' for c in key):
                continue
            param_key = key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    self.bash_lines.append(f'    --{param_key} \\')
            else:
                self.bash_lines.append(f'    --{param_key} {value} \\')
        self.bash_lines.append('"')
        self.bash_lines.append("")
    
    def _add_moe_params(self) -> None:
        """添加 MOE 参数"""
        moe = self.config.get('moe', {})
        if not self._has_moe():
            self.bash_lines.append('MOE_ARGS=""')
            self.bash_lines.append("")
            return

        # 遍历所有 key value
        self.bash_lines.append('MOE_ARGS="\\')
        for key, value in moe.items():
            # 跳过大写字母和下划线的 key
            if key.isupper() and all(c.isupper() or c == '_' for c in key):
                continue
            if value in (None, "", "None", "null"):
                continue
            param_key = key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    self.bash_lines.append(f'    --{param_key} \\')
            else:
                self.bash_lines.append(f'    --{param_key} {value} \\')
        self.bash_lines.append('"')
        self.bash_lines.append("")
        
        # self.bash_lines.append('MOE_ARGS="\\')
        # if moe.get('moe_grouped_gemm', True):
        #     self.bash_lines.append('    --moe-grouped-gemm \\')
        # if moe.get('moe_permutation_async_comm', True):
        #     self.bash_lines.append('    --moe-permutation-async-comm \\')
        # self.bash_lines.append(f'    --moe-token-dispatcher-type {moe.get("moe_token_dispatcher_type", "alltoall")} \\')
        # if moe.get('use_fused_moe_token_permute_and_unpermute', True):
        #     self.bash_lines.append('    --use-fused-moe-token-permute-and-unpermute \\')
        # self.bash_lines.append(f'    --first-k-dense-replace {moe.get("first_k_dense_replace", 1)} \\')
        # self.bash_lines.append(f'    --moe-layer-freq {moe.get("moe_layer_freq", 1)} \\')
        # self.bash_lines.append(f'    --n-shared-experts {moe.get("n_shared_experts", 2)} \\')
        # self.bash_lines.append(f'    --num-experts {moe.get("num_experts", 64)} \\')
        # self.bash_lines.append(f'    --moe-router-topk {moe.get("moe_router_topk", 6)} \\')
        # self.bash_lines.append(f'    --moe-intermediate-size {moe.get("moe_intermediate_size", 1408)} \\')
        # self.bash_lines.append(f'    --moe-router-load-balancing-type {moe.get("moe_router_load_balancing_type", "pai_megatron_aux_loss")} \\')
        # # 如果 topk_group 存在且大于 1, 则添加 topk_group 参数
        # if moe.get("topk_group", 1) > 1:
        #     self.bash_lines.append(f'    --topk-group {moe["topk_group"]} \\')
        # self.bash_lines.append(f'    --moe-aux-loss-coeff {moe.get("moe_aux_loss_coeff", 0.01)} \\')
        # self.bash_lines.append(f'    --routed-scaling-factor {moe.get("routed_scaling_factor", 1.0)} \\')
        # if moe.get('seq_aux', True):
        #     self.bash_lines.append('    --seq-aux \\')
        # self.bash_lines.append('"')
        # self.bash_lines.append("")
    
    def _add_rope_params(self) -> None:
        """添加 ROPE 参数"""
        
        # 遍历 key value
        rope = dict(self.config.get('rope', {}) or {})
        rope_type = str(rope.get('rope_scaling_type', '') or '').strip().lower()
        if rope_type in {'', 'none', 'null'}:
            rope.pop('rope_scaling_type', None)
            rope.pop('rope_scaling_factor', None)
            rope.pop('rope_scaling_original_max_position_embeddings', None)
        elif rope_type == 'longrope':
            factor_count = self._infer_rope_factor_count()
            if factor_count is not None and factor_count > 0:
                for key in ('long_factor', 'short_factor'):
                    if key in rope:
                        values = self._split_csv_values(rope[key])
                        if values:
                            rope[key] = ','.join(values[:factor_count])

        self.bash_lines.append('ROPE_ARGS="\\')
        for key, value in rope.items():
            if key.isupper() and all(c.isupper() or c == '_' for c in key):
                continue
            param_key = key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    self.bash_lines.append(f'    --{param_key} \\')
            else:
                self.bash_lines.append(f'    --{param_key} {value} \\')
        self.bash_lines.append('"')
        self.bash_lines.append("")
        
        # rope = self.config.get('rope', {})
        
        # self.bash_lines.append('ROPE_ARGS="\\')
        # self.bash_lines.append(f'    --rope-scaling-type {rope.get("rope_scaling_type", "yarn")} \\')
        # self.bash_lines.append(f'    --rope-scaling-factor {rope.get("rope_scaling_factor", 40)} \\')
        # self.bash_lines.append(f'    --rope-scaling-beta-fast {rope.get("rope_scaling_beta_fast", 32)} \\')
        # self.bash_lines.append(f'    --rope-scaling-beta-slow {rope.get("rope_scaling_beta_slow", 1)} \\')
        # self.bash_lines.append(f'    --rope-scaling-mscale {rope.get("rope_scaling_mscale", 0.707)} \\')
        # self.bash_lines.append(f'    --rope-scaling-mscale-all-dim {rope.get("rope_scaling_mscale_all_dim", 0.707)} \\')
        # self.bash_lines.append(f'    --rope-scaling-original-max-position-embeddings {rope.get("rope_scaling_original_max_position_embeddings", 4096)} \\')
        # self.bash_lines.append('"')
        # self.bash_lines.append("")
    
    def _get_training_params(self) -> list[str]:
        """添加训练参数"""
        training = self.config.get('training', {})

        training_params_str = []
        # 遍历训练参数
        for key, value in training.items():
            # 跳过占位/聚合变量，避免生成无效参数
            if key.isupper() and all(c.isupper() or c == '_' for c in key):
                continue
            if 'data' in key and 'path' in key:
                continue  # 跳过 data_path,放在 data 中
            param_key = key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    training_params_str.append(f'    --{param_key} \\')
            else:
                training_params_str.append(f'    --{param_key} {value} \\')
        
        # 插入训练参数到 GPT_ARGS
        return training_params_str
                    
        
        # # 添加到 GPT_ARGS
        # gpt_args_start = self.bash_lines.index('GPT_ARGS="\\') + 1
        
        # # 插入训练参数到 GPT_ARGS
        # training_params = [
        #     f'    --micro-batch-size {training.get("micro_batch_size", 1)} \\',
        #     f'    --global-batch-size {training.get("global_batch_size", 128)} \\',
        #     f'    --train-iters {training.get("train_iters", 2000)} \\',
        #     f'    --lr {training.get("lr", 2.0e-6)} \\',
        #     f'    --lr-decay-style {training.get("lr_decay_style", "cosine")} \\',
        #     f'    --lr-decay-iters {training.get("lr_decay_iters", 2000)} \\',
        #     f'    --min-lr {training.get("min_lr", 1.0e-8)} \\',
        #     f'    --weight-decay {training.get("weight_decay", 0.1)} \\',
        #     f'    --lr-warmup-iters {training.get("lr_warmup_iters", 100)} \\',
        #     f'    --clip-grad {training.get("clip_grad", 1.0)} \\',
        #     f'    --adam-beta1 {training.get("adam_beta1", 0.9)} \\',
        #     f'    --adam-beta2 {training.get("adam_beta2", 0.95)} \\',
        #     f'    --initial-loss-scale {training.get("initial_loss_scale", 65536)} \\',
        # ]
        
        # if training.get('bf16', True):
        #     training_params.append('    --bf16 \\')
        # if training.get('finetune', True):
        #     training_params.append('    --finetune \\')
        # if training.get('use_distributed_optimizer', True):
        #     training_params.append('    --use-distributed-optimizer \\')
        # if training.get('no_load_optim', True):
        #     gpt_args_end = self.bash_lines.index('"')
        
        # # 插入训练参数
        # for i, param in enumerate(training_params):
        #     self.bash_lines.insert(gpt_args_end + i, param)
    
    def _add_data_params(self) -> None:
        """添加数据参数"""
        data = self.config.get('data', {})
        path = self.config.get('paths', {})
        
        self.bash_lines.append('DATA_ARGS="\\')
        self.bash_lines.append(f'    --data-path {path.get("data_path", "gpt2_text_document")} \\')
        self.bash_lines.append(f'    --split {data.get("split", "99,1,0")} \\')
        self.bash_lines.append('"')
        self.bash_lines.append("")
    
    def _add_output_params(self) -> None:
        """添加输出参数"""
        output = self.config.get('output', {})
        
        self.bash_lines.append('OUTPUT_ARGS="\\')
        self.bash_lines.append(f'    --log-interval {output.get("log_interval", 1)} \\')
        self.bash_lines.append(f'    --save-interval {output.get("save_interval", 1000)} \\')
        self.bash_lines.append(f'    --eval-interval {output.get("eval_interval", 10000)} \\')
        self.bash_lines.append(f'    --eval-iters {output.get("eval_iters", 10)} \\')
        if output.get('no_save_optim', True):
            self.bash_lines.append('    --no-save-optim \\')
        if output.get('no_save_rng', True):
            self.bash_lines.append('    --no-save-rng \\')
        self.bash_lines.append('"')
        self.bash_lines.append("")
    
    def _build_final_command(self) -> None:
        """构建分布式参数变量"""
        self.bash_lines.append('DISTRIBUTED_ARGS="\\')
        self.bash_lines.append('    --nproc_per_node $NPUS_PER_NODE \\')
        self.bash_lines.append('    --nnodes $NNODES \\')
        self.bash_lines.append('    --node_rank $NODE_RANK \\')
        self.bash_lines.append('    --master_addr $MASTER_ADDR \\')
        self.bash_lines.append('    --master_port $MASTER_PORT \\')
        self.bash_lines.append('"')
        self.bash_lines.append("")

        # 添加最终的 torchrun 命令
        self.bash_lines.append('torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \\')
        self.bash_lines.append('  $MOE_ARGS \\')
        self.bash_lines.append('  $MLA_ARGS \\')
        self.bash_lines.append('  $GPT_ARGS \\')
        self.bash_lines.append('  $ROPE_ARGS \\')
        self.bash_lines.append('  $DATA_ARGS \\')
        self.bash_lines.append('  $OUTPUT_ARGS \\')
        self.bash_lines.append('  $PATH_ARGS \\')
        self.bash_lines.append('  --distributed-backend nccl \\')
        self.bash_lines.append('  | tee logs/.log')

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def save_bash_script(script: str, file_path: str) -> None:
    """保存 Bash 脚本"""
    script = strip_pta_unsupported_script_flags(script)
    with open(file_path, 'w') as f:
        f.write(script)
    # 添加执行权限
    os.chmod(file_path, 0o755)

def main():
    parser = argparse.ArgumentParser(description="YAML 到 Bash 转换器")
    parser.add_argument("input_yaml", help="输入 YAML 配置文件路径")
    parser.add_argument("--output", "-o", default="run_training.sh", help="输出 Bash 脚本路径")
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = load_yaml_config(args.input_yaml)
    except Exception as e:
        print(f"错误: 无法加载配置文件 {args.input_yaml}: {e}")
        return
    
    # 转换配置
    converter = YamlToBashConverter(config)
    bash_script = converter.convert()
    
    # 保存脚本
    try:
        save_bash_script(bash_script, args.output)
        print(f"Bash 脚本已保存到: {args.output}")
    except Exception as e:
        print(f"错误: 无法保存脚本 {args.output}: {e}")

if __name__ == "__main__":
    main()
