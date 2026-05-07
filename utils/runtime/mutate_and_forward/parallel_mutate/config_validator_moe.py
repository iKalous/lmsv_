#!/usr/bin/env python3
"""
增强版 Megatron-LM YAML 配置校验修正器
检查并行策略（TP、PP、EP、CP）与其他参数的兼容性，并自动修正冲突
支持 MOE、MLA、MTP 等高级功能的参数校验
"""

import yaml
import math
import argparse
import random
import os
from typing import Dict, Any, Tuple, List, Set

class EnhancedMegatronConfigValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.issues = []
        self.fixes_applied = 0
        self.warnings = []
        self.seq_cap_enabled = self._env_bool("LMSV_MOE_SEQ_CAP_ENABLE", True)
        self.max_seq_length_cap = self._env_int("LMSV_MOE_MAX_SEQ_LENGTH", 4096)
        self.flash_attention_seq_limit = self._env_int("LMSV_FLASH_ATTN_MAX_SEQ_LENGTH", 2048)

    @staticmethod
    def _to_bool(value: Any) -> bool:
        """Parse bool-like values from yaml/json robustly."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "y"}
        return bool(value)

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = int(raw)
            return value if value > 0 else default
        except ValueError:
            return default
    
    def validate_and_fix(self) -> Tuple[Dict[str, Any], List[str], List[str], int]:
        """
        验证并修正配置
        """
        # 提取关键参数
        parallel = self.config.get('parallel', {})
        model_cfg = self.config.get('model', {})
        training = self.config.get('training', {})

        tp = parallel.get('tensor_model_parallel_size', 1)
        pp = parallel.get('pipeline_model_parallel_size', 1)
        ep = parallel.get('expert_model_parallel_size', 1)
        cp = parallel.get('context_parallel_size', 1)
        vpp = parallel.get('num_layers_per_virtual_pipeline_stage', 1)
        
        hidden_size = model_cfg.get('hidden_size', 0)
        num_attention_heads = model_cfg.get('num_attention_heads', 0)
        ffn_hidden_size = model_cfg.get('ffn_hidden_size', 0)
        num_layers = model_cfg.get('num_layers', 0)
        global_batch_size = training.get('global_batch_size', 0)
        micro_batch_size = training.get('micro_batch_size', 0)

        # world_size 固定为8（如需读取真实值，可从distributed配置中解析）
        world_size = 8
        tp, pp, ep, cp = self._check_parallel_constraints(tp, pp, ep, cp, world_size)
        dp = world_size // (tp * pp * ep * cp)
        
        # 核心结构/并行维度检查
        self._check_ffn_hidden_size_limit(hidden_size, ffn_hidden_size)
        self._check_hidden_size(tp, hidden_size)
        self._check_attention_heads(tp, cp, num_attention_heads)
        self._check_ffn_hidden_size(tp, ffn_hidden_size)
        self._check_num_layers(pp, num_layers, vpp)
        self._check_global_batch_size(dp, global_batch_size, micro_batch_size)

        # 新增：交错调度与 PP/VPP 约束
        self._check_pipeline_interleaving(tp, pp, ep, cp, vpp, num_layers, world_size)

        # ALiBi 模型在当前 PTA/MindSpeed-LLM 组合下不走 CP 变异，
        # 并且显式传入 CP=1 可能触发上游字符串类型分支。
        self._check_alibi_constraints()

        # 先处理 group-limited 路由的必填参数 (moe_router_group_topk)
        self._ensure_group_limited_router_groups(ep)

        # 兼容旧链路: 部分 MoE 参数可能仍落在 model 段，先归一化再校验。
        self._normalize_moe_section()

        # MOE 相关
        if self.config.get('moe', {}) != {}:
            self._check_moe_parameters(tp, ep)

        # 新增：学习率/预热/衰减约束
        self._check_lr_schedule()

        # 新增：FlashAttention 序列长度约束（NPU/Ascend）
        self._check_flash_attention_constraints()

        # 以前未调用的检查补齐
        self._check_mla_parameters(tp, hidden_size, num_attention_heads)
        self._check_rope_parameters()
        self._check_sequence_length()
        self._check_vocab_size()
        self._check_recompute_constraints()
        
        # 变异重计算/优化器并行（保留原逻辑）
        mutate_space = {}
        self._mutate_recompute(mutate_space)
        # _mutate_recompute 会重写 training 段，这里再兜底一次，避免重新引入非法组合。
        self._check_recompute_constraints()

        mutate_space_optimizer = {
            "use_distributed_optimizer": [True, False],
            "use_megatron_fsdp": [False],
            "use_torch_fsdp2": [False],
            "overlap_param_gather": [True, False],
            "overlap_param_gather_with_optimizer_step": [False],
            "overlap_grad_reduce": [True, False],
            "fp8_param_gather": [False],
            "enable_gloo_process_groups": [False],
            "use_dist_ckpt": [False],
            "ckpt_fully_parallel_save": [True, False],
            "dist_ckpt_optim_fully_reshardable": [False],
            "distrib_optim_fully_reshardable_mem_efficient": [False]
        }
        self._mutate_distributed_optimizer(mutate_space_optimizer)
        
        return self.config, self.issues, self.warnings, self.fixes_applied

    def _check_alibi_constraints(self) -> None:
        """ALiBi 模型约束：禁用 CP，并关闭显式 flash attention。"""
        model_cfg = self.config.setdefault('model', {})
        parallel_cfg = self.config.setdefault('parallel', {})

        position_type = str(model_cfg.get('position_embedding_type', '') or '').strip().lower()
        if position_type != 'alibi':
            return

        raw_cp = parallel_cfg.get('context_parallel_size', 1)
        try:
            cp = int(raw_cp)
        except (TypeError, ValueError):
            cp = 1

        if raw_cp != cp:
            self._apply_fix(
                "parallel.context_parallel_size",
                raw_cp,
                cp,
                "ALiBi 模型的 context_parallel_size 必须为整数"
            )

        if cp != 1:
            self._apply_fix(
                "parallel.context_parallel_size",
                cp,
                1,
                "ALiBi 位置编码暂不支持 context parallel，固定为 1"
            )
        else:
            parallel_cfg["context_parallel_size"] = 1

        for key in ("use_flash_attn", "use_flash_attention"):
            raw_flash = model_cfg.get(key)
            if raw_flash in (None, False):
                continue
            self._apply_fix(
                f"model.{key}",
                raw_flash,
                False,
                "ALiBi 模型默认关闭 flash attention"
            )

    def _check_ffn_hidden_size_limit(self, hidden_size: int, ffn_hidden_size: int) -> None:
        """检查 FFN 中间维度是否超过隐藏层维度的 6 倍"""
        max_ffn_hidden_size = hidden_size * 6
        if ffn_hidden_size > max_ffn_hidden_size:
            self._apply_fix(
                "model.ffn_hidden_size",
                ffn_hidden_size,
                max_ffn_hidden_size,
                f"FFN 中间维度 {ffn_hidden_size} 超过隐藏层维度 {hidden_size} 的 6 倍"
            )

    def _check_hidden_size(self, tp: int, hidden_size: int) -> None:
        """检查隐藏层维度是否可被 TP 整除"""
        if hidden_size % tp != 0:
            new_hidden_size = self._find_nearest_divisible(hidden_size, tp)
            self._apply_fix(
                "model.hidden_size",
                hidden_size,
                new_hidden_size,
                f"隐藏层维度 {hidden_size} 不能被 TP={tp} 整除"
            )
    
    def _check_attention_heads(self, tp: int, cp: int, num_attention_heads: int) -> None:
        """检查注意力头数是否可被 TP/CP 切分。

        Ulysses context parallel 会要求:
        num_attention_heads % (context_parallel_size * tensor_model_parallel_size) == 0
        这里统一按 TP*CP 做兜底，避免整网变异产生训练期非法组合。
        """
        divisor = max(1, tp * cp)
        if num_attention_heads % divisor != 0:
            new_num_heads = self._find_nearest_divisible(num_attention_heads, divisor)
            self._apply_fix(
                "model.num_attention_heads",
                num_attention_heads,
                new_num_heads,
                f"注意力头数 {num_attention_heads} 不能被 TP*CP={tp}*{cp}={divisor} 整除"
            )
    
    def _check_ffn_hidden_size(self, tp: int, ffn_hidden_size: int) -> None:
        """检查 FFN 中间维度是否可被 TP 整除"""
        if ffn_hidden_size % tp != 0:
            new_ffn_size = self._find_nearest_divisible(ffn_hidden_size, tp)
            self._apply_fix(
                "model.ffn_hidden_size",
                ffn_hidden_size,
                new_ffn_size,
                f"FFN 中间维度 {ffn_hidden_size} 不能被 TP={tp} 整除"
            )

        # Swiglu 会将 FFN 维度对半切分；TP 切分后每卡维度仍需为偶数。
        # 约束等价于: ffn_hidden_size 必须能被 (2 * TP) 整除。
        model_cfg = self.config.get('model', {})
        hidden_act = str(model_cfg.get('hidden_act', '') or '').lower()
        use_swiglu = bool(model_cfg.get('swiglu', False)) or hidden_act == 'swiglu'
        required_divisor = max(1, 2 * tp)
        if use_swiglu and ffn_hidden_size % required_divisor != 0:
            new_ffn_size = self._find_nearest_divisible(ffn_hidden_size, required_divisor)
            self._apply_fix(
                "model.ffn_hidden_size",
                ffn_hidden_size,
                new_ffn_size,
                f"Swiglu 模式下 FFN 中间维度 {ffn_hidden_size} 不能被 2*TP={required_divisor} 整除"
            )
    
    def _check_num_layers(self, pp: int, num_layers: int, vpp: int) -> None:
        """检查层数是否可被 PP 整除，并考虑虚拟流水线并行"""
        if num_layers % pp != 0:
            new_num_layers = self._find_nearest_divisible(num_layers, pp)
            self._apply_fix(
                "model.num_layers",
                num_layers,
                new_num_layers,
                f"总层数 {num_layers} 不能被 PP={pp} 整除"
            )
        
        # 检查虚拟流水线并行
        if vpp > 1 and num_layers % (pp * vpp) != 0:
            new_num_layers = self._find_nearest_divisible(num_layers, pp * vpp)
            self._apply_fix(
                "model.num_layers",
                num_layers,
                new_num_layers,
                f"虚拟流水线并行: 总层数 {num_layers} 不能被 (PP={pp} * VPP={vpp}) 整除"
            )
    
    def _check_global_batch_size(self, dp: int, global_batch_size: int, micro_batch_size: int) -> None:
        """检查全局批次大小是否可被 DP * 微批次大小整除"""
        if dp <= 0:
            self.issues.append("错误: 数据并行大小计算为 0 或负数")
            return

        # 兜底：兼容字符串/浮点等输入，避免类型异常导致校验失效。
        try:
            global_batch_size = int(global_batch_size)
        except (TypeError, ValueError):
            self._apply_fix(
                "training.global_batch_size",
                global_batch_size,
                max(1, dp),
                f"全局批次大小 {global_batch_size} 非法，自动修正为可运行值"
            )
            global_batch_size = max(1, dp)

        try:
            micro_batch_size = int(micro_batch_size)
        except (TypeError, ValueError):
            self._apply_fix(
                "training.micro_batch_size",
                micro_batch_size,
                1,
                f"微批次大小 {micro_batch_size} 非法，自动修正为 1"
            )
            micro_batch_size = 1

        max_global_bs_exclusive = 64
        try:
            max_global_bs_exclusive = max(
                2,
                int(os.getenv("LMSV_MAX_GLOBAL_BATCH_SIZE_EXCLUSIVE", "64"))
            )
        except ValueError:
            max_global_bs_exclusive = 64
        max_global_bs = max_global_bs_exclusive

        if micro_batch_size <= 0:
            self._apply_fix(
                "training.micro_batch_size",
                micro_batch_size,
                1,
                f"微批次大小 {micro_batch_size} 非法，自动修正为 1"
            )
            micro_batch_size = 1

        divisor = micro_batch_size * dp

        # 若 divisor 已超过上限，先下调 micro_batch_size 以保证可满足上限约束。
        if divisor > max_global_bs:
            new_micro_batch_size = max(1, max_global_bs // dp)
            if new_micro_batch_size != micro_batch_size:
                self._apply_fix(
                    "training.micro_batch_size",
                    micro_batch_size,
                    new_micro_batch_size,
                    f"DP={dp} 下 micro_batch_size={micro_batch_size} 导致最小全局批次 {divisor} 超过上限 {max_global_bs}"
                )
                micro_batch_size = new_micro_batch_size
                divisor = micro_batch_size * dp

        if divisor > max_global_bs:
            # 上限约束与并行约束冲突时，优先保证可运行性：
            # global_batch_size 至少应为 divisor（保证整除）。
            self.warnings.append(
                f"无法严格限制 global_batch_size<{max_global_bs_exclusive}: DP={dp}, micro_batch_size={micro_batch_size}, 最小可行值={divisor}；将优先保证整除与可运行"
            )
            if global_batch_size != divisor:
                self._apply_fix(
                    "training.global_batch_size",
                    global_batch_size,
                    divisor,
                    f"在当前 DP/MBS 约束下，将全局批次大小调整为最小可行值 {divisor}"
                )
            return

        if global_batch_size > max_global_bs:
            capped_global_bs = (max_global_bs // divisor) * divisor
            capped_global_bs = max(divisor, capped_global_bs)
            self._apply_fix(
                "training.global_batch_size",
                global_batch_size,
                capped_global_bs,
                f"全局批次大小 {global_batch_size} 超过上限 {max_global_bs}"
            )
            global_batch_size = capped_global_bs
            
        if micro_batch_size > 0 and global_batch_size % (micro_batch_size * dp) != 0:
            lower = (global_batch_size // divisor) * divisor
            if lower <= 0:
                lower = divisor
            new_global_batch_size = min(lower, max_global_bs)
            self._apply_fix(
                "training.global_batch_size",
                global_batch_size,
                new_global_batch_size,
                f"全局批次大小 {global_batch_size} 不能被 (DP={dp} * 微批次大小={micro_batch_size}) 整除"
            )

    def _check_parallel_constraints(self, tp: int, pp: int, ep: int, cp: int, world_size: int) -> Tuple[int, int, int, int]:
        """兜底修正: CP>1 时要求 TP>=2，且总并行规模不超过 world_size。"""
        parallel = self.config.setdefault('parallel', {})

        def normalize_positive(value: Any, key_path: str) -> int:
            if not isinstance(value, int) or value < 1:
                self._apply_fix(key_path, value, 1, f"{key_path} 必须为正整数")
                return 1
            return value

        tp = normalize_positive(tp, "parallel.tensor_model_parallel_size")
        pp = normalize_positive(pp, "parallel.pipeline_model_parallel_size")
        ep = normalize_positive(ep, "parallel.expert_model_parallel_size")
        cp = normalize_positive(cp, "parallel.context_parallel_size")
        sp = self._to_bool(parallel.get("sequence_parallel", False))

        if cp > 1 and tp < 2:
            self._apply_fix(
                "parallel.context_parallel_size",
                cp,
                1,
                "CP>1 时 TP 必须 >=2，已禁用 CP"
            )
            cp = 1

        product = tp * pp * ep * cp
        if product > world_size:
            self._apply_fix(
                "parallel.context_parallel_size",
                cp,
                1,
                f"并行总规模 {product} 超过 world_size={world_size}，禁用 CP"
            )
            cp = 1

        if tp < 2 and sp:
            self._apply_fix(
                "parallel.sequence_parallel",
                sp,
                False,
                "TP=1 时不能启用 sequence_parallel，自动关闭"
            )

        parallel["tensor_model_parallel_size"] = tp
        parallel["pipeline_model_parallel_size"] = pp
        parallel["expert_model_parallel_size"] = ep
        parallel["context_parallel_size"] = cp
        return tp, pp, ep, cp

    @staticmethod
    def _get_moe_value(moe_config: Dict[str, Any], key: str, default: Any = None) -> Any:
        """兼容 snake_case 与 kebab-case 的 MOE 参数读取。"""
        if key in moe_config:
            return moe_config[key]
        dashed_key = key.replace('_', '-')
        if dashed_key in moe_config:
            return moe_config[dashed_key]
        return default

    def _normalize_moe_section(self) -> None:
        """将误放在 model 段的 MoE 参数回填到 moe 段。"""
        model_cfg = self.config.get('model', {})
        moe_cfg = self.config.setdefault('moe', {})
        moe_keys = (
            'num_experts',
            'num_moe_experts',
            'moe_grouped_gemm',
            'moe_permutation_async_comm',
            'moe_token_dispatcher_type',
            'use_fused_moe_token_permute_and_unpermute',
            'first_k_dense_replace',
            'moe_layer_freq',
            'n_shared_experts',
            'moe_router_topk',
            'moe_router_pre_softmax',
            'moe_router_group_topk',
            'moe_intermediate_size',
            'moe_router_load_balancing_type',
            'moe_router_num_groups',
            'topk_group',
            'moe_aux_loss_coeff',
            'routed_scaling_factor',
            'seq_aux',
        )
        for key in moe_keys:
            value = self._get_moe_value(model_cfg, key, default=None)
            if value is None:
                continue
            target_key = 'num_experts' if key == 'num_moe_experts' else key
            if self._get_moe_value(moe_cfg, target_key, default=None) is None:
                moe_cfg[target_key] = value

        # 兼容旧字段 moe_router_group_topk 与新字段 topk_group。
        alias_value = self._get_moe_value(moe_cfg, 'moe_router_group_topk', default=None)
        canonical_value = self._get_moe_value(moe_cfg, 'topk_group', default=None)
        if canonical_value is None and alias_value is not None:
            moe_cfg['topk_group'] = alias_value
        elif alias_value is None and canonical_value is not None:
            moe_cfg['moe_router_group_topk'] = canonical_value
    
    def _check_moe_parameters(self, tp: int, ep: int) -> None:
        """检查 MOE 相关参数的兼容性 (集成 topk_group 下限与 EP 约束)"""
        moe_config = self.config.get('moe', {})
        lb_type = moe_config.get('moe_router_load_balancing_type', "")
        model_cfg = self.config.get('model', {})
        parallel_cfg = self.config.setdefault('parallel', {})
        moe_router_group_topk = self._get_moe_value(moe_config, 'topk_group', default=None)
        # 先做基础整除/尺寸检查
        num_experts = moe_config.get('num_experts', 0)
        if num_experts > 0 and ep > 0 and num_experts % ep != 0:
            new_num_experts = self._find_nearest_divisible(num_experts, ep)
            self._apply_fix("moe.num_experts", num_experts, new_num_experts,
                            f"专家数 {num_experts} 不能被 EP={ep} 整除")
        
        moe_intermediate_size = moe_config.get('moe_intermediate_size', 0)
        if moe_intermediate_size > 0 and moe_intermediate_size % tp != 0:
            new_moe_size = self._find_nearest_divisible(moe_intermediate_size, tp)
            self._apply_fix("moe.moe_intermediate_size", moe_intermediate_size, new_moe_size,
                            f"MOE 中间维度 {moe_intermediate_size} 不能被 TP={tp} 整除")
        
        # 路由相关
        moe_router_topk = self._get_moe_value(moe_config, 'moe_router_topk', default=0)
        if moe_router_topk == 0:
            self._apply_fix("moe.moe_router_topk", 0, 1,
                            "MOE 路由参数: moe_router_topk 不可为 0，修复为 1")
            moe_router_topk = 1
        if num_experts > 0 and moe_router_topk > num_experts:
            self._apply_fix(
                "moe.moe_router_topk",
                moe_router_topk,
                num_experts,
                f"MOE 路由参数: moe_router_topk={moe_router_topk} 不能超过专家数 num_experts={num_experts}"
            )
            moe_router_topk = num_experts
        moe_router_pre_softmax = bool(
            self._get_moe_value(moe_config, 'moe_router_pre_softmax', default=False)
        )
        if moe_router_topk == 1 and not moe_router_pre_softmax:
            self._apply_fix(
                "moe.moe_router_pre_softmax",
                self._get_moe_value(moe_config, 'moe_router_pre_softmax', default=None),
                True,
                "MOE 路由参数: 当 topk=1 时必须启用 moe_router_pre_softmax"
            )
        
        
        # MoE 不支持 add_bias_linear
        if model_cfg.get('add_bias_linear', None) is True:
            self._apply_fix(
                "model.add_bias_linear",
                True,
                False,
                "MoE 不支持 add_bias_linear，自动关闭"
            )
        if model_cfg.get('disable_bias_linear', None) is False:
            self._apply_fix(
                "model.disable_bias_linear",
                False,
                True,
                "MoE 需要关闭 bias linear，自动设置 disable_bias_linear=True"
            )

        # Megatron 训练期约束: 启用 MoE 且 TP>1 时必须同时启用 sequence_parallel。
        sequence_parallel = self._to_bool(parallel_cfg.get("sequence_parallel", False))
        if num_experts > 0 and tp > 1 and not sequence_parallel:
            self._apply_fix(
                "parallel.sequence_parallel",
                sequence_parallel,
                True,
                f"MoE 训练在 TP={tp} 时必须启用 sequence_parallel，自动开启"
            )

        # group_limited_greedy 额外逻辑: 需要 ep>moe_router_group_topk>=1
        if lb_type == "group_limited_greedy":
            invalid_group_topk = (
                not isinstance(moe_router_group_topk, int) or
                moe_router_group_topk < 1 or
                moe_router_group_topk >= ep
            )
            if ep <= 1 or invalid_group_topk:
                # 无法满足条件 -> 降级 load balancing 类型
                new_type = "aux_loss"
                self._apply_fix("moe.moe_router_load_balancing_type", lb_type, new_type,
                                f"group_limited_greedy 需要 EP>1 且 moe_router_group_topk<EP 不满足 (EP={ep}, moe_router_group_topk={moe_router_group_topk})，降级为 {new_type}")
        # 结束

    def _ensure_group_limited_router_groups(self, ep: int) -> None:
        """确保 group-limited 路由下 moe_router_num_groups 被显式设置。"""
        moe_config = self.config.get('moe', {})

        # 如果没有moe配置或 moe_router_group_topk 未启用，则无需强制要求 moe_router_num_groups
        if not isinstance(moe_config, dict) or not moe_config:
            return

        group_topk = self._get_moe_value(moe_config, 'topk_group', default=None)
        if not group_topk:
            return

        moe_router_num_groups = moe_config.get('moe_router_num_groups', None)
        if moe_router_num_groups is None or not isinstance(moe_router_num_groups, int) or moe_router_num_groups < 1:
            default_groups = ep if ep > 0 else 1
            self._apply_fix(
                "moe.moe_router_num_groups",
                moe_router_num_groups,
                default_groups,
                f"group-limited 路由需要 moe_router_num_groups，自动设置为 {default_groups}"
            )


    # 新增：交错调度（interleaved）与 PP/VPP 合法性修复
    def _check_pipeline_interleaving(self, tp: int, pp: int, ep: int, cp: int, vpp: int, num_layers: int, world_size: int) -> None:
        """
        交错调度校验与自动修正逻辑（与运行期 _validate_vpp 保持一致思路）:
        1. 若 vpp<=1 或未设置 -> 视为未启用交错，直接禁用 (设为 None)。
        2. 若 vpp>1 但 pp==1 -> 尝试提升 pp=2（需满足 world_size 整除）；失败则禁用交错。
        3. 若 num_layers 不能被 pp 整除 -> 交错判定无意义，直接返回（由其他检查修复）。
        4. 计算每个物理 pipeline stage 的层数 layers_per_stage = num_layers // pp。
           - 若 vpp >= layers_per_stage：无意义，禁用交错。
           - 若 layers_per_stage % vpp != 0：降为最大合法因子；若不存在则禁用。
        """
        parallel = self.config.setdefault('parallel', {})
        pp_eff = parallel.get('pipeline_model_parallel_size', pp)
        vpp_cur = parallel.get('num_layers_per_virtual_pipeline_stage', vpp)

        # 未设置或 <=1: 禁用
        if vpp_cur is None or vpp_cur <= 1:
            if vpp_cur is not None:
                self._apply_fix(
                    "parallel.num_layers_per_virtual_pipeline_stage",
                    vpp_cur,
                    False,
                    "VPP<=1 视为未启用交错，禁用交错(设为 None)"
                )
            parallel["num_layers_per_virtual_pipeline_stage"] = False
            return

        # vpp>1 但 pp==1 -> 尝试提升 pp=2
        if pp_eff == 1:
            target_pp = 2
            if (tp * target_pp * ep * cp) > 0 and world_size % (tp * target_pp * ep * cp) == 0:
                self._apply_fix(
                    "parallel.pipeline_model_parallel_size",
                    pp_eff,
                    target_pp,
                    "交错调度需要 PP>1，提升 PP 为 2"
                )
                pp_eff = target_pp
            else:
                # 禁用交错
                self._apply_fix(
                    "parallel.num_layers_per_virtual_pipeline_stage",
                    vpp_cur,
                    False,
                    "无法提升 PP 以支持交错，禁用交错(VPP=None)"
                )
                parallel["num_layers_per_virtual_pipeline_stage"] = False
                return

        # 若 num_layers 不可被 pp_eff 整除，交错无意义（其它检查会处理）
        if num_layers <= 0 or num_layers % pp_eff != 0:
            return

        layers_per_stage = num_layers // pp_eff

        # vpp >= 每 stage 层数 -> 禁用
        if vpp_cur >= layers_per_stage:
            self._apply_fix(
                "parallel.num_layers_per_virtual_pipeline_stage",
                vpp_cur,
                False,
                f"VPP={vpp_cur} >= 每个 stage 层数 {layers_per_stage}，禁用交错(VPP=None)"
            )
            parallel["num_layers_per_virtual_pipeline_stage"] = False
            return

        # 需要整除每 stage 层数
        if layers_per_stage % vpp_cur == 0:
            return  # 合法

        # 寻找最大合法因子
        new_vpp = None
        for cand in range(min(vpp_cur, layers_per_stage - 1), 1, -1):
            if layers_per_stage % cand == 0:
                new_vpp = cand
                break

        if new_vpp is None:
            self._apply_fix(
                "parallel.num_layers_per_virtual_pipeline_stage",
                vpp_cur,
                False,
                f"VPP={vpp_cur} 不整除每 stage 层数 {layers_per_stage}，且无合法因子，禁用交错(VPP=None)"
            )
            parallel["num_layers_per_virtual_pipeline_stage"] = False
        else:
            self._apply_fix(
                "parallel.num_layers_per_virtual_pipeline_stage",
                vpp_cur,
                new_vpp,
                f"VPP={vpp_cur} 不整除每 stage 层数 {layers_per_stage}，调整为最大合法因子 {new_vpp}"
            )

    # 新增：学习率/预热/衰减关系
    def _check_lr_schedule(self) -> None:
        tr = self.config.setdefault('training', {})
        lr_decay_style = tr.get('lr_decay_style', None)
        warmup = tr.get('lr_warmup_iters', None)
        train_iters = tr.get('train_iters', None)
        decay_iters = tr.get('lr_decay_iters', None)

        # 当使用 cosine 衰减时，若未设置 lr_decay_iters，默认与 train_iters 对齐
        if lr_decay_style == 'cosine' and (decay_iters is None or decay_iters <= 0):
            if isinstance(train_iters, int) and train_iters > 0:
                self._apply_fix(
                    "training.lr_decay_iters",
                    decay_iters,
                    train_iters,
                    "使用 cosine 衰减但未设置 lr_decay_iters，修复为 train_iters"
                )
                decay_iters = train_iters
            else:
                self.warnings.append("学习率: 使用 cosine 衰减但未设置 lr_decay_iters 且 train_iters 不可用")

        # 预热步数必须小于 min(train_iters, lr_decay_iters)
        mins = [x for x in [train_iters, decay_iters] if isinstance(x, int) and x > 0]
        if warmup is not None and isinstance(warmup, int) and mins:
            min_iters = min(mins)
            if warmup >= min_iters:
                new_warmup = max(1, int(min_iters * 0.1))
                self._apply_fix(
                    "training.lr_warmup_iters",
                    warmup,
                    new_warmup,
                    f"预热步数必须小于 min(train_iters, lr_decay_iters)={min_iters}，修复为 {new_warmup}"
                )
                warmup = new_warmup

        # train_iters 必须大于预热步数
        if isinstance(train_iters, int) and isinstance(warmup, int):
            if train_iters <= warmup:
                new_train_iters = warmup + 1
                self._apply_fix(
                    "training.train_iters",
                    train_iters,
                    new_train_iters,
                    "train_iters 必须大于 lr_warmup_iters，自动加一"
                )

    # 新增：FlashAttention 序列长度约束（Ascend NPU 要求 2048）
    def _check_flash_attention_constraints(self) -> None:
        model = self.config.setdefault('model', {})
        attention_impl = model.get('attention_impl', '')
        use_flash_flag = bool(model.get('use_flash_attention', False)) or \
                         (isinstance(attention_impl, str) and 'flash' in attention_impl.lower())

        if not use_flash_flag:
            return

        seq_len = model.get('seq_length', 0)
        max_pos = model.get('max_position_embeddings', 0)
        limit = self.flash_attention_seq_limit

        # 若未设置 seq_length，则以 max_position_embeddings 视为序列上限
        effective = seq_len if seq_len > 0 else max_pos

        if effective > limit:
            # 首选将 seq_length 限制到 2048
            if seq_len and seq_len > limit:
                self._apply_fix(
                    "model.seq_length",
                    seq_len,
                    limit,
                    f"FlashAttention 限制: 序列长度需 <= {limit}，修复 seq_length"
                )
            # 同步限制 max_position_embeddings，避免后续越界
            if max_pos and max_pos > limit:
                self._apply_fix(
                    "model.max_position_embeddings",
                    max_pos,
                    limit,
                    f"FlashAttention 限制: 最大位置编码需 <= {limit}，修复 max_position_embeddings"
                )
            if not seq_len and max_pos and max_pos > limit:
                self.warnings.append(
                    f"FlashAttention: 未显式设置 seq_length，已限制 max_position_embeddings 到 {limit}"
                )

    def _check_mla_parameters(self, tp: int, hidden_size: int, num_attention_heads: int) -> None:
        """检查 MLA (Multi-Head Latent Attention) 相关参数的兼容性"""
        mla_config = self.config.get('mla', {})
        
        if not mla_config.get('multi_head_latent_attention', False):
            return

        parallel_cfg = self.config.setdefault('parallel', {})
        cp = int(parallel_cfg.get('context_parallel_size', 1) or 1)
        pp = int(parallel_cfg.get('pipeline_model_parallel_size', 1) or 1)
        ep = int(parallel_cfg.get('expert_model_parallel_size', 1) or 1)
        world_size = 8

        # DeepSeek MLA 在 TP>1 时对并行切分更敏感；
        # 优先对齐 CP=TP，若资源不满足则回退 TP=1/CP=1。
        if tp > 1 and cp != tp:
            candidate = tp * pp * ep * tp
            if candidate > 0 and world_size % candidate == 0:
                self._apply_fix(
                    "parallel.context_parallel_size",
                    cp,
                    tp,
                    f"MLA 模式下建议 CP 与 TP 对齐，修复 CP: {cp} -> {tp}"
                )
                cp = tp
                parallel_cfg['context_parallel_size'] = cp
            else:
                self._apply_fix(
                    "parallel.tensor_model_parallel_size",
                    tp,
                    1,
                    "MLA 并行组合无法满足 CP=TP 且 world_size 约束，回退 TP=1"
                )
                if cp != 1:
                    self._apply_fix(
                        "parallel.context_parallel_size",
                        cp,
                        1,
                        "MLA 并行组合回退时同步设置 CP=1"
                    )
                tp = 1
                cp = 1
                parallel_cfg['tensor_model_parallel_size'] = tp
                parallel_cfg['context_parallel_size'] = cp
        
        # 检查 LoRA 秩是否合理
        q_lora_rank = mla_config.get('q_lora_rank', 0)
        kv_lora_rank = mla_config.get('kv_lora_rank', 0)
        
        if q_lora_rank > 0 and q_lora_rank % tp != 0:
            new_q_lora_rank = self._find_nearest_divisible(q_lora_rank, tp)
            self._apply_fix(
                "mla.q_lora_rank",
                q_lora_rank,
                new_q_lora_rank,
                f"Q LoRA 秩 {q_lora_rank} 不能被 TP={tp} 整除"
            )
        
        if kv_lora_rank > 0 and kv_lora_rank % tp != 0:
            new_kv_lora_rank = self._find_nearest_divisible(kv_lora_rank, tp)
            self._apply_fix(
                "mla.kv_lora_rank",
                kv_lora_rank,
                new_kv_lora_rank,
                f"KV LoRA 秩 {kv_lora_rank} 不能被 TP={tp} 整除"
            )
        
        # 检查头维度是否合理
        #
        # 注意：对于 DeepSeekV3 这类 MLA 模型，qk_rope/qk_nope/v_head_dim
        # 并不满足 hidden_size * tp / num_attention_heads 这类普通 MHA/GQA 推导公式。
        # 例如 deepseekv3 模板里 hidden_size=5120、num_attention_heads=128，
        # 但 qk_rope_head_dim=64、qk_nope_head_dim=128、v_head_dim=128。
        #
        # 之前这里会把这些值强行改写成 expected_head_dim，导致 task1 生成的
        # PTA 脚本把原本合法的 MLA 配置改坏，最终在运行期出现 shape mismatch。
        #
        # 因此这里不再对 MLA 头维度做“自动修复”，只保留存在性/正值检查。
        qk_rope_head_dim = mla_config.get('qk_rope_head_dim', 0)
        qk_nope_head_dim = mla_config.get('qk_nope_head_dim', 0)
        v_head_dim = mla_config.get('v_head_dim', 0)

        for field_name, value in (
            ("mla.qk_rope_head_dim", qk_rope_head_dim),
            ("mla.qk_nope_head_dim", qk_nope_head_dim),
            ("mla.v_head_dim", v_head_dim),
        ):
            if value is None:
                continue
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                continue
            if numeric <= 0:
                self.warnings.append(f"{field_name}={value} 非法，建议保留原模板中的正整数配置")
    
    def _check_rope_parameters(self) -> None:
        """检查 ROPE 相关参数的合理性"""
        rope_config = self.config.get('rope', {})
        model_config = self.config.get('model', {})
        
        seq_length = model_config.get('seq_length', 0)
        max_position_embeddings = model_config.get('max_position_embeddings', 0)
        rope_scaling_factor = rope_config.get('rope_scaling_factor', 1.0)
        rope_scaling_original_max_position_embeddings = rope_config.get(
            'rope_scaling_original_max_position_embeddings', 0
        )
        
        if rope_scaling_factor > 1.0 and rope_scaling_original_max_position_embeddings > 0:
            effective_max_length = rope_scaling_original_max_position_embeddings * rope_scaling_factor
            
            if max_position_embeddings > effective_max_length:
                self.warnings.append(
                    f"ROPE 参数: 最大位置编码 {max_position_embeddings} 超过 ROPE 缩放后的有效长度 {effective_max_length}"
                )
            
            if seq_length > effective_max_length:
                self.warnings.append(
                    f"ROPE 参数: 序列长度 {seq_length} 超过 ROPE 缩放后的有效长度 {effective_max_length}"
                )
    
    def _check_sequence_length(self) -> None:
        """检查序列长度和位置编码的兼容性"""
        model_config = self.config.get('model', {})
        
        seq_length = model_config.get('seq_length', 0)
        max_position_embeddings = model_config.get('max_position_embeddings', 0)
        if self.seq_cap_enabled and self.max_seq_length_cap > 0:
            cap = self.max_seq_length_cap
            if seq_length and seq_length > cap:
                self._apply_fix(
                    "model.seq_length",
                    seq_length,
                    cap,
                    f"序列长度 {seq_length} 超过上限 {cap}，自动限幅"
                )
                seq_length = cap
            if max_position_embeddings and max_position_embeddings > cap:
                self._apply_fix(
                    "model.max_position_embeddings",
                    max_position_embeddings,
                    cap,
                    f"最大位置编码 {max_position_embeddings} 超过上限 {cap}，自动限幅"
                )
                max_position_embeddings = cap
        
        if seq_length > max_position_embeddings:
            self._apply_fix(
                "model.max_position_embeddings",
                max_position_embeddings,
                seq_length,
                f"序列长度 {seq_length} 大于最大位置编码 {max_position_embeddings}"
            )
    
    def _check_vocab_size(self) -> None:
        """检查词汇表大小的兼容性"""
        model_config = self.config.get('model', {})
        
        vocab_size = model_config.get('vocab_size', 0)
        padded_vocab_size = model_config.get('padded_vocab_size', 0)
        make_vocab_size_divisible_by = model_config.get('make_vocab_size_divisible_by', 1)
        
        if padded_vocab_size > 0 and padded_vocab_size < vocab_size:
            self._apply_fix(
                "model.padded_vocab_size",
                padded_vocab_size,
                vocab_size,
                f"填充后的词汇表大小 {padded_vocab_size} 小于原始词汇表大小 {vocab_size}"
            )
        
        if make_vocab_size_divisible_by > 1 and vocab_size % make_vocab_size_divisible_by != 0:
            new_vocab_size = self._find_nearest_divisible(vocab_size, make_vocab_size_divisible_by)
            self._apply_fix(
                "model.vocab_size",
                vocab_size,
                new_vocab_size,
                f"词汇表大小 {vocab_size} 不能被 make_vocab_size_divisible_by={make_vocab_size_divisible_by} 整除"
            )

    def _check_recompute_constraints(self) -> None:
        """检查激活重计算相关互斥约束。"""
        parallel_cfg = self.config.get("parallel", {})
        training_cfg = self.config.get("training", {})

        if self._to_bool(parallel_cfg.get("sequence_parallel", False)) and self._to_bool(training_cfg.get("distribute_saved_activations", False)):
            self._apply_fix(
                "training.distribute_saved_activations",
                True,
                False,
                "sequence_parallel=True 时不能启用 distribute_saved_activations"
            )

        if self._to_bool(training_cfg.get("distribute_saved_activations", False)) and training_cfg.get("recompute_granularity") != "full":
            self._apply_fix(
                "training.distribute_saved_activations",
                True,
                False,
                "Megatron 仅在 recompute_granularity=full 时支持 distribute_saved_activations"
            )
    
    # 对于每个参数都有可选范围
    def _mutate_recompute(self, mutation_space: Dict[str, List[Any]]={}) -> Dict[str, Any]:
        """
        依据约束与可选范围对重计算相关参数进行变异。
        mutation_space 例:
        {
            "recompute_activations": [True, False],
            "recompute_granularity": ["selective","full"],
            "recompute_method": ["uniform","block"],
            "recompute_num_layers": [4,8,16],
            "distribute_saved_activations": [True]
        }
        若某键未出现在 mutation_space 中, 则随机选择允许值。
        返回本次变异的参数字典。
        """
        training_cfg = self.config.setdefault("training", {})
        parallel_cfg = self.config.get("parallel", {})
        moe_cfg = self.config.get("moe", {})
        use_moe_grouped_gemm = moe_cfg.get("moe_grouped_gemm", False)
        if not use_moe_grouped_gemm:
            return {}

        num_layers = self.config.get("model", {}).get("num_layers", 0)
        tp = parallel_cfg.get("tensor_model_parallel_size", 1)
        pp = parallel_cfg.get("pipeline_model_parallel_size", 1)
        sp = self._to_bool(parallel_cfg.get("sequence_parallel", False))

        def choose(key: str, candidates: List[Any]) -> Any:
            return random.choice(candidates)

        mutations = {}

        # 1. recompute_activations
        if "recompute_activations" in mutation_space:
            recompute_activations = choose("recompute_activations", mutation_space["recompute_activations"])
        else:
            recompute_activations = random.choice([True, False])
        training_cfg["recompute_activations"] = recompute_activations
        mutations["recompute_activations"] = recompute_activations

        # 如果不启用, 清理相关参数并返回
        if not recompute_activations:
            for k in ["recompute_granularity", "recompute_method", "recompute_num_layers", "distribute_saved_activations"]:
                training_cfg.pop(k, None)
            return mutations

        # 2. granularity
        if "recompute_granularity" in mutation_space:
            granularity = choose("recompute_granularity", mutation_space["recompute_granularity"])
        else:
            granularity = random.choice(["selective", "full"])
        training_cfg["recompute_granularity"] = granularity
        mutations["recompute_granularity"] = granularity

        # 3. distribute_saved_activations 仅在启用、TP>1、未开启 SP 且 full recompute 时允许
        if granularity == "full" and tp > 1 and not sp:
            if "distribute_saved_activations" in mutation_space:
                distribute_flag = choose("distribute_saved_activations", mutation_space["distribute_saved_activations"])
            else:
                distribute_flag = random.choice([True, False])
            training_cfg["distribute_saved_activations"] = distribute_flag
            mutations["distribute_saved_activations"] = distribute_flag
        else:
            training_cfg.pop("distribute_saved_activations", None)

        # granularity = selective, 移除 method/num_layers
        if granularity != "full":
            training_cfg.pop("distribute_saved_activations", None)
            training_cfg.pop("recompute_method", None)
            training_cfg.pop("recompute_num_layers", None)
            return mutations
        # 我们保证下面的都是 full

        # 4. method (only when full)
        if "recompute_method" in mutation_space:
            method = choose("recompute_method", mutation_space["recompute_method"])
        else:
            method = random.choice(["uniform", "block"])
        training_cfg["recompute_method"] = method
        mutations["recompute_method"] = method

        # 5. num_layers (required when full)
        # 计算可允许范围
        if method == "uniform":
            max_allowed = max(1, num_layers)
        else:  # block
            # 每个 pipeline stage 的层数 (向下取整, 若不能整除仍允许选择 <= ceil)
            stage_layers_floor = num_layers // max(1, pp)
            stage_layers_ceil = math.ceil(num_layers / max(1, pp))
            max_allowed = max(stage_layers_floor, 1)
            # 为更稳健, 如果 floor=0 使用 ceil
            if max_allowed == 0:
                max_allowed = stage_layers_ceil

        if "recompute_num_layers" in mutation_space:
            candidate_list = mutation_space["recompute_num_layers"]
            # 过滤不合法
            valid_candidates = [c for c in candidate_list if isinstance(c, int) and c > 0 and c <= max_allowed]
            if not valid_candidates:
                # 回退: 使用 max_allowed
                recompute_num_layers = max_allowed
            else:
                recompute_num_layers = choose("recompute_num_layers", valid_candidates)
        else:
            # 随机挑选 (倾向于较小值避免过度分块)
            if max_allowed == 1:
                recompute_num_layers = 1
            else:
                recompute_num_layers = random.choice([1, max_allowed]) if max_allowed <= 4 else random.randint(1, max_allowed)
        training_cfg["recompute_num_layers"] = recompute_num_layers
        mutations["recompute_num_layers"] = recompute_num_layers

        if granularity == "full":
            training_cfg["recompute_activations"] = False
            mutations["recompute_activations"] = False

        # 可选: 记录到 issues 方便追踪
        self.issues.append(f"变异: 重计算配置 -> {mutations}")
        return mutations

    def _mutate_distributed_optimizer(self, mutation_space: Dict[str, List[Any]] = {}) -> Dict[str, Any]:
        """
        依据约束与可选范围对“优化器并行”相关参数进行变异，并原地写回到 self.config['training']。
        支持的键及默认候选（若 mutation_space 中未提供）:
        - use_distributed_optimizer: [True, False]
        - use_megatron_fsdp: [True, False]
        - use_torch_fsdp2: [True, False]
        - overlap_param_gather: [True, False]
        - overlap_param_gather_with_optimizer_step: [True, False]
        - overlap_grad_reduce: [True, False]
        - fp8_param_gather: [True, False]
        - enable_gloo_process_groups: [True, False]
        - use_dist_ckpt: [True, False]
        - ckpt_fully_parallel_save: [True, False]
        - dist_ckpt_optim_fully_reshardable: [True, False]
        - distrib_optim_fully_reshardable_mem_efficient: [True, False]

        约束落实：
        1) overlap_param_gather 仅在 use_distributed_optimizer 或 use_megatron_fsdp 时允许
           且必须搭配 overlap_grad_reduce=True，且禁止 legacy models
        2) use_torch_fsdp2 需要 torch>=2.4.0 且不能与 use_distributed_optimizer 同时启用
        3) overlap_param_gather_with_optimizer_step 仅在 use_distributed_optimizer 时允许
        4) fp8_param_gather 需满足 (use_distributed_optimizer 或 use_torch_fsdp2 或 use_megatron_fsdp 或 inference_mode)
        6) 若未启用 enable_gloo_process_groups，且 dist_ckpt_optim_fully_reshardable=True，
           则必须将 distrib_optim_fully_reshardable_mem_efficient=False
        7) 若 use_dist_ckpt=True 且 ckpt_fully_parallel_save=False 且 use_distributed_optimizer=True，则给出警告
        8) 若 delay_wgrad_compute=True：
           - transformer_impl 必须为 'transformer_engine'，否则关闭 delay_wgrad_compute
           - 当 overlap_grad_reduce=True 时，要求 TE>=2.8.0；否则关闭 overlap_grad_reduce
           - 当 gradient_accumulation_fusion=False 时，要求 TE>=2.7.0；否则强制开启 gradient_accumulation_fusion
        """
        training = self.config.setdefault("training", {})
        model_cfg = self.config.get("model", {})
        env_cfg = self.config.get("env", {})  # 可选: 提供 torch_version: "2.4.0", te_version: "2.8.0", inference_mode: bool
        mutations: Dict[str, Any] = {}

        def choose(key: str, default_candidates: List[Any]):
            cands = mutation_space.get(key, default_candidates)
            cands = [c for c in cands] if isinstance(cands, list) and len(cands) > 0 else default_candidates
            return random.choice(cands)

        def parse_version(v: str) -> Tuple[int, int, int]:
            try:
                parts = v.split("+")[0].split(".")
                major = int(parts[0]) if len(parts) > 0 else 0
                minor = int(parts[1]) if len(parts) > 1 else 0
                patch = int(parts[2]) if len(parts) > 2 else 0
                return (major, minor, patch)
            except Exception:
                return (0, 0, 0)

        # 1) 先随机/受限选择各参数
        use_distributed_optimizer = choose("use_distributed_optimizer", [True, False])
        use_megatron_fsdp = choose("use_megatron_fsdp", [False, True])  # 默认偏向禁用
        use_torch_fsdp2 = choose("use_torch_fsdp2", [False, True])      # 默认偏向禁用
        overlap_param_gather = choose("overlap_param_gather", [False, True])
        overlap_param_gather_with_optimizer_step = choose("overlap_param_gather_with_optimizer_step", [False, True])
        overlap_grad_reduce = choose("overlap_grad_reduce", [False, True])
        fp8_param_gather = choose("fp8_param_gather", [False, True])
        enable_gloo_process_groups = choose("enable_gloo_process_groups", [True, False])
        ckpt_fully_parallel_save = choose("ckpt_fully_parallel_save", [True, False])
        dist_ckpt_optim_fully_reshardable = choose("dist_ckpt_optim_fully_reshardable", [False, True])
        distrib_optim_fully_reshardable_mem_efficient = choose("distrib_optim_fully_reshardable_mem_efficient", [False, True])

        # 读取可能影响约束的其他配置
        use_legacy_models = bool(model_cfg.get("use_legacy_models", False))
        transformer_impl = training.get("transformer_impl", model_cfg.get("transformer_impl", "transformer_engine"))
        delay_wgrad_compute = bool(training.get("delay_wgrad_compute", False))
        gradient_accumulation_fusion = training.get("gradient_accumulation_fusion", True)
        inference_mode = bool(env_cfg.get("inference_mode", training.get("inference", False)))

        # 2) 逐条应用约束修正

        # 2.1 FSDP2 版本与互斥约束
        torch_v_str = env_cfg.get("torch_version", "")
        torch_ok = parse_version(torch_v_str) >= parse_version("2.4.0") if torch_v_str else False
        if use_torch_fsdp2 and use_distributed_optimizer:
            self.issues.append("已修复: use_torch_fsdp2 与 use_distributed_optimizer 互斥，关闭 use_torch_fsdp2")
            use_torch_fsdp2 = False
        if use_torch_fsdp2 and not torch_ok:
            self.issues.append("已修复: FSDP2 需要 PyTorch>=2.4.0，关闭 use_torch_fsdp2")
            use_torch_fsdp2 = False

        # 2.2 overlap_param_gather 仅在分布式优化器或Megatron FSDP下支持
        if overlap_param_gather and not (use_distributed_optimizer or use_megatron_fsdp):
            self.issues.append("已修复: overlap_param_gather 仅在分布式优化器或Megatron FSDP下支持，关闭 overlap_param_gather")
            overlap_param_gather = False

        # 2.3 overlap_param_gather_with_optimizer_step 仅在分布式优化器下支持
        if overlap_param_gather_with_optimizer_step and not use_distributed_optimizer:
            self.issues.append("已修复: overlap_param_gather_with_optimizer_step 仅在分布式优化器下支持，关闭该选项")
            overlap_param_gather_with_optimizer_step = False

        # 2.4 强约束：使用 overlap_param_gather 时，必须 overlap_grad_reduce=True 且禁止 legacy models
        if overlap_param_gather:
            if not overlap_grad_reduce:
                self.issues.append("已修复: 使用 overlap_param_gather 时必须启用 overlap_grad_reduce，强制开启 overlap_grad_reduce")
                overlap_grad_reduce = True
            if use_legacy_models:
                self.issues.append("已修复: overlap_param_gather 仅支持 MCore 模型，检测到 use_legacy_models=True，关闭 overlap_param_gather")
                overlap_param_gather = False

        # 2.5 fp8_param_gather 支持条件
        if fp8_param_gather and not (use_distributed_optimizer or use_torch_fsdp2 or use_megatron_fsdp or inference_mode):
            self.issues.append("已修复: fp8_param_gather 需要分布式优化器、Torch FSDP2、Megatron FSDP或推理模式，关闭 fp8_param_gather")
            fp8_param_gather = False

        # 2.7 未启用 Gloo PG 时，fully_reshardable 模式禁止 mem_efficient
        if not enable_gloo_process_groups and dist_ckpt_optim_fully_reshardable and distrib_optim_fully_reshardable_mem_efficient:
            self.issues.append("已修复: 未启用 Gloo PG 时，禁止 --distrib-optim-fully-reshardable-mem-efficient，关闭该选项")
            distrib_optim_fully_reshardable_mem_efficient = False

        # 2.9 delay_wgrad_compute 相关约束
        te_v_str = env_cfg.get("te_version", "")
        te_v = parse_version(te_v_str) if te_v_str else (0, 0, 0)

        if delay_wgrad_compute and transformer_impl != "transformer_engine":
            # 不改变实现，直接关闭 delay_wgrad_compute
            self.issues.append("已修复: delay_wgrad_compute 仅支持 transformer_engine，实现不符，关闭 delay_wgrad_compute")
            delay_wgrad_compute = False
            training["delay_wgrad_compute"] = False

        # overlap_grad_reduce 在 delay_wgrad_compute 下需要 TE>=2.8.0
        if delay_wgrad_compute and overlap_grad_reduce and te_v < parse_version("2.8.0"):
            self.issues.append("已修复: overlap_grad_reduce 需 TE>=2.8.0 才能与 delay_wgrad_compute 同用，关闭 overlap_grad_reduce")
            overlap_grad_reduce = False

        # 当关闭梯度累加融合时，需要 TE>=2.7.0
        if delay_wgrad_compute and (not gradient_accumulation_fusion) and te_v < parse_version("2.7.0"):
            self.issues.append("已修复: delay_wgrad_compute 配合关闭 gradient_accumulation_fusion 需 TE>=2.7.0，强制开启 gradient_accumulation_fusion")
            gradient_accumulation_fusion = True
            training["gradient_accumulation_fusion"] = True

        # 3) 写回配置
        training["use_distributed_optimizer"] = use_distributed_optimizer
        training["use_megatron_fsdp"] = use_megatron_fsdp
        training["use_torch_fsdp2"] = use_torch_fsdp2
        training["overlap_param_gather"] = overlap_param_gather
        training["overlap_param_gather_with_optimizer_step"] = overlap_param_gather_with_optimizer_step
        training["overlap_grad_reduce"] = overlap_grad_reduce
        training["fp8_param_gather"] = fp8_param_gather
        training["enable_gloo_process_groups"] = enable_gloo_process_groups
        training["ckpt_fully_parallel_save"] = ckpt_fully_parallel_save
        training["dist_ckpt_optim_fully_reshardable"] = dist_ckpt_optim_fully_reshardable
        training["distrib_optim_fully_reshardable_mem_efficient"] = distrib_optim_fully_reshardable_mem_efficient

        mutations.update({
            "use_distributed_optimizer": use_distributed_optimizer,
            "use_megatron_fsdp": use_megatron_fsdp,
            "use_torch_fsdp2": use_torch_fsdp2,
            "overlap_param_gather": overlap_param_gather,
            "overlap_param_gather_with_optimizer_step": overlap_param_gather_with_optimizer_step,
            "overlap_grad_reduce": overlap_grad_reduce,
            "fp8_param_gather": fp8_param_gather,
            "enable_gloo_process_groups": enable_gloo_process_groups,
            "ckpt_fully_parallel_save": ckpt_fully_parallel_save,
            "dist_ckpt_optim_fully_reshardable": dist_ckpt_optim_fully_reshardable,
            "distrib_optim_fully_reshardable_mem_efficient": distrib_optim_fully_reshardable_mem_efficient,
        })

        self.issues.append(f"变异: 优化器并行配置 -> {mutations}")
        return mutations

    def _find_nearest_divisible(self, value: int, divisor: int) -> int:
        """找到最接近 value 且能被 divisor 整除的数"""
        if divisor == 0:
            return value
            
        # 计算上下两个最近的整除值
        lower = (value // divisor) * divisor
        higher = lower + divisor
        
        # 返回更大的,防止出现0
        return higher
    
    def _apply_fix(self, key_path: str, old_value: Any, new_value: Any, issue: str) -> None:
        """应用修复并记录问题"""
        self.issues.append(f"已修复: {issue}，将 {old_value} 改为 {new_value}")
        
        # 更新配置
        keys = key_path.split('.')
        current = self.config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = new_value
        
        self.fixes_applied += 1

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml_config(config: Dict[str, Any], file_path: str) -> None:
    """保存 YAML 配置文件"""
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def main():
    parser = argparse.ArgumentParser(description="增强版 Megatron-LM YAML 配置校验修正器")
    parser.add_argument("input_config", help="输入 YAML 配置文件路径")
    parser.add_argument("--output", "-o", help="输出 YAML 配置文件路径（可选）")
    parser.add_argument("--dry-run", "-d", action="store_true", 
                       help="干运行模式，只显示问题不修改文件")
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = load_yaml_config(args.input_config)
    except Exception as e:
        print(f"错误: 无法加载配置文件 {args.input_config}: {e}")
        return
    
    # 验证和修正配置
    validator = EnhancedMegatronConfigValidator(config)
    fixed_config, issues, warnings, fixes_applied = validator.validate_and_fix()
    
    # 输出结果
    print(f"检查完成，发现 {len(issues)} 个问题，{len(warnings)} 个警告，应用了 {fixes_applied} 个修复")
    print("=" * 50)
    
    if issues:
        print("\n问题列表:")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print("\n警告列表:")
        for warning in warnings:
            print(f"  ⚠ {warning}")
    
    # 保存修正后的配置
    if not args.dry_run and args.output:
        try:
            save_yaml_config(fixed_config, args.output)
            print(f"\n修正后的配置已保存到: {args.output}")
        except Exception as e:
            print(f"错误: 无法保存配置文件 {args.output}: {e}")
    elif args.dry_run:
        print("\n干运行模式: 未保存任何更改")
    elif not args.output:
        print("\n未指定输出文件: 使用 --output 参数指定输出文件以保存更改")

if __name__ == "__main__":
    main()
