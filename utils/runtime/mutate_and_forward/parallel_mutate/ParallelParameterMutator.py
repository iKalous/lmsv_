#!/usr/bin/env python3
"""
Megatron-LM 并行参数变异器
确保并行参数都是2的幂次，并且所有并行参数的乘积乘以数据并行大小等于总GPU数量
"""

import yaml
import math
import argparse
import random
from typing import Dict, Any, List, Tuple

class ParallelParameterMutator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.changes = []

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
    def _normalize_expert_count(value: Any) -> int:
        """Normalize expert count for MoE detection."""
        if value in (None, "", "None", "null"):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
        
    def mutate_parallel_parameters(self) -> Tuple[Dict[str, Any], List[str]]:
        """
        变异并行参数，确保它们都是2的幂次且乘积乘以数据并行大小等于总GPU数量
        
        Returns:
            Tuple[修正后的配置, 变更描述列表]
        """
        # 获取总GPU数量
        total_gpus = self._get_total_gpus()
        # total_gpus = 8
        if total_gpus <= 0:
            raise ValueError(f"无效的总GPU数量: {total_gpus}")
        print("总GPU数量:",total_gpus)
        
        # 检查总GPU数量是否为2的幂次
        if not self._is_power_of_two(total_gpus):
            self.changes.append(f"警告: 总GPU数量 {total_gpus} 不是2的幂次，这可能导致性能问题")
        
        # 获取并行配置部分，如果没有则创建
        parallel_config = self.config.setdefault('parallel', {})
        
        # 获取当前并行参数，如果没有则设置默认值
        tp = parallel_config.get('--tensor-model-parallel-size', parallel_config.get('tensor_model_parallel_size', 1))
        pp = parallel_config.get('--pipeline-model-parallel-size', parallel_config.get('pipeline_model_parallel_size', 1))
        ep = parallel_config.get('--expert-model-parallel-size', parallel_config.get('expert_model_parallel_size', 1))
        cp = parallel_config.get('--context-parallel-size', parallel_config.get('context_parallel_size', 1))
        vpp = parallel_config.get('--num-layers-per-virtual-pipeline-stage', parallel_config.get('num_layers_per_virtual_pipeline_stage', 1))
        sp = parallel_config.get('--sequence-parallel', parallel_config.get('sequence_parallel', False))
        
        # 检查是否有MOE配置来决定是否使用EP
        expert_count = 0
        if 'moe' in self.config:
            expert_count = self._normalize_expert_count(
                self.config['moe'].get('--num-experts', self.config['moe'].get('num_experts', 0))
            )
        has_moe = expert_count > 1
        use_ep = has_moe  # 只有MOE模型使用EP
        # use_ep = True
        # 计算当前数据并行大小
        current_dp = total_gpus // (tp * pp * (ep if use_ep else 1) * cp)
        
        # 生成新的并行参数组合
        new_tp, new_pp, new_ep, new_cp, new_dp = self._generate_parallel_combination(
            total_gpus, use_ep
        )

        # 约束: CP>1 时 TP 必须 >=2，尽量通过挪动因子修复
        if new_cp > 1 and new_tp < 2:
            adjusted = False
            if new_cp % 2 == 0:
                new_cp //= 2
                new_tp *= 2
                adjusted = True
            elif new_dp % 2 == 0:
                new_dp //= 2
                new_tp *= 2
                adjusted = True
            elif use_ep and new_ep % 2 == 0:
                new_ep //= 2
                new_tp *= 2
                adjusted = True
            elif new_pp % 2 == 0:
                new_pp //= 2
                new_tp *= 2
                adjusted = True

            if not adjusted:
                new_dp *= new_cp
                new_cp = 1
        
        # 应用新的并行参数
        # 需要首先判断一下原本是 "--tensor-model-parallel-size" 还是 "tensor_model_parallel_size"
        key = '--tensor-model-parallel-size' if '--tensor-model-parallel-size' in parallel_config else 'tensor_model_parallel_size'
        parallel_config[key] = new_tp
        self.changes.append(f"TP: {tp} -> {new_tp}")

        # 需要首先判断一下原本是 "--pipeline-model-parallel-size" 还是 "pipeline_model_parallel_size"
        key = '--pipeline-model-parallel-size' if '--pipeline-model-parallel-size' in parallel_config else 'pipeline_model_parallel_size'
        parallel_config[key] = new_pp
        self.changes.append(f"PP: {pp} -> {new_pp}")

        if use_ep:
            # 需要首先判断一下原本是 "--expert-model-parallel-size" 还是 "expert_model_parallel_size"
            key = '--expert-model-parallel-size' if '--expert-model-parallel-size' in parallel_config else 'expert_model_parallel_size'
            parallel_config[key] = new_ep
            self.changes.append(f"EP: {ep} -> {new_ep}")
        elif not use_ep:
            removed = False
            for key in ('--expert-model-parallel-size', 'expert_model_parallel_size'):
                if key in parallel_config:
                    del parallel_config[key]
                    removed = True
            if removed:
                self.changes.append("移除 EP 参数（非MOE模型）")
    
        # 需要首先判断一下原本是 "--context-parallel-size" 还是 "context_parallel_size"
        key = '--context-parallel-size' if '--context-parallel-size' in parallel_config else 'context_parallel_size'
        parallel_config[key] = new_cp
        self.changes.append(f"CP: {cp} -> {new_cp}")
        # 只需要打印，不需要在配置中设置DP
        self.changes.append(f"DP: {current_dp} -> {new_dp}")

        # 约束: TP=1 时不能启用 sequence_parallel；MoE 且 TP>1 时必须启用。
        key = '--sequence-parallel' if '--sequence-parallel' in parallel_config else 'sequence_parallel'
        current_sp = self._to_bool(sp)
        if new_tp == 1 and current_sp:
            parallel_config[key] = False
            self.changes.append("SP: True -> False (TP=1 禁用)")
        elif use_ep and new_tp > 1 and not current_sp:
            parallel_config[key] = True
            self.changes.append("SP: False -> True (MoE + TP>1 强制启用)")
        
        # 确保虚拟流水线并行参数合理
        num_layers = self.config.get('model', {}).get('--num-layers', self.config.get('model', {}).get('num_layers', 0))
        if num_layers > 0 and new_pp > 1 and vpp > 1:
            # 检查虚拟流水线并行是否合理
            if num_layers % (new_pp * vpp) != 0:
                # 调整虚拟流水线并行参数
                new_vpp = self._find_optimal_vpp(num_layers, new_pp)
                # 需要首先判断一下原本是 "--num-layers-per-virtual-pipeline-stage" 还是 "num_layers_per_virtual_pipeline_stage"
                key = '--num-layers-per-virtual-pipeline-stage' if '--num-layers-per-virtual-pipeline-stage' in parallel_config else 'num_layers_per_virtual_pipeline_stage'
                parallel_config[key] = new_vpp
                self.changes.append(f"VPP 从 {vpp} 改为 {new_vpp}")
        
        return self.config, self.changes
    
    def _get_total_gpus(self) -> int:
        """从配置中获取总GPU数量"""
        distributed_config = self.config.get('distributed', {})
        num_nodes = distributed_config.get('NNODES', distributed_config.get('num_nodes', 1))
        npus_per_node = distributed_config.get('NPUS_PER_NODE', distributed_config.get('npus_per_node', 1))
        if (num_nodes * npus_per_node) == 1:
            return 8 # 默认单节点8卡
        return num_nodes * npus_per_node
    
    def _generate_parallel_combination(self, total_gpus: int, use_ep: bool) -> Tuple[int, int, int, int, int]:
        """
        通过质因数分解生成满足条件的并行参数组合，确保乘积等于 total_gpus。
        
        Args:
            total_gpus: 总 GPU 数量
            use_ep: 是否使用专家并行 (EP)
            
        Returns:
            Tuple[TP, PP, EP, CP, DP]
        """
        if total_gpus < 1:
            raise ValueError(f"总 GPU 数量 {total_gpus} 必须大于 0")
        
        # 获取质因数分解列表（包括重复因子）
        prime_factors = self._get_prime_factors(total_gpus)
        
        # 初始化并行维度
        tp, pp, ep, cp, dp = 1, 1, 1, 1, 1
        
        # 确定可分配的维度数量
        available_dims = 5 if use_ep else 4
        
        # 随机分配每个质因数到不同的并行维度
        for factor in prime_factors:
            dim_choice = random.randint(0, available_dims - 1)
            
            if dim_choice == 0:
                tp *= factor
            elif dim_choice == 1:
                pp *= factor
            elif dim_choice == 2:
                cp *= factor
            elif dim_choice == 3:
                dp *= factor
            else:  # dim_choice == 4，只有 use_ep=True 时才会执行
                ep *= factor
        
        return tp, pp, ep, cp, dp

    def _get_prime_factors(self, n: int) -> List[int]:
        """
        获取一个数的所有质因数（包括重复的）
        
        Args:
            n: 要分解的数
            
        Returns:
            质因数列表，包含所有质因数（重复出现）
        """
        if n == 1:
            return [1]
        
        factors = []
        temp = n
        
        # 处理因子 2
        while temp % 2 == 0:
            factors.append(2)
            temp //= 2
        
        # 处理奇数因子
        f = 3
        while f * f <= temp:
            if temp % f == 0:
                factors.append(f)
                temp //= f
            else:
                f += 2
        
        # 如果还有剩余且大于1，说明它本身是质数
        if temp > 1:
            factors.append(temp)
        
        return factors
    
    def _get_power_of_two_factors(self, n: int) -> List[int]:
        """
        获取一个数的所有2的幂次因子
        
        Args:
            n: 要分解的数
            
        Returns:
            2的幂次因子列表
        """
        factors = []
        current = n
        
        # 从大到小尝试所有可能的2的幂次因子
        max_exponent = int(math.log2(n)) if n > 0 else 0
        for i in range(max_exponent, 0, -1):
            factor = 2 ** i
            if current % factor == 0:
                factors.append(factor)
                current //= factor
        
        # 如果还有剩余，添加剩余的部分
        if current > 1:
            factors.append(current)
        
        return factors
    
    def _find_optimal_vpp(self, num_layers: int, pp: int) -> int:
        """
        找到最优的虚拟流水线并行参数
        
        Args:
            num_layers: 总层数
            pp: 流水线并行大小
            
        Returns:
            最优的VPP值
        """
        # 找到能整除层数/PP的最大因子
        layers_per_stage = num_layers // pp
        best_vpp = 1
        
        for i in range(2, layers_per_stage + 1):
            if layers_per_stage % i == 0:
                best_vpp = i
        
        return best_vpp
    
    def _is_power_of_two(self, n: int) -> bool:
        """检查一个数是否是2的幂次"""
        return n > 0 and (n & (n - 1)) == 0
    
    def _find_nearest_power_of_two(self, n: int) -> int:
        """找到最接近n的2的幂次数"""
        if n <= 0:
            return 1
        
        # 计算上下两个最近的2的幂次数
        lower = 2 ** int(math.log2(n))
        higher = 2 ** (int(math.log2(n)) + 1)
        
        # 返回更接近的一个
        if n - lower < higher - n:
            return lower
        else:
            return higher

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(file_path, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml_config(config: Dict[str, Any], file_path: str) -> None:
    """保存 YAML 配置文件"""
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def main():
    parser = argparse.ArgumentParser(description="Megatron-LM 并行参数变异器")
    parser.add_argument("input_config", help="输入 YAML 配置文件路径")
    parser.add_argument("--output", "-o", help="输出 YAML 配置文件路径（可选）")
    parser.add_argument("--dry-run", "-d", action="store_true", 
                       help="干运行模式，只显示变更不修改文件")
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = load_yaml_config(args.input_config)
    except Exception as e:
        print(f"错误: 无法加载配置文件 {args.input_config}: {e}")
        return
    
    # 变异并行参数
    mutator = ParallelParameterMutator(config)
    try:
        mutated_config, changes = mutator.mutate_parallel_parameters()
    except Exception as e:
        print(f"错误: 无法变异并行参数: {e}")
        return
    
    # 输出结果
    if changes:
        print("已应用以下变更:")
        for change in changes:
            print(f"  - {change}")
    else:
        print("无需变更，并行参数已符合要求")
    
    # 保存修正后的配置
    if not args.dry_run and args.output:
        try:
            save_yaml_config(mutated_config, args.output)
            print(f"\n修正后的配置已保存到: {args.output}")
        except Exception as e:
            print(f"错误: 无法保存配置文件 {args.output}: {e}")
    elif args.dry_run:
        print("\n干运行模式: 未保存任何更改")
    elif not args.output:
        print("\n未指定输出文件: 使用 --output 参数指定输出文件以保存更改")

if __name__ == "__main__":
    main()
