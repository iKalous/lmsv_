import copy
import unittest

from utils.runtime.mf_converter import (
    _apply_mf_seed_mutations,
    _normalize_qwen3_config,
    _resolve_mf_train_dataset_sample_size,
    _resolve_runtime_global_batch_size,
)
from utils.runtime.convert_ckpt import _patch_qwen3_mg2hf_router_compat


class MfConverterSeedMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = {
            "supported_models": ["qwen3"],
            "mutation_count": {"min": 2, "max": 2},
            "parameters": [
                {
                    "name": "memory_optimize_level",
                    "target": ["context", "memory_optimize_level"],
                    "values": ["O0", "O1"],
                },
                {
                    "name": "jit_level",
                    "target": ["context", "jit_config", "jit_level"],
                    "values": ["O0", "O1", "O2"],
                },
                {
                    "name": "layer_decay",
                    "target": ["layer_decay"],
                    "values": [0.5, 0.75, 0.9],
                },
                {
                    "name": "gradient_accumulation_shard",
                    "target": ["parallel", "parallel_optimizer_config", "gradient_accumulation_shard"],
                    "values": [True, False],
                },
                {
                    "name": "parallel_optimizer_threshold",
                    "target": ["parallel", "parallel_optimizer_config", "parallel_optimizer_threshold"],
                    "values": [32, 64, 128],
                },
            ],
        }
        self.base_config = {
            "context": {
                "memory_optimize_level": "O0",
                "jit_config": {"jit_level": "O0"},
            },
            "parallel": {
                "parallel_optimizer_config": {
                    "gradient_accumulation_shard": False,
                    "parallel_optimizer_threshold": 64,
                }
            },
            "recompute_config": {
                "recompute": False,
                "select_recompute": False,
            },
            "layer_decay": 0.65,
        }

    def test_same_seed_is_deterministic(self) -> None:
        first = copy.deepcopy(self.base_config)
        second = copy.deepcopy(self.base_config)

        _apply_mf_seed_mutations(first, {"seed": 1, "model_name": "qwen3"}, "qwen3", schema=self.schema)
        _apply_mf_seed_mutations(second, {"seed": 1, "model_name": "qwen3"}, "qwen3", schema=self.schema)

        self.assertEqual(first, second)

    def test_different_seeds_produce_different_mutations(self) -> None:
        seed_one = copy.deepcopy(self.base_config)
        seed_two = copy.deepcopy(self.base_config)

        _apply_mf_seed_mutations(seed_one, {"seed": 1, "model_name": "qwen3"}, "qwen3", schema=self.schema)
        _apply_mf_seed_mutations(seed_two, {"seed": 2, "model_name": "qwen3"}, "qwen3", schema=self.schema)

        self.assertNotEqual(seed_one, seed_two)
        self.assertTrue(
            seed_one["context"]["memory_optimize_level"] != self.base_config["context"]["memory_optimize_level"]
            or seed_one["context"]["jit_config"]["jit_level"] != self.base_config["context"]["jit_config"]["jit_level"]
            or seed_one["layer_decay"] != self.base_config["layer_decay"]
        )
        self.assertTrue(
            seed_two["context"]["memory_optimize_level"] != self.base_config["context"]["memory_optimize_level"]
            or seed_two["context"]["jit_config"]["jit_level"] != self.base_config["context"]["jit_config"]["jit_level"]
            or seed_two["layer_decay"] != self.base_config["layer_decay"]
        )

    def test_unsupported_model_is_not_mutated(self) -> None:
        config = copy.deepcopy(self.base_config)

        _apply_mf_seed_mutations(config, {"seed": 1, "model_name": "qwen2"}, "qwen2", schema=self.schema)

        self.assertEqual(config, self.base_config)

    def test_runtime_global_batch_size_uses_micro_batch_num_only(self) -> None:
        self.assertEqual(_resolve_runtime_global_batch_size(1, 2, 16), 32)
        self.assertEqual(_resolve_runtime_global_batch_size(1, 2, 1), 2)
        self.assertEqual(_resolve_runtime_global_batch_size(1, 4, 8), 32)

    def test_train_dataset_samples_use_runtime_batch_size(self) -> None:
        self.assertEqual(_resolve_mf_train_dataset_sample_size(2, 1, 32, 128), 128)
        self.assertEqual(_resolve_mf_train_dataset_sample_size(2, 0, 32, 128), 256)

    def test_qwen3_template_moe_fields_removed_without_explicit_moe_args(self) -> None:
        config = {
            "model": {
                "model_config": {
                    "model_type": "qwen3",
                    "hidden_size": 896,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 8,
                    "n_routed_experts": 8,
                    "moe_router_topk": 2,
                    "head_dim": 128,
                    "offset": [0, 0],
                }
            },
            "parallel_config": {},
            "parallel": {},
            "context": {},
            "train_dataset": {
                "input_columns": ["input_ids", "labels", "attention_mask"],
            },
        }
        all_args = {
            "model_type": "qwen3",
            "hidden_size": 896,
            "num_attention_heads": 8,
            "num_query_groups": 2,
            "kv_channels": 128,
            "group_query_attention": True,
            "qk_layernorm": True,
        }

        _normalize_qwen3_config(config, all_args, dp=2, tp=1, pp=2, cp=1)

        model_cfg = config["model"]["model_config"]
        self.assertNotIn("n_routed_experts", model_cfg)
        self.assertNotIn("moe_router_topk", model_cfg)
        self.assertEqual(model_cfg["num_key_value_heads"], 2)

    def test_qwen3_mg2hf_router_key_compat_patch_suppresses_missing_router_key(self) -> None:
        class Converter:
            def set_model_layer_mlp(self, *args, **kwargs):
                raise KeyError("layers_mlp_router")

        converter = Converter()
        _patch_qwen3_mg2hf_router_compat(converter, "qwen3")

        self.assertIsNone(converter.set_model_layer_mlp({}, {}, 0, 0))


if __name__ == "__main__":
    unittest.main()
