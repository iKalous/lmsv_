import tempfile
import unittest
from pathlib import Path

from utils.task.task1 import (
    align_bias_linear_flags,
    apply_deepseekv3_unified_low_memory_profile,
    apply_multinode_script_settings,
    apply_script_constraints,
    _contains_mf_fatal_text,
    sanitize_moe_expert_bias_aux_loss,
    sanitize_swiglu_fusion_script,
    sanitize_task1_mutation_runtime_flags,
)


class Task1ScriptSanitizerTests(unittest.TestCase):
    def test_task1_mutation_runtime_flags_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "task1.sh"
            script_path.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --use-flash-attn \\",
                        "    --overlap-grad-reduce \\",
                        "    --overlap-param-gather \\",
                        "    --train-iters 10 \\",
                        '"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(sanitize_task1_mutation_runtime_flags(script_path))

            content = script_path.read_text(encoding="utf-8")
            self.assertNotIn("--use-flash-attn", content)
            self.assertNotIn("--overlap-grad-reduce", content)
            self.assertNotIn("--overlap-param-gather", content)
            self.assertIn("--train-iters 10", content)

    def test_moe_expert_bias_forces_positive_aux_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "task1.sh"
            script_path.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --moe-router-enable-expert-bias \\",
                        "    --moe-aux-loss-coeff 0.0 \\",
                        '"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(sanitize_moe_expert_bias_aux_loss(script_path))

            content = script_path.read_text(encoding="utf-8")
            self.assertIn("--moe-router-enable-expert-bias", content)
            self.assertIn("--moe-aux-loss-coeff 0.01", content)
            self.assertNotIn("--moe-aux-loss-coeff 0.0 \\", content)

    def test_multinode_pta_env_injects_all_socket_ifnames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "task1.sh"
            script_path.write_text(
                "\n".join(
                    [
                        "NPUS_PER_NODE=8",
                        "MASTER_ADDR=127.0.0.1",
                        "MASTER_PORT=29500",
                        "NNODES=1",
                        "NODE_RANK=0",
                        "WORLD_SIZE=8",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(
                apply_multinode_script_settings(
                    script_path,
                    local_workers=8,
                    total_workers=16,
                    nnodes=2,
                    node_rank=1,
                    master_addr="10.0.0.1",
                    master_port=29501,
                    enable_pta_env=True,
                )
            )

            content = script_path.read_text(encoding="utf-8")
            self.assertIn("GLOO_SOCKET_IFNAME", content)
            self.assertIn("TP_SOCKET_IFNAME", content)
            self.assertIn("HCCL_SOCKET_IFNAME", content)
            self.assertIn("unset RANK_TABLE_FILE", content)
            self.assertIn("unset RANK_SIZE", content)
            self.assertIn("unset RANK_ID", content)
            self.assertIn("unset LOCAL_RANK", content)
            self.assertIn("unset LOCAL_WORLD_SIZE", content)
            self.assertIn("unset TORCHELASTIC_RUN_ID", content)

    def test_single_node_pta_env_does_not_inject_socket_ifnames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "task1.sh"
            script_path.write_text(
                "\n".join(
                    [
                        "NPUS_PER_NODE=8",
                        "MASTER_ADDR=127.0.0.1",
                        "MASTER_PORT=29500",
                        "NNODES=1",
                        "NODE_RANK=0",
                        "WORLD_SIZE=8",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(
                apply_multinode_script_settings(
                    script_path,
                    local_workers=8,
                    total_workers=8,
                    nnodes=1,
                    node_rank=0,
                    master_addr="127.0.0.1",
                    master_port=29500,
                    enable_pta_env=True,
                )
            )

            content = script_path.read_text(encoding="utf-8")
            self.assertNotIn("GLOO_SOCKET_IFNAME", content)
            self.assertNotIn("TP_SOCKET_IFNAME", content)
            self.assertNotIn("HCCL_SOCKET_IFNAME", content)
            self.assertNotIn("unset RANK_TABLE_FILE", content)
            self.assertNotIn("unset LOCAL_RANK", content)
            self.assertNotIn("unset LOCAL_WORLD_SIZE", content)

    def test_gqa_tensor_parallel_is_reduced_to_query_group_divisor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "qwen3.sh"
            script_path.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --num-attention-heads 8 \\",
                        "    --num-query-groups 2 \\",
                        "    --tensor-model-parallel-size 4 \\",
                        '"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(apply_script_constraints(script_path))

            content = script_path.read_text(encoding="utf-8")
            self.assertIn("--tensor-model-parallel-size 2", content)

    def test_msa_bias_linear_flags_follow_pta_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pta_script = Path(tmpdir) / "pta.sh"
            msa_script = Path(tmpdir) / "msa.sh"
            pta_script.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --train-iters 10 \\",
                        "    --disable-bias-linear \\",
                        '"',
                    ]
                ),
                encoding="utf-8",
            )
            msa_script.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --train-iters 10 \\",
                        '"',
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(align_bias_linear_flags(pta_script, msa_script))

            content = msa_script.read_text(encoding="utf-8")
            self.assertIn("--disable-bias-linear", content)
            self.assertNotIn("--add-bias-linear", content)

    def test_num_moe_experts_triggers_disable_bias_linear(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "moe.sh"
            script_path.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --train-iters 10 \\",
                        "    --num-moe-experts 4 \\",
                        '"',
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(sanitize_swiglu_fusion_script(script_path))

            content = script_path.read_text(encoding="utf-8")
            self.assertIn("--disable-bias-linear", content)
            self.assertIn("--no-bias-swiglu-fusion", content)

    def test_deepseek_low_memory_profile_reduces_large_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "deepseekv3.sh"
            script_path.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --num-layers 16 \\",
                        "    --hidden-size 5120 \\",
                        "    --ffn-hidden-size 12288 \\",
                        "    --num-attention-heads 128 \\",
                        "    --seq-length 4096 \\",
                        "    --max-position-embeddings 4096 \\",
                        "    --micro-batch-size 1 \\",
                        "    --global-batch-size 256 \\",
                        "    --pipeline-model-parallel-size 1 \\",
                        "    --tensor-model-parallel-size 1 \\",
                        "    --train-iters 1 \\",
                        '"',
                        'MLA_ARGS="',
                        "    --q-lora-rank 1536 \\",
                        "    --kv-lora-rank 512 \\",
                        '"',
                        'MOE_ARGS="',
                        "    --first-k-dense-replace 1 \\",
                        "    --moe-intermediate-size 1536 \\",
                        "    --moe-layer-freq 1 \\",
                        "    --moe-router-topk 6 \\",
                        "    --n-shared-experts 2 \\",
                        "    --num-experts 32 \\",
                        '"',
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(apply_deepseekv3_unified_low_memory_profile(script_path))

            content = script_path.read_text(encoding="utf-8")
            self.assertIn("--num-layers 8", content)
            self.assertIn("--hidden-size 1024", content)
            self.assertIn("--ffn-hidden-size 2048", content)
            self.assertIn("--num-attention-heads 16", content)
            self.assertIn("--q-lora-rank 192", content)
            self.assertIn("--kv-lora-rank 64", content)
            self.assertIn("--seq-length 1024", content)
            self.assertIn("--global-batch-size 8", content)
            self.assertIn("--num-experts 16", content)
            self.assertIn("--moe-router-topk 2", content)
            self.assertIn("--first-k-dense-replace 7", content)

    def test_mf_worker_ignores_nfs_temp_cleanup_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "worker_0.log"
            log_path.write_text(
                "\n".join(
                    [
                        "Traceback (most recent call last):",
                        '  File "/root/miniconda3/envs/mindformers/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap',
                        "    self.run()",
                        "SystemExit: 0",
                        "",
                        "During handling of the above exception, another exception occurred:",
                        "",
                        "Traceback (most recent call last):",
                        '  File "/root/miniconda3/envs/mindformers/lib/python3.11/multiprocessing/util.py", line 303, in _run_finalizers',
                        "    finalizer()",
                        '  File "/root/miniconda3/envs/mindformers/lib/python3.11/multiprocessing/util.py", line 136, in _remove_temp_dir',
                        "    rmtree(tempdir, onerror=onerror)",
                        "OSError: [Errno 16] Device or resource busy: '.nfs0000000008a6084a00000089'",
                        "2026-04-25 22:13:34,321 - INFO - Safetensors Convert Complete",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertIsNone(_contains_mf_fatal_text(log_path, ["Traceback (most recent call last)"]))


if __name__ == "__main__":
    unittest.main()
