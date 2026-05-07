import builtins
import os
import unittest
from unittest.mock import patch

from utils.runtime import cluster_runtime
from utils.task import task2, task3


_ORIGINAL_IMPORT = builtins.__import__


def _block_torch_import(name, *args, **kwargs):
    if name == "torch":
        raise ImportError("torch blocked for deterministic device detection test")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


class RuntimeDeviceDetectionTests(unittest.TestCase):
    def test_cluster_health_ignores_distributed_world_size_env(self) -> None:
        with (
            patch.dict(os.environ, {"WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8"}, clear=True),
            patch("builtins.__import__", side_effect=_block_torch_import),
            patch("utils.runtime.cluster_runtime.subprocess.run", side_effect=OSError),
        ):
            self.assertEqual(cluster_runtime.detect_visible_device_count(), 1)

    def test_task2_visible_device_inference_ignores_distributed_world_size_env(self) -> None:
        with (
            patch.dict(os.environ, {"WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8"}, clear=True),
            patch("builtins.__import__", side_effect=_block_torch_import),
            patch("utils.task.task2.subprocess.run", side_effect=OSError),
        ):
            self.assertEqual(task2._infer_visible_device_count(), 1)

    def test_task3_visible_device_inference_ignores_distributed_world_size_env(self) -> None:
        with (
            patch.dict(os.environ, {"WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8"}, clear=True),
            patch("builtins.__import__", side_effect=_block_torch_import),
            patch("utils.task.task3.subprocess.run", side_effect=OSError),
        ):
            self.assertEqual(task3._infer_visible_device_count(), 1)

    def test_task123_cluster_config_prefers_multinode_ssh_backend(self) -> None:
        cfg = cluster_runtime.parse_task123_cluster_config(
            {
                "MULTI_NODE": {
                    "ENABLED": True,
                    "MASTER_ADDR": "10.0.0.1",
                    "MASTER_PORT": 6123,
                    "LOCAL_NPUS_PER_NODE": 8,
                    "OTHER_NODES": [
                        {
                            "HOST": "worker@10.0.0.2",
                            "SSH_PORT": 2222,
                            "LMSV_PATH": "/data/lm-sv",
                            "PTA_NAME": "pta_env",
                            "MSA_NAME": "msa_env",
                            "MF_NAME": "mf_env",
                            "PTA_PATH": "/opt/pta",
                            "MSA_PATH": "/opt/msa",
                            "NPUS_PER_NODE": 4,
                        }
                    ],
                },
                "CLUSTER": {"ENABLED": True, "SLAVES": [{"HOST": "legacy", "PORT": 19001}]},
            }
        )

        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.backend, "ssh")
        self.assertEqual(cfg.master_addr, "10.0.0.1")
        self.assertEqual(cfg.master_port, 6123)
        self.assertEqual(cfg.local_npus_per_node, 8)
        self.assertEqual(cfg.slaves[0].host, "worker@10.0.0.2")
        self.assertEqual(cfg.slaves[0].ssh_port, 2222)
        self.assertEqual(cfg.slaves[0].npus_per_node, 4)
        self.assertEqual(cfg.slaves[0].mf_name, "mf_env")

    def test_task123_cluster_config_falls_back_to_legacy_cluster(self) -> None:
        cfg = cluster_runtime.parse_task123_cluster_config(
            {
                "MULTI_NODE": {"ENABLED": False},
                "CLUSTER": {
                    "ENABLED": True,
                    "MASTER_ADDR": "192.168.1.10",
                    "SLAVES": [{"ENDPOINT": "192.168.1.11:19002"}],
                },
            }
        )

        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.backend, "http")
        self.assertEqual(cfg.master_addr, "192.168.1.10")
        self.assertEqual(cfg.slaves[0].endpoint, "192.168.1.11:19002")


if __name__ == "__main__":
    unittest.main()
