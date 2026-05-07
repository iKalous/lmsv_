#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from utils.runtime.cluster_runtime import ClusterConfig, parse_cluster_config, run_slave_server


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.json"


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def main() -> int:
    config = _load_config()
    cluster = parse_cluster_config(config.get("CLUSTER"))
    if not cluster.enabled:
        cluster = ClusterConfig(
            enabled=False,
            listen_host="0.0.0.0",
            listen_port=19001,
        )
    run_slave_server(cluster)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
