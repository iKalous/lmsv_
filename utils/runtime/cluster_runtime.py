#!/usr/bin/env python3
"""Shared multi-node runtime for Task1/2/3."""

from __future__ import annotations

import http.client
import io
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
import base64
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from utils.task import runtime_helpers


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
CONFIG_PATH = REPO_ROOT / "config.json"
SLAVE_SESSION_ROOT = REPO_ROOT / "tmp" / "slave_sessions"
SSH_SESSION_ROOT = REPO_ROOT / "tmp" / "ssh_sessions"

TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in TRUE_VALUES


def _parse_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return parsed if parsed > 0 else int(default)


def _parse_optional_positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _is_loopback_master_addr(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"127.0.0.1", "localhost", "::1"} or text.startswith("127.")


def detect_visible_device_count() -> int:
    for env_name in (
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        raw = str(os.environ.get(env_name, "")).strip()
        if not raw:
            continue
        parts = [item.strip() for item in raw.split(",") if item.strip()]
        if parts:
            return len(parts)

    try:
        import torch

        if hasattr(torch, "npu") and callable(getattr(torch.npu, "device_count", None)):
            count = int(torch.npu.device_count())
            if count > 0:
                return count
        if callable(getattr(torch.cuda, "device_count", None)):
            count = int(torch.cuda.device_count())
            if count > 0:
                return count
    except Exception:
        pass

    for cmd in (
        ["bash", "-lc", "npu-smi info -l | grep -c '^\\s*NPU ID'"],
        ["bash", "-lc", "npu-smi info | grep -c '| NPU '"],
    ):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except OSError:
            continue
        if result.returncode != 0:
            continue
        count = _parse_optional_positive_int((result.stdout or "").strip())
        if count > 0:
            return count

    return 1


@dataclass(slots=True)
class ClusterNode:
    endpoint: str
    host: str
    port: int
    node_rank: int
    ssh_port: int = 22
    label: str = ""
    npus_per_node: int = 0
    lmsv_path: str = ""
    pta_name: str = ""
    msa_name: str = ""
    mf_name: str = ""
    pta_path: str = ""
    msa_path: str = ""
    has_container: bool = False
    container_name: str = ""
    health: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClusterConfig:
    enabled: bool = False
    backend: str = "http"
    master_addr: str = "127.0.0.1"
    master_port: int = 8118
    node_rank: int = 0
    listen_host: str = "0.0.0.0"
    listen_port: int = 19001
    request_timeout: int = 30
    session_timeout: int = 7200
    local_npus_per_node: int = 0
    ssh_bin: str = "ssh"
    rsync_bin: str = "rsync"
    slaves: list[ClusterNode] = field(default_factory=list)

    @property
    def nnodes(self) -> int:
        return 1 + len(self.slaves)

    @property
    def broadcast_master_addr(self) -> str:
        return self.master_addr

    @property
    def local_master_addr(self) -> str:
        if self.nnodes <= 1 and self.node_rank == 0 and not _is_loopback_master_addr(self.master_addr):
            return "127.0.0.1"
        return self.master_addr


@dataclass(slots=True)
class RemoteJobHandle:
    node: ClusterNode
    session_id: str
    job_id: str
    stage_name: str


@dataclass
class _DirectRemoteJob:
    handle: RemoteJobHandle
    process: subprocess.Popen
    log_path: Path
    timeout: int
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    exit_code: int | None = None
    status: str = "running"
    cancelled: bool = False


def parse_cluster_config(raw_config: Any) -> ClusterConfig:
    if not isinstance(raw_config, dict):
        return ClusterConfig(enabled=False)

    enabled = _to_bool(raw_config.get("ENABLED"))
    if not enabled:
        return ClusterConfig(enabled=False)

    listen_port = _parse_positive_int(raw_config.get("LISTEN_PORT"), 19001)
    request_timeout = _parse_positive_int(raw_config.get("REQUEST_TIMEOUT"), 30)
    session_timeout = _parse_positive_int(raw_config.get("SESSION_TIMEOUT"), 7200)
    local_npus_per_node = _parse_optional_positive_int(raw_config.get("LOCAL_NPUS_PER_NODE"))

    slaves: list[ClusterNode] = []
    raw_slaves = raw_config.get("SLAVES")
    if isinstance(raw_slaves, list):
        for index, item in enumerate(raw_slaves):
            if isinstance(item, str):
                endpoint = item.strip()
                node_rank = index + 1
                label = ""
                npus_per_node = 0
            elif isinstance(item, dict):
                endpoint = str(item.get("ENDPOINT", "")).strip()
                node_rank = int(item.get("NODE_RANK", index + 1))
                label = str(item.get("LABEL", "")).strip()
                npus_per_node = _parse_optional_positive_int(item.get("NPUS_PER_NODE"))
            else:
                continue

            if not endpoint or ":" not in endpoint:
                continue
            host, port_text = endpoint.rsplit(":", 1)
            port = _parse_positive_int(port_text, listen_port)
            slaves.append(
                ClusterNode(
                    endpoint=f"{host}:{port}",
                    host=host,
                    port=port,
                    node_rank=node_rank,
                    label=label or f"node-{node_rank}",
                    npus_per_node=npus_per_node,
                )
            )

    return ClusterConfig(
        enabled=enabled,
        backend="http",
        master_addr=str(raw_config.get("MASTER_ADDR", "127.0.0.1")).strip() or "127.0.0.1",
        master_port=_parse_positive_int(raw_config.get("MASTER_PORT"), 8118),
        node_rank=int(raw_config.get("NODE_RANK", 0)),
        listen_host=str(raw_config.get("LISTEN_HOST", "0.0.0.0")).strip() or "0.0.0.0",
        listen_port=listen_port,
        request_timeout=request_timeout,
        session_timeout=session_timeout,
        local_npus_per_node=local_npus_per_node,
        slaves=sorted(slaves, key=lambda item: item.node_rank),
    )


def parse_multinode_config(raw_config: Any, params: dict[str, Any] | None = None) -> ClusterConfig:
    """Parse Task4/Task5-style MULTI_NODE config for Task1/2/3 ssh execution."""
    if not isinstance(raw_config, dict):
        return ClusterConfig(enabled=False)

    enabled = _to_bool(raw_config.get("ENABLED"))
    if not enabled:
        return ClusterConfig(enabled=False)

    params = params if isinstance(params, dict) else {}
    request_timeout = _parse_positive_int(raw_config.get("REQUEST_TIMEOUT"), 30)
    session_timeout = _parse_positive_int(raw_config.get("SESSION_TIMEOUT"), 7200)
    local_npus_per_node = _parse_optional_positive_int(
        raw_config.get("LOCAL_NPUS_PER_NODE", params.get("TARGET_NPUS_PER_NODE"))
    )
    master_port = _parse_positive_int(
        raw_config.get("MASTER_PORT", params.get("TARGET_MASTER_PORT", 6000)),
        6000,
    )
    mf_name_default = str(params.get("MF_NAME", os.environ.get("MF_NAME", "mindf_py311"))).strip()

    slaves: list[ClusterNode] = []
    raw_nodes = raw_config.get("OTHER_NODES")
    if isinstance(raw_nodes, list):
        for index, item in enumerate(raw_nodes):
            if not isinstance(item, dict):
                continue
            host = str(item.get("HOST", "")).strip()
            lmsv_path = str(item.get("LMSV_PATH", "")).strip()
            pta_name = str(item.get("PTA_NAME", "")).strip()
            msa_name = str(item.get("MSA_NAME", "")).strip()
            pta_path = str(item.get("PTA_PATH", "")).strip()
            msa_path = str(item.get("MSA_PATH", "")).strip()
            has_container = _to_bool(item.get("HAS_CONTAINER"))
            container_name = str(item.get("CONTAINER_NAME", "")).strip()
            if not host or not lmsv_path:
                continue
            ssh_port = _parse_positive_int(item.get("SSH_PORT"), 22)
            node_rank = _parse_positive_int(item.get("NODE_RANK"), index + 1)
            slaves.append(
                ClusterNode(
                    endpoint=host,
                    host=host,
                    port=0,
                    ssh_port=ssh_port,
                    node_rank=node_rank,
                    label=str(item.get("LABEL", "")).strip() or f"node-{node_rank}",
                    npus_per_node=_parse_optional_positive_int(item.get("NPUS_PER_NODE")),
                    lmsv_path=lmsv_path,
                    pta_name=pta_name,
                    msa_name=msa_name,
                    mf_name=str(item.get("MF_NAME", mf_name_default)).strip() or mf_name_default,
                    pta_path=pta_path,
                    msa_path=msa_path,
                    has_container=has_container,
                    container_name=container_name,
                )
            )

    return ClusterConfig(
        enabled=True,
        backend="ssh",
        master_addr=str(raw_config.get("MASTER_ADDR", "127.0.0.1")).strip() or "127.0.0.1",
        master_port=master_port,
        node_rank=0,
        request_timeout=request_timeout,
        session_timeout=session_timeout,
        local_npus_per_node=local_npus_per_node,
        ssh_bin=str(os.environ.get("LMSV_SSH_BIN", raw_config.get("SSH_BIN", "ssh"))).strip() or "ssh",
        rsync_bin=str(os.environ.get("LMSV_RSYNC_BIN", raw_config.get("RSYNC_BIN", "rsync"))).strip() or "rsync",
        slaves=sorted(slaves, key=lambda item: item.node_rank),
    )


def parse_task123_cluster_config(params: dict[str, Any] | None) -> ClusterConfig:
    """Prefer Task5-style MULTI_NODE, then fall back to legacy CLUSTER."""
    params = params if isinstance(params, dict) else {}
    multi_cfg = parse_multinode_config(params.get("MULTI_NODE"), params)
    if multi_cfg.enabled:
        return multi_cfg
    return parse_cluster_config(params.get("CLUSTER"))


def load_local_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or CONFIG_PATH
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_local_runtime_settings(config: dict[str, Any] | None = None) -> dict[str, str]:
    data = config if isinstance(config, dict) else load_local_config()
    return {
        "PTA_NAME": str(data.get("PTA_NAME", os.environ.get("PTA_NAME", "mindspeed"))),
        "MSA_NAME": str(data.get("MSA_NAME", os.environ.get("MSA_NAME", "msadapter"))),
        "MF_NAME": str(data.get("MF_NAME", os.environ.get("MF_NAME", "mindf_py311"))),
        "PTA_PATH": str(data.get("PTA_PATH", os.environ.get("PTA_PATH", os.environ.get("PTAPATH", "")))),
        "MSA_PATH": str(data.get("MSA_PATH", os.environ.get("MSA_PATH", os.environ.get("MSAPATH", "")))),
        "MINDSPEED_MM_PATH": str(
            data.get(
                "MINDSPEED_MM_PATH",
                os.environ.get("MINDSPEED_MM_PATH", ""),
            )
        ),
    }


def resolve_cluster_from_config(config: dict[str, Any] | None = None) -> ClusterConfig:
    data = config if isinstance(config, dict) else load_local_config()
    raw = data.get("CLUSTER")
    return parse_cluster_config(raw)


class ClusterHttpError(RuntimeError):
    """Raised when the slave service returns a non-success response."""


class ClusterMaster:
    def __init__(self, config: ClusterConfig, log_info, log_warn, log_error) -> None:
        self.config = config
        self._log_info = log_info
        self._log_warn = log_warn
        self._log_error = log_error
        self._direct_jobs: dict[str, _DirectRemoteJob] = {}
        SSH_SESSION_ROOT.mkdir(parents=True, exist_ok=True)

    @property
    def _uses_ssh(self) -> bool:
        return self.config.backend == "ssh"

    def _remote_repo_root(self, node: ClusterNode) -> str:
        base = Path(node.lmsv_path).expanduser()
        try:
            rel = REPO_ROOT.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = Path(REPO_ROOT.name)
        return (base / rel).as_posix()

    def _remote_shell_command(self, node: ClusterNode, shell_body: str) -> str:
        remote_body = f"set -e -o pipefail\n{shell_body}"
        if node.has_container:
            if not node.container_name:
                raise RuntimeError(f"[{node.host}] HAS_CONTAINER=true 但缺少 CONTAINER_NAME")
            return f"docker exec {shlex.quote(node.container_name)} bash -lc {shlex.quote(remote_body)}"
        return f"bash -lc {shlex.quote(remote_body)}"

    def _ssh_command(self, node: ClusterNode, shell_body: str) -> list[str]:
        return [
            self.config.ssh_bin,
            "-p",
            str(node.ssh_port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            str(node.host),
            self._remote_shell_command(node, shell_body),
        ]

    def _run_ssh_capture(self, node: ClusterNode, shell_body: str, *, timeout: int | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            self._ssh_command(node, shell_body),
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    def _runtime_settings_for_node(self, node: ClusterNode) -> dict[str, str]:
        defaults = resolve_local_runtime_settings()
        return {
            "PTA_NAME": node.pta_name or defaults["PTA_NAME"],
            "MSA_NAME": node.msa_name or defaults["MSA_NAME"],
            "MF_NAME": node.mf_name or defaults["MF_NAME"],
            "PTA_PATH": node.pta_path or defaults["PTA_PATH"],
            "MSA_PATH": node.msa_path or defaults["MSA_PATH"],
            "MINDSPEED_MM_PATH": defaults["MINDSPEED_MM_PATH"],
        }

    def _request(
        self,
        node: ClusterNode,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        expect_json: bool = True,
    ) -> Any:
        conn = http.client.HTTPConnection(node.host, node.port, timeout=self.config.request_timeout)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            payload = response.read()
        finally:
            conn.close()

        if response.status < 200 or response.status >= 300:
            message = payload.decode("utf-8", errors="ignore")
            raise ClusterHttpError(f"{node.endpoint} {method} {path} -> {response.status}: {message}")

        if not expect_json:
            return payload
        if not payload:
            return {}
        return json.loads(payload.decode("utf-8"))

    def preflight(self) -> dict[int, dict[str, Any]]:
        if self.config.slaves and _is_loopback_master_addr(self.config.master_addr):
            raise RuntimeError(
                "多机 MASTER_ADDR 不能是 localhost/127.*：需要填写远端节点可访问的主节点IP"
            )
        if self._uses_ssh:
            if not self.config.slaves:
                raise RuntimeError("MULTI_NODE.ENABLED=true 时，必须至少配置一个 OTHER_NODES 节点")
            resolved_ssh = shutil.which(self.config.ssh_bin)
            if not resolved_ssh:
                raise RuntimeError(f"多机模式缺少 SSH 客户端命令：{self.config.ssh_bin}")
            resolved_rsync = shutil.which(self.config.rsync_bin)
            if not resolved_rsync:
                raise RuntimeError(f"多机模式缺少 rsync 命令：{self.config.rsync_bin}")
            self.config.ssh_bin = resolved_ssh
            self.config.rsync_bin = resolved_rsync
            return self._preflight_ssh()

        results: dict[int, dict[str, Any]] = {}
        for node in self.config.slaves:
            payload = self._request(node, "GET", "/health", expect_json=True)
            node.health = payload
            results[node.node_rank] = payload
            self._log_info(
                f"[集群] 从机就绪 | rank={node.node_rank} | endpoint={node.endpoint} | "
                f"hostname={payload.get('hostname', '-')} | visible_devices={payload.get('visible_devices', '-')}"
            )
        return results

    def _preflight_ssh(self) -> dict[int, dict[str, Any]]:
        results: dict[int, dict[str, Any]] = {}
        for node in self.config.slaves:
            if node.has_container and not node.container_name:
                raise RuntimeError(f"[{node.host}] HAS_CONTAINER=true 但缺少 CONTAINER_NAME")
            missing = [
                name
                for name, value in (
                    ("LMSV_PATH", node.lmsv_path),
                    ("PTA_NAME", node.pta_name),
                    ("MSA_NAME", node.msa_name),
                    ("PTA_PATH", node.pta_path),
                    ("MSA_PATH", node.msa_path),
                )
                if not str(value or "").strip()
            ]
            if missing:
                raise RuntimeError(f"[{node.host}] MULTI_NODE 节点配置不完整，缺少: {', '.join(missing)}")

            runner = f"""
cd {shlex.quote(self._remote_repo_root(node))}
LMSV_REMOTE_PYTHON="${{LMSV_REMOTE_PYTHON:-python}}"
if ! command -v "$LMSV_REMOTE_PYTHON" >/dev/null 2>&1; then
  LMSV_REMOTE_PYTHON=python3
fi
"$LMSV_REMOTE_PYTHON" - <<'PY'
import json
import os
from utils.runtime.cluster_runtime import detect_visible_device_count
print(json.dumps({{
    "hostname": os.uname().nodename,
    "visible_devices": detect_visible_device_count(),
}}, ensure_ascii=False))
PY
"""
            result = self._run_ssh_capture(node, runner, timeout=self.config.request_timeout)
            if result.returncode != 0:
                raise RuntimeError(
                    f"[{node.host}] SSH 预检失败，请确认主节点可免密登录且远端 LMSV_PATH 正确: "
                    f"{(result.stderr or result.stdout or '').strip()}"
                )
            try:
                payload = json.loads((result.stdout or "{}").strip().splitlines()[-1])
            except Exception:
                payload = {"hostname": "-", "visible_devices": 0}
            node.health = payload
            results[node.node_rank] = payload
            self._log_info(
                f"[集群] SSH节点就绪 | rank={node.node_rank} | host={node.host} | "
                f"hostname={payload.get('hostname', '-')} | visible_devices={payload.get('visible_devices', '-')}"
            )
        return results

    def local_worker_count(self) -> int:
        configured = _parse_optional_positive_int(self.config.local_npus_per_node)
        if configured > 0:
            return configured
        return detect_visible_device_count()

    def slave_worker_count(self, node: ClusterNode) -> int:
        configured = _parse_optional_positive_int(node.npus_per_node)
        if configured > 0:
            return configured
        health_value = _parse_optional_positive_int((node.health or {}).get("visible_devices"))
        if health_value > 0:
            return health_value
        return 1

    def total_workers(self) -> int:
        total = self.local_worker_count()
        for node in self.config.slaves:
            total += self.slave_worker_count(node)
        return max(1, total)

    def prepare_session(self, session_id: str) -> None:
        if self._uses_ssh:
            (SSH_SESSION_ROOT / session_id / "job_logs").mkdir(parents=True, exist_ok=True)
            return
        payload = json.dumps({"session_id": session_id}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        for node in self.config.slaves:
            self._request(node, "POST", f"/sessions/{session_id}/prepare", body=payload, headers=headers)

    def cleanup_session(self, session_id: str) -> None:
        if self._uses_ssh:
            for job_id, job in list(self._direct_jobs.items()):
                if job.handle.session_id == session_id:
                    if job.status == "running":
                        self.cancel_job(job.handle)
                    self._direct_jobs.pop(job_id, None)
            shutil.rmtree(SSH_SESSION_ROOT / session_id, ignore_errors=True)
            return
        for node in self.config.slaves:
            try:
                self._request(node, "POST", f"/sessions/{session_id}/cleanup", body=b"{}", headers={"Content-Type": "application/json"})
            except Exception as exc:
                self._log_warn(f"[集群] 清理从机会话失败 | node={node.endpoint} | {exc}")

    def upload_paths(
        self,
        node: ClusterNode,
        session_id: str,
        items: list[tuple[Path, str]],
    ) -> None:
        if not items:
            return
        if self._uses_ssh:
            self._upload_paths_ssh(node, items)
            return
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as handle:
            with tarfile.open(handle.name, "w:gz") as archive:
                for src_path, arcname in items:
                    if not src_path.exists():
                        continue
                    archive.add(str(src_path), arcname=arcname)
            payload = Path(handle.name).read_bytes()
        headers = {"Content-Type": "application/gzip", "Content-Length": str(len(payload))}
        self._request(node, "POST", f"/sessions/{session_id}/bundle", body=payload, headers=headers)

    def _upload_paths_ssh(self, node: ClusterNode, items: list[tuple[Path, str]]) -> None:
        remote_root = self._remote_repo_root(node)
        for src_path, arcname in items:
            src = Path(src_path)
            if not src.exists():
                continue
            rel = str(arcname).strip().lstrip("/")
            if not rel:
                continue
            remote_path = (Path(remote_root) / rel).as_posix()
            remote_parent = Path(remote_path).parent.as_posix()
            mkdir = [
                self.config.ssh_bin,
                "-p",
                str(node.ssh_port),
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                str(node.host),
                f"mkdir -p {shlex.quote(remote_parent)}",
            ]
            mkdir_result = subprocess.run(mkdir, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
            if mkdir_result.returncode != 0:
                raise RuntimeError(f"[{node.host}] 创建远端目录失败: {remote_parent} | {mkdir_result.stderr}")
            src_arg = src.as_posix() + ("/" if src.is_dir() else "")
            dst_arg = f"{node.host}:{remote_path}{'/' if src.is_dir() else ''}"
            result = subprocess.run(
                [
                    self.config.rsync_bin,
                    "-az",
                    "-e",
                    f"{self.config.ssh_bin} -p {node.ssh_port} -o BatchMode=yes -o StrictHostKeyChecking=no",
                    src_arg,
                    dst_arg,
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"[{node.host}] rsync 上传失败: {src} -> {remote_path} | {result.stderr}")

    def start_job(
        self,
        node: ClusterNode,
        session_id: str,
        job_type: str,
        payload: dict[str, Any],
        *,
        stage_name: str,
    ) -> RemoteJobHandle:
        if self._uses_ssh:
            return self._start_job_ssh(node, session_id, job_type, payload, stage_name=stage_name)

        body = json.dumps({"job_type": job_type, "payload": payload}, ensure_ascii=False).encode("utf-8")
        response = self._request(
            node,
            "POST",
            f"/sessions/{session_id}/jobs",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        job_id = str(response.get("job_id", "")).strip()
        if not job_id:
            raise ClusterHttpError(f"{node.endpoint} 未返回 job_id")
        return RemoteJobHandle(node=node, session_id=session_id, job_id=job_id, stage_name=stage_name)

    def _start_job_ssh(
        self,
        node: ClusterNode,
        session_id: str,
        job_type: str,
        payload: dict[str, Any],
        *,
        stage_name: str,
    ) -> RemoteJobHandle:
        job_id = uuid.uuid4().hex
        handle = RemoteJobHandle(node=node, session_id=session_id, job_id=job_id, stage_name=stage_name)
        payload_for_remote = dict(payload)
        payload_for_remote["_session_id"] = session_id
        runner_payload = {
            "job_type": job_type,
            "payload": payload_for_remote,
            "runtime_settings": self._runtime_settings_for_node(node),
        }
        encoded = base64.b64encode(json.dumps(runner_payload, ensure_ascii=False).encode("utf-8")).decode("ascii")
        shell_body = (
            f"cd {shlex.quote(self._remote_repo_root(node))}\n"
            'LMSV_REMOTE_PYTHON="${LMSV_REMOTE_PYTHON:-python}"\n'
            'if ! command -v "$LMSV_REMOTE_PYTHON" >/dev/null 2>&1; then LMSV_REMOTE_PYTHON=python3; fi\n'
            f'"$LMSV_REMOTE_PYTHON" -m utils.runtime.cluster_runtime direct-job {shlex.quote(encoded)}'
        )
        command = self._ssh_command(node, shell_body)
        log_path = SSH_SESSION_ROOT / session_id / "job_logs" / f"{stage_name}_{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle_fp:
            handle_fp.write(f"[START] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
            handle_fp.write("[COMMAND]\n")
            handle_fp.write(" ".join(shlex.quote(part) for part in command))
            handle_fp.write("\n\n")
            handle_fp.flush()
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        log_handle.close()
        self._direct_jobs[job_id] = _DirectRemoteJob(
            handle=handle,
            process=process,
            log_path=log_path,
            timeout=_job_timeout(payload),
        )
        return handle

    def _update_direct_job_state(self, job: _DirectRemoteJob) -> _DirectRemoteJob:
        if job.status in {"success", "failed", "timeout", "cancelled"}:
            return job
        elapsed = time.time() - job.started_at
        if job.timeout > 0 and elapsed > job.timeout + 60:
            try:
                os.killpg(job.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            job.status = "timeout"
            job.exit_code = 124
            job.ended_at = time.time()
            return job
        return_code = job.process.poll()
        if return_code is None:
            return job
        job.exit_code = int(return_code)
        job.ended_at = time.time()
        if job.cancelled:
            job.status = "cancelled"
        elif return_code == 0:
            job.status = "success"
        elif return_code == 124:
            job.status = "timeout"
        else:
            job.status = "failed"
        return job

    def job_state(self, handle: RemoteJobHandle) -> dict[str, Any]:
        if self._uses_ssh:
            job = self._direct_jobs.get(handle.job_id)
            if job is None:
                return {"job_id": handle.job_id, "status": "failed", "error": "job not found"}
            self._update_direct_job_state(job)
            return {
                "job_id": handle.job_id,
                "status": job.status,
                "exit_code": job.exit_code,
                "timed_out": job.status == "timeout",
                "cancelled": job.cancelled,
                "error": "",
                "started_at": job.started_at,
                "ended_at": job.ended_at,
                "process_pid": job.process.pid,
            }
        return self._request(
            handle.node,
            "GET",
            f"/sessions/{handle.session_id}/jobs/{handle.job_id}",
            expect_json=True,
        )

    def cancel_job(self, handle: RemoteJobHandle) -> None:
        if self._uses_ssh:
            job = self._direct_jobs.get(handle.job_id)
            if job is None:
                return
            job.cancelled = True
            try:
                os.killpg(job.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self._update_direct_job_state(job)
            return
        try:
            self._request(
                handle.node,
                "POST",
                f"/sessions/{handle.session_id}/jobs/{handle.job_id}/cancel",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )
        except Exception as exc:
            self._log_warn(
                f"[集群] 取消远端任务失败 | node={handle.node.endpoint} | job={handle.job_id} | {exc}"
            )

    def download_items(
        self,
        node: ClusterNode,
        session_id: str,
        items: list[dict[str, Any]],
        target_dir: Path,
    ) -> None:
        if not items:
            return
        if self._uses_ssh:
            self._download_items_ssh(node, items, target_dir)
            return
        body = json.dumps({"items": items}, ensure_ascii=False).encode("utf-8")
        payload = self._request(
            node,
            "POST",
            f"/sessions/{session_id}/collect",
            body=body,
            headers={"Content-Type": "application/json"},
            expect_json=False,
        )
        if not payload:
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            archive.extractall(path=target_dir)

    def _download_items_ssh(self, node: ClusterNode, items: list[dict[str, Any]], target_dir: Path) -> None:
        remote_root = self._remote_repo_root(node)
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            if not isinstance(item, dict):
                continue
            rel_path = str(item.get("path", "")).strip().lstrip("/")
            if not rel_path:
                continue
            flatten = _to_bool(item.get("flatten"))
            remote_path = (Path(remote_root) / rel_path).as_posix()
            if flatten:
                remote_arg = f"{node.host}:{remote_path}/"
                local_arg = target_dir.as_posix() + "/"
            else:
                remote_arg = f"{node.host}:{remote_path}"
                local_arg = (target_dir / Path(rel_path).name).as_posix()
            result = subprocess.run(
                [
                    self.config.rsync_bin,
                    "-az",
                    "-e",
                    f"{self.config.ssh_bin} -p {node.ssh_port} -o BatchMode=yes -o StrictHostKeyChecking=no",
                    remote_arg,
                    local_arg,
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                self._log_warn(f"[{node.host}] rsync 回收产物失败: {rel_path} | {result.stderr}")

    def stream_job_log(
        self,
        handle: RemoteJobHandle,
        dst_path: Path,
        stop_event: threading.Event,
        *,
        poll_interval: float = 2.0,
    ) -> threading.Thread:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        def _worker() -> None:
            offset = 0
            while not stop_event.is_set():
                try:
                    if self._uses_ssh:
                        job = self._direct_jobs.get(handle.job_id)
                        if job is None or not job.log_path.exists():
                            payload = {"chunk": "", "offset": offset, "finished": False}
                        else:
                            text = job.log_path.read_text(encoding="utf-8", errors="ignore")
                            payload = {
                                "chunk": text[offset:],
                                "offset": len(text),
                                "finished": self._update_direct_job_state(job).status in {"success", "failed", "timeout", "cancelled"},
                            }
                    else:
                        payload = self._request(
                            handle.node,
                            "GET",
                            f"/sessions/{handle.session_id}/jobs/{handle.job_id}/logs?offset={offset}",
                            expect_json=True,
                        )
                except Exception as exc:
                    self._log_warn(
                        f"[集群] 远端日志轮询失败 | node={handle.node.endpoint} | job={handle.job_id} | {exc}"
                    )
                    time.sleep(poll_interval)
                    continue

                chunk = str(payload.get("chunk", ""))
                if chunk:
                    with dst_path.open("a", encoding="utf-8") as handle_fp:
                        handle_fp.write(chunk)
                    offset = int(payload.get("offset", offset))
                if payload.get("finished"):
                    break
                time.sleep(poll_interval)

        thread = threading.Thread(target=_worker, name=f"cluster-log-{handle.node.node_rank}", daemon=True)
        thread.start()
        return thread

    def job_log_text(self, handle: RemoteJobHandle) -> str:
        if self._uses_ssh:
            job = self._direct_jobs.get(handle.job_id)
            if job is None:
                return "<failed to fetch remote log: job not found>"
            if not job.log_path.exists():
                return ""
            return job.log_path.read_text(encoding="utf-8", errors="ignore")
        try:
            payload = self._request(
                handle.node,
                "GET",
                f"/sessions/{handle.session_id}/jobs/{handle.job_id}/logs?offset=0",
                expect_json=True,
            )
        except Exception as exc:
            return f"<failed to fetch remote log: {exc}>"
        return str(payload.get("chunk", ""))

    def job_log_tail(self, handle: RemoteJobHandle, *, max_chars: int = 12000) -> str:
        text = self.job_log_text(handle)
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]


@dataclass
class _SlaveJob:
    job_id: str
    session_id: str
    job_type: str
    payload: dict[str, Any]
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    ended_at: float = 0.0
    log_path: Path | None = None
    process_pid: int = 0
    exit_code: int | None = None
    error: str = ""
    timed_out: bool = False
    cancelled: bool = False
    thread: threading.Thread | None = None


@dataclass
class _SlaveSession:
    session_id: str
    root: Path
    project_tmp_root: Path
    jobs: dict[str, _SlaveJob] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    current_job_id: str = ""
    last_touch: float = field(default_factory=time.time)


class _SlaveState:
    def __init__(self) -> None:
        self.sessions: dict[str, _SlaveSession] = {}
        self.lock = threading.Lock()
        SLAVE_SESSION_ROOT.mkdir(parents=True, exist_ok=True)

    def ensure_session(self, session_id: str) -> _SlaveSession:
        with self.lock:
            session = self.sessions.get(session_id)
            if session is not None:
                session.last_touch = time.time()
                return session

            root = (SLAVE_SESSION_ROOT / session_id).resolve()
            project_tmp_root = root / "project_tmp"
            project_tmp_root.mkdir(parents=True, exist_ok=True)
            runtime_helpers.configure_project_tmp_env(project_tmp_root)
            session = _SlaveSession(
                session_id=session_id,
                root=root,
                project_tmp_root=project_tmp_root,
            )
            self.sessions[session_id] = session
            return session

    def cleanup_session(self, session_id: str) -> None:
        with self.lock:
            session = self.sessions.pop(session_id, None)
        if session is None:
            return
        for job in list(session.jobs.values()):
            if job.status == "running" and job.process_pid > 0:
                try:
                    os.killpg(job.process_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        shutil.rmtree(session.root, ignore_errors=True)


_SLAVE_STATE = _SlaveState()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _bytes_response(handler: BaseHTTPRequestHandler, status: int, payload: bytes, content_type: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = _parse_optional_positive_int(handler.headers.get("Content-Length"))
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    if not raw:
        return {}
    data = json.loads(raw.decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _write_job_log_header(log_path: Path, command_text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[START] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        handle.write("[COMMAND]\n")
        handle.write(command_text)
        handle.write("\n\n")


def _append_job_log_footer(log_path: Path, return_code: int, *, timed_out: bool = False, cancelled: bool = False) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[END] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        if timed_out:
            handle.write("[TIMED_OUT] 1\n")
        if cancelled:
            handle.write("[CANCELLED] 1\n")
        handle.write(f"[RETURNCODE] {return_code}\n")


def _job_timeout(payload: dict[str, Any], default_timeout: int = 7200) -> int:
    return _parse_positive_int(payload.get("timeout"), default_timeout)


def _build_cleanup_block(paths: list[str] | tuple[str, ...] | None) -> str:
    if not paths:
        return ""

    lines: list[str] = []
    seen: set[str] = set()
    for path_value in paths:
        path_text = str(path_value or "").strip()
        if not path_text or path_text in seen:
            continue
        seen.add(path_text)
        lines.append(f"rm -rf {shlex.quote(path_text)}")
    return "\n".join(lines)


def _build_task1_run_script_command(payload: dict[str, Any], runtime_settings: dict[str, str]) -> str:
    from utils.task import task1

    script_rel = str(payload["script_rel"])
    env_type = int(payload["env_type"])
    csv_rel = str(payload.get("csv_rel", "")).strip()
    cleanup_paths = [str(item) for item in payload.get("cleanup_paths", []) if str(item).strip()]
    if csv_rel:
        cleanup_paths.append(csv_rel)
    if env_type == 2:
        cleanup_paths.append("msrun_log")
    cleanup_block = _build_cleanup_block(cleanup_paths)
    runtime_hook_env = task1._build_runtime_hook_env(csv_rel if csv_rel else None)
    pretrain_resolver = task1._build_external_pretrain_resolver(env_type)
    sigterm_shield = runtime_helpers.build_sigterm_shield_block()

    if env_type == 1:
        env_name = runtime_settings["PTA_NAME"]
        pta_path = runtime_settings["PTA_PATH"]
        return f"""
{task1.build_conda_activate_block(env_name, load_ascend=True)}
export PTAPATH={task1.shlex.quote(pta_path)}
source scripts/envset/pta.sh
{sigterm_shield}
{cleanup_block}
{runtime_hook_env}
{pretrain_resolver}
bash -x -e -o pipefail {task1.shlex.quote(script_rel)}
"""
    if env_type == 2:
        env_name = runtime_settings["MSA_NAME"]
        msa_path = runtime_settings["MSA_PATH"]
        return f"""
{task1.build_conda_activate_block(env_name, load_ascend=True)}
export MSAPATH={task1.shlex.quote(msa_path)}
source scripts/envset/msa.sh
{cleanup_block}
{runtime_hook_env}
{pretrain_resolver}
bash -x -e -o pipefail {task1.shlex.quote(script_rel)}
"""
    raise RuntimeError(f"unsupported env_type for task1 script run: {env_type}")


def _build_task1_mf_training_command(payload: dict[str, Any], runtime_settings: dict[str, str]) -> str:
    from utils.task import task1

    yaml_rel = str(payload["yaml_rel"])
    csv_rel = str(payload.get("csv_rel", "")).strip()
    local_workers = int(payload["local_workers"])
    total_workers = int(payload["total_workers"])
    master_addr = str(payload["master_addr"])
    master_port = int(payload["master_port"])
    node_rank = int(payload["node_rank"])
    cleanup_paths = [str(item) for item in payload.get("cleanup_paths", []) if str(item).strip()]

    csv_export = "unset LMSV_MF_TRAINING_LOG_CSV"
    if csv_rel:
        cleanup_paths.append(csv_rel)
        csv_export = f"export LMSV_MF_TRAINING_LOG_CSV={task1.shlex.quote(task1._to_abs_path(csv_rel))}"
    cleanup_paths.append("msrun_log")
    cleanup_block = _build_cleanup_block(cleanup_paths)

    return f"""
{task1.build_conda_activate_block(runtime_settings["MF_NAME"], load_ascend=True)}
export PYTHONPATH={task1.shlex.quote(str(task1.LMSV_ROOT))}:${{PYTHONPATH:-}}
export LMSV_MSRUN_MASTER_PORT={master_port}
export LMSV_MF_WORKER_NUM={total_workers}
export LMSV_MF_LOCAL_WORKER={local_workers}
export LMSV_MF_MASTER_ADDR={task1.shlex.quote(master_addr)}
export LMSV_MF_MASTER_PORT={master_port}
export LMSV_MF_NODE_RANK={node_rank}
{cleanup_block}
{csv_export}
bash -x {task1.shlex.quote(task1.repo_rel(task1.RUNTIME_SCRIPT_DIR / "mf_start.sh"))} {task1.shlex.quote(task1._to_abs_path(yaml_rel))} {local_workers}
"""


def _build_task1_mf_prepare_command(payload: dict[str, Any], runtime_settings: dict[str, str]) -> str:
    from utils.task import task1

    yaml_rel = str(payload["yaml_rel"])
    ckpt_load_dir = str(payload["ckpt_load_dir"])
    ckpt_save_dir = str(payload["ckpt_save_dir"])
    model_name = str(payload["model_name"])
    tp = int(payload["tp"])
    pp = int(payload["pp"])
    ep = int(payload["ep"])
    enable_weight_load = _to_bool(payload.get("enable_weight_load"))
    enable_weight_convert = _to_bool(payload.get("enable_weight_convert"))
    enable_weight_load_literal = "True" if enable_weight_load else "False"

    convert_entry, convert_shell = task1.resolve_weight_convert_assets()
    if enable_weight_load and not enable_weight_convert and not Path(ckpt_save_dir).exists():
        raise RuntimeError(f"MF ckpt目录不存在: {ckpt_save_dir}")

    if enable_weight_load and enable_weight_convert:
        if not convert_entry or not convert_shell:
            raise RuntimeError("未找到 PTA 权重转换入口 convert_ckpt.py")

        model_name_for_convert = task1.resolve_task1_weight_convert_model_alias(model_name)
        convert_block = f"""
{task1.build_conda_activate_block(runtime_settings["PTA_NAME"], load_ascend=True)}
export PTAPATH={task1.shlex.quote(runtime_settings["PTA_PATH"])}
source scripts/envset/pta.sh
export LMSV_CONVERT_CKPT_ENTRY={task1.shlex.quote(convert_entry)}
rm -rf {task1.shlex.quote(ckpt_save_dir)}
mkdir -p {task1.shlex.quote(str(Path(ckpt_save_dir).parent))}
bash -x {task1.shlex.quote(convert_shell)} \\
  --load-model-type mg \\
  --save-model-type hf \\
  --load-dir {task1.shlex.quote(ckpt_load_dir)} \\
  --save-dir {task1.shlex.quote(ckpt_save_dir)} \\
  --model-type-hf {task1.shlex.quote(model_name_for_convert)} \\
  --target-tensor-parallel-size {tp} \\
  --target-pipeline-parallel-size {pp} \\
  --target-expert-parallel-size {ep}
"""
    else:
        convert_block = "true"

    yaml_prepare = f"""
python - <<'PY'
from utils.task import task1
import sys
ok = task1.update_mf_yaml_load_checkpoint({yaml_rel!r}, {ckpt_save_dir!r}) if {enable_weight_load_literal} else task1.disable_mf_yaml_load_checkpoint({yaml_rel!r})
sys.exit(0 if ok else 1)
PY
"""
    return f"{convert_block}\n{yaml_prepare}"


def _configure_task2_for_job(payload: dict[str, Any], runtime_settings: dict[str, str]) -> None:
    from utils.task import task2

    task2.configure_project_tmp_env()
    task2.Config.PTA_ENV = runtime_settings["PTA_NAME"]
    task2.Config.MSA_ENV = runtime_settings["MSA_NAME"]
    task2.Config.MF_ENV = runtime_settings["MF_NAME"]
    task2.Config.PTA_MAX_RUNTIME = int(payload["pta_max_runtime"])
    task2.Config.MSA_MAX_RUNTIME = int(payload["msa_max_runtime"])
    task2.Config.LOG_INIT_WAIT = int(payload["log_init_wait"])
    task2.Config.LOG_STABLE_THRESHOLD = int(payload["log_stable_threshold"])
    task2.Config.TARGET_TENSOR_PARALLEL_SIZE = int(payload["tp"])
    task2.Config.TARGET_PIPELINE_PARALLEL_SIZE = int(payload["pp"])
    task2.Config.TARGET_EXPERT_PARALLEL_SIZE = int(payload["ep"])
    task2.Config.TARGET_NPUS_PER_NODE = int(payload["local_workers"])
    task2.Config.TARGET_WORLD_SIZE = int(payload["total_workers"])
    task2.Config.TARGET_NNODES = int(payload["nnodes"])
    task2.Config.TARGET_NODE_RANK = int(payload["node_rank"])
    task2.Config.TARGET_MASTER_ADDR = str(payload["master_addr"])
    task2.Config.TARGET_MASTER_PORT = int(payload["master_port"])
    task2.Config.MSA_MONITOR_LOG = task2.resolve_msa_monitor_log()


def _configure_task3_for_job(payload: dict[str, Any], runtime_settings: dict[str, str]) -> None:
    from utils.task import task3

    task3.configure_project_tmp_env()
    task3.Config.PTA_ENV = runtime_settings["PTA_NAME"]
    task3.Config.MSA_ENV = runtime_settings["MSA_NAME"]
    task3.Config.MF_ENV = runtime_settings["MF_NAME"]
    task3.Config.PTA_MAX_RUNTIME = int(payload["pta_max_runtime"])
    task3.Config.MSA_MAX_RUNTIME = int(payload["msa_max_runtime"])
    task3.Config.LOG_INIT_WAIT = int(payload["log_init_wait"])
    task3.Config.LOG_STABLE_THRESHOLD = int(payload["log_stable_threshold"])
    task3.Config.TARGET_TENSOR_PARALLEL_SIZE = int(payload["tp"])
    task3.Config.TARGET_PIPELINE_PARALLEL_SIZE = int(payload["pp"])
    task3.Config.TARGET_EXPERT_PARALLEL_SIZE = int(payload["ep"])
    task3.Config.TARGET_NPUS_PER_NODE = int(payload["local_workers"])
    task3.Config.TARGET_WORLD_SIZE = int(payload["total_workers"])
    task3.Config.TARGET_NNODES = int(payload["nnodes"])
    task3.Config.TARGET_NODE_RANK = int(payload["node_rank"])
    task3.Config.TARGET_MASTER_ADDR = str(payload["master_addr"])
    task3.Config.TARGET_MASTER_PORT = int(payload["master_port"])
    task3.Config.MSA_MONITOR_LOG = task3.resolve_msa_monitor_log()


def _build_task2_command(job_type: str, payload: dict[str, Any], runtime_settings: dict[str, str]) -> str:
    from utils.task import task2

    _configure_task2_for_job(payload, runtime_settings)
    pta_path = runtime_settings["PTA_PATH"]
    msa_path = runtime_settings["MSA_PATH"]
    cleanup_paths = []
    step_log_csv_path = str(payload.get("step_log_csv_path", "")).strip()
    if step_log_csv_path:
        cleanup_paths.append(step_log_csv_path)
    if job_type == "task2_msa_verify":
        cleanup_paths.append("msrun_log")
    if job_type == "task2_pta_verify" and str(payload.get("shared_mode", "")).strip().lower() == "save":
        cleanup_paths.append(str(payload.get("shared_weight_path", "")).strip())

    if job_type == "task2_pta_verify":
        command_text = task2.build_pta_verify_stage_cmd(
            int(payload["iter_num"]),
            str(payload["mutate_args"]),
            runtime_settings["PTA_NAME"],
            pta_path,
            str(payload["shared_weight_path"]),
            str(payload["shared_mode"]),
            int(payload["train_iters"]),
            step_log_csv_path=payload.get("step_log_csv_path"),
        )
        return f"{_build_cleanup_block(cleanup_paths)}\n{command_text}"
    if job_type == "task2_msa_verify":
        command_text = task2.build_msa_verify_load_cmd(
            int(payload["iter_num"]),
            str(payload["mutate_args"]),
            runtime_settings["MSA_NAME"],
            msa_path,
            str(payload["shared_weight_path"]),
            int(payload["train_iters"]),
            step_log_csv_path=payload.get("step_log_csv_path"),
        )
        return f"{_build_cleanup_block(cleanup_paths)}\n{command_text}"
    raise RuntimeError(f"unsupported task2 job_type: {job_type}")


def _build_task3_command(job_type: str, payload: dict[str, Any], runtime_settings: dict[str, str]) -> str:
    from utils.task import task3

    _configure_task3_for_job(payload, runtime_settings)
    pta_path = runtime_settings["PTA_PATH"]
    msa_path = runtime_settings["MSA_PATH"]
    cleanup_paths = []
    step_log_csv_path = str(payload.get("step_log_csv_path", "")).strip()
    if step_log_csv_path:
        cleanup_paths.append(step_log_csv_path)
    if job_type == "task3_msa_verify":
        cleanup_paths.append("msrun_log")
    if job_type == "task3_pta_verify" and str(payload.get("shared_mode", "")).strip().lower() == "save":
        cleanup_paths.append(str(payload.get("shared_weight_path", "")).strip())
    if job_type == "task3_pta_verify":
        command_text = task3.build_pta_verify_stage_cmd(
            int(payload["iter_num"]),
            str(payload["mutate_args"]),
            str(payload["load_path"]),
            runtime_settings["PTA_NAME"],
            pta_path,
            str(payload["shared_weight_path"]),
            str(payload["shared_mode"]),
            int(payload["train_iters"]),
            step_log_csv_path=payload.get("step_log_csv_path"),
        )
        return f"{_build_cleanup_block(cleanup_paths)}\n{command_text}"
    if job_type == "task3_msa_verify":
        command_text = task3.build_msa_verify_load_cmd(
            int(payload["iter_num"]),
            str(payload["mutate_args"]),
            str(payload["load_path"]),
            runtime_settings["MSA_NAME"],
            msa_path,
            str(payload["shared_weight_path"]),
            int(payload["train_iters"]),
            step_log_csv_path=payload.get("step_log_csv_path"),
        )
        return f"{_build_cleanup_block(cleanup_paths)}\n{command_text}"
    raise RuntimeError(f"unsupported task3 job_type: {job_type}")


def _build_command_for_job(job_type: str, payload: dict[str, Any], runtime_settings: dict[str, str]) -> str:
    if job_type == "task1_run_script":
        return _build_task1_run_script_command(payload, runtime_settings)
    if job_type == "task1_mf_train":
        return _build_task1_mf_training_command(payload, runtime_settings)
    if job_type == "task1_mf_prepare":
        return _build_task1_mf_prepare_command(payload, runtime_settings)
    if job_type in {"task2_pta_verify", "task2_msa_verify"}:
        return _build_task2_command(job_type, payload, runtime_settings)
    if job_type in {"task3_pta_verify", "task3_msa_verify"}:
        return _build_task3_command(job_type, payload, runtime_settings)
    raise RuntimeError(f"unsupported job_type: {job_type}")


def _build_shell_for_task2_mf(payload: dict[str, Any], runtime_settings: dict[str, str]) -> str:
    from utils.task import task2

    _configure_task2_for_job(payload, runtime_settings)
    dist_cfg = task2.resolve_distributed_config()
    use_msrun = int(dist_cfg.get("world_size", 1)) > 1
    train_iters = int(payload["train_iters"])
    mutate_args = str(payload["mutate_args"])
    load_path = str(payload["load_path"])
    mf_args_path = str(payload["mf_args_path"])
    shared_weight_ckpt_path = str(payload.get("shared_weight_ckpt_path", ""))
    step_log_csv_path = payload.get("step_log_csv_path")
    cleanup_paths = ["msrun_log"]
    if step_log_csv_path:
        cleanup_paths.append(str(step_log_csv_path))
        step_log_block = f"export LMSV_TRAINING_LOG_CSV={task2.shlex.quote(str(Path(step_log_csv_path).resolve()))}"
    else:
        step_log_block = "unset LMSV_TRAINING_LOG_CSV"
    if use_msrun:
        launch_cmd = f"""
NPUS_PER_NODE={dist_cfg["npus_per_node"]}
MASTER_ADDR={task2.shlex.quote(dist_cfg["master_addr"])}
MASTER_PORT={dist_cfg["master_port"]}
NNODES={dist_cfg["nnodes"]}
NODE_RANK={dist_cfg["node_rank"]}
WORLD_SIZE={dist_cfg["world_size"]}

DISTRIBUTED_ARGS="
    --master_addr $MASTER_ADDR \
    --node_rank $NODE_RANK \
    --worker_num $WORLD_SIZE \
    --local_worker_num $NPUS_PER_NODE \
    --master_port $MASTER_PORT \
    --log_dir=msrun_log \
    --join=True \
    --cluster_time_out=300 \
    --bind_core=True
"

msrun $DISTRIBUTED_ARGS python utils/runtime/mf_mutate_and_forward/load_and_forward_submodule.py \
    $MUTATE_ARGS \
    --load-path {task2.shlex.quote(load_path)} \
    --train-iters {train_iters} \
    --args_path {task2.shlex.quote(mf_args_path)}
"""
    else:
        launch_cmd = f"""
python utils/runtime/mf_mutate_and_forward/load_and_forward_submodule.py \
    $MUTATE_ARGS \
    --load-path {task2.shlex.quote(load_path)} \
    --train-iters {train_iters} \
    --args_path {task2.shlex.quote(mf_args_path)}
"""
    return f"""
{task2.build_conda_activate_block(runtime_settings["MF_NAME"], load_ascend=True)}
export MUTATE_ROUND={int(payload["iter_num"])}
export MUTATE_ARGS={task2.shlex.quote(mutate_args)}
export LMSV_SHARED_WEIGHT_CKPT_PATH={task2.shlex.quote(shared_weight_ckpt_path)}
export LMSV_ALIGN_ADD_QKV_BIAS=${{LMSV_ALIGN_ADD_QKV_BIAS:-1}}
export LMSV_TASK3_FORCE_MF_SAFE=${{LMSV_TASK3_FORCE_MF_SAFE:-0}}
export LMSV_STRICT_ATTN_CONFIG_MATCH=${{LMSV_STRICT_ATTN_CONFIG_MATCH:-0}}
export LMSV_STRICT_ATTN_PARAM_LOAD=${{LMSV_STRICT_ATTN_PARAM_LOAD:-1}}
export LMSV_STRICT_DECODER_CONFIG_MATCH=${{LMSV_STRICT_DECODER_CONFIG_MATCH:-0}}
export LMSV_TRAIN_ITERS={train_iters}
{_build_cleanup_block(cleanup_paths)}
{step_log_block}
{launch_cmd}
"""


def _build_shell_for_task3_mf(payload: dict[str, Any], runtime_settings: dict[str, str]) -> str:
    from utils.task import task3

    _configure_task3_for_job(payload, runtime_settings)
    dist_cfg = task3.resolve_distributed_config()
    train_iters = int(payload["train_iters"])
    mutate_args = str(payload["mutate_args"])
    load_path = str(payload["load_path"])
    mf_args_path = str(payload["mf_args_path"])
    shared_weight_ckpt_path = str(payload.get("shared_weight_ckpt_path", ""))
    step_log_csv_path = payload.get("step_log_csv_path")
    cleanup_paths = ["msrun_log"]
    if step_log_csv_path:
        cleanup_paths.append(str(step_log_csv_path))
        step_log_block = f"export LMSV_TRAINING_LOG_CSV={task3.shlex.quote(str(Path(step_log_csv_path).resolve()))}"
    else:
        step_log_block = "unset LMSV_TRAINING_LOG_CSV"

    use_msrun = int(dist_cfg.get("world_size", 1)) > 1
    if use_msrun:
        launch_cmd = f"""
NPUS_PER_NODE={dist_cfg["npus_per_node"]}
MASTER_ADDR={task3.shlex.quote(dist_cfg["master_addr"])}
MASTER_PORT={dist_cfg["master_port"]}
NNODES={dist_cfg["nnodes"]}
NODE_RANK={dist_cfg["node_rank"]}
WORLD_SIZE={dist_cfg["world_size"]}

DISTRIBUTED_ARGS="
    --master_addr $MASTER_ADDR \
    --node_rank $NODE_RANK \
    --worker_num $WORLD_SIZE \
    --local_worker_num $NPUS_PER_NODE \
    --master_port $MASTER_PORT \
    --log_dir=msrun_log \
    --join=True \
    --cluster_time_out=300 \
    --bind_core=True
"

msrun $DISTRIBUTED_ARGS python utils/runtime/mf_mutate_and_forward/load_and_forward_graph.py \
    $MUTATE_ARGS \
    --load-path {task3.shlex.quote(load_path)} \
    --shared-weight-ckpt {task3.shlex.quote(shared_weight_ckpt_path)} \
    --train-iters {train_iters} \
    --args_path {task3.shlex.quote(mf_args_path)}
"""
    else:
        launch_cmd = f"""
python utils/runtime/mf_mutate_and_forward/load_and_forward_graph.py \
    $MUTATE_ARGS \
    --load-path {task3.shlex.quote(load_path)} \
    --shared-weight-ckpt {task3.shlex.quote(shared_weight_ckpt_path)} \
    --train-iters {train_iters} \
    --args_path {task3.shlex.quote(mf_args_path)}
"""
    return f"""
{task3.build_conda_activate_block(runtime_settings["MF_NAME"], load_ascend=True)}
export MUTATE_ROUND={int(payload["iter_num"])}
export MUTATE_ARGS={task3.shlex.quote(mutate_args)}
export LMSV_SHARED_WEIGHT_CKPT_PATH={task3.shlex.quote(shared_weight_ckpt_path)}
export LMSV_ALIGN_ADD_QKV_BIAS=${{LMSV_ALIGN_ADD_QKV_BIAS:-1}}
export LMSV_TASK3_FORCE_MF_SAFE=${{LMSV_TASK3_FORCE_MF_SAFE:-0}}
export LMSV_STRICT_ATTN_CONFIG_MATCH=${{LMSV_STRICT_ATTN_CONFIG_MATCH:-0}}
export LMSV_STRICT_ATTN_PARAM_LOAD=${{LMSV_STRICT_ATTN_PARAM_LOAD:-1}}
export LMSV_STRICT_DECODER_CONFIG_MATCH=${{LMSV_STRICT_DECODER_CONFIG_MATCH:-0}}
export LMSV_TRAIN_ITERS={train_iters}
{_build_cleanup_block(cleanup_paths)}
{step_log_block}
{launch_cmd}
"""


def _launch_shell_job(job: _SlaveJob, command_text: str, *, timeout: int) -> None:
    assert job.log_path is not None
    shell_cmd = f"set -e -o pipefail\n{command_text}"
    _write_job_log_header(job.log_path, command_text)
    with job.log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            ["bash", "-lxc", shell_cmd],
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        job.process_pid = process.pid
        job.status = "running"
        job.started_at = time.time()
        timed_out = False
        cancelled = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        except BaseException:
            cancelled = job.cancelled
            raise
        finally:
            job.exit_code = int(process.returncode)
            job.ended_at = time.time()
            job.timed_out = timed_out
            job.cancelled = cancelled or job.cancelled
            if job.cancelled:
                job.status = "cancelled"
            elif timed_out:
                job.status = "timeout"
            elif process.returncode == 0:
                job.status = "success"
            else:
                job.status = "failed"
            _append_job_log_footer(
                job.log_path,
                int(process.returncode),
                timed_out=timed_out,
                cancelled=job.cancelled,
            )


def run_direct_job_payload(encoded_payload: str) -> int:
    """Run one ssh-dispatched job on the remote node without a resident slave process."""
    try:
        data = json.loads(base64.b64decode(encoded_payload.encode("ascii")).decode("utf-8"))
        job_type = str(data["job_type"])
        payload = data["payload"]
        runtime_settings = data["runtime_settings"]
        if not isinstance(payload, dict) or not isinstance(runtime_settings, dict):
            raise RuntimeError("invalid direct job payload")

        session_id = str(payload.get("_session_id") or "direct")
        runtime_helpers.configure_project_tmp_env(REPO_ROOT / "tmp" / "ssh_direct" / session_id)
        timeout = _job_timeout(payload)
        if job_type == "task2_mf_verify":
            command_text = _build_shell_for_task2_mf(payload, runtime_settings)
        elif job_type == "task3_mf_verify":
            command_text = _build_shell_for_task3_mf(payload, runtime_settings)
        else:
            command_text = _build_command_for_job(job_type, payload, runtime_settings)
    except Exception as exc:
        print(f"[JOB_ERROR] {exc}", file=sys.stderr, flush=True)
        print("[RETURNCODE] 1", flush=True)
        return 1

    print(f"[START] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)
    print("[COMMAND]", flush=True)
    print(command_text, flush=True)
    shell_cmd = f"set -e -o pipefail\n{command_text}"
    process = subprocess.Popen(
        ["bash", "-lxc", shell_cmd],
        cwd=str(REPO_ROOT),
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    return_code = int(process.returncode)
    if timed_out and return_code == 0:
        return_code = 124
    print(f"\n[END] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)
    if timed_out:
        print("[TIMED_OUT] 1", flush=True)
    print(f"[RETURNCODE] {return_code}", flush=True)
    return 124 if timed_out else return_code


def _run_job(job: _SlaveJob) -> None:
    runtime_settings = resolve_local_runtime_settings()
    session = _SLAVE_STATE.ensure_session(job.session_id)
    runtime_helpers.configure_project_tmp_env(session.project_tmp_root)
    if job.log_path is None:
        job.log_path = session.root / "job_logs" / f"{job.job_type}_{job.job_id}.log"
    timeout = _job_timeout(job.payload)
    try:
        if job.job_type == "task2_mf_verify":
            command_text = _build_shell_for_task2_mf(job.payload, runtime_settings)
        elif job.job_type == "task3_mf_verify":
            command_text = _build_shell_for_task3_mf(job.payload, runtime_settings)
        else:
            command_text = _build_command_for_job(job.job_type, job.payload, runtime_settings)
        _launch_shell_job(job, command_text, timeout=timeout)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        job.ended_at = time.time()
        if job.log_path is not None:
            with job.log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[JOB_ERROR] {exc}\n")
                handle.write(f"[END] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                handle.write("[RETURNCODE] 1\n")
    finally:
        current = session.jobs.get(session.current_job_id)
        if current is job:
            session.current_job_id = ""


class SlaveRequestHandler(BaseHTTPRequestHandler):
    server_version = "LMSVSlave/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            runtime_settings = resolve_local_runtime_settings()
            cluster_cfg = resolve_cluster_from_config()
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "hostname": os.uname().nodename,
                    "visible_devices": detect_visible_device_count(),
                    "pta_env": runtime_settings["PTA_NAME"],
                    "msa_env": runtime_settings["MSA_NAME"],
                    "mf_env": runtime_settings["MF_NAME"],
                    "pta_path": runtime_settings["PTA_PATH"],
                    "msa_path": runtime_settings["MSA_PATH"],
                    "listen_port": cluster_cfg.listen_port,
                    "configured_node_rank": cluster_cfg.node_rank,
                },
            )
            return

        if parsed.path.startswith("/sessions/") and parsed.path.endswith("/logs"):
            parts = [item for item in parsed.path.split("/") if item]
            if len(parts) != 5:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "invalid log path"})
                return
            session_id = parts[1]
            job_id = parts[3]
            session = _SLAVE_STATE.ensure_session(session_id)
            job = session.jobs.get(job_id)
            if job is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "job not found"})
                return
            if job.log_path is None:
                _json_response(self, HTTPStatus.OK, {"chunk": "", "offset": 0, "finished": False})
                return
            query = parse_qs(parsed.query)
            offset = _parse_optional_positive_int((query.get("offset") or ["0"])[0])
            if not job.log_path.exists():
                _json_response(self, HTTPStatus.OK, {"chunk": "", "offset": offset, "finished": False})
                return
            text = job.log_path.read_text(encoding="utf-8", errors="ignore")
            chunk = text[offset:]
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "chunk": chunk,
                    "offset": len(text),
                    "finished": job.status in {"success", "failed", "timeout", "cancelled"},
                },
            )
            return

        if parsed.path.startswith("/sessions/") and "/jobs/" in parsed.path:
            parts = [item for item in parsed.path.split("/") if item]
            if len(parts) != 4:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "invalid job path"})
                return
            session_id = parts[1]
            job_id = parts[3]
            session = _SLAVE_STATE.ensure_session(session_id)
            job = session.jobs.get(job_id)
            if job is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "job not found"})
                return
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "exit_code": job.exit_code,
                    "timed_out": job.timed_out,
                    "cancelled": job.cancelled,
                    "error": job.error,
                    "started_at": job.started_at,
                    "ended_at": job.ended_at,
                    "process_pid": job.process_pid,
                },
            )
            return

        _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/sessions/") and parsed.path.endswith("/prepare"):
            parts = [item for item in parsed.path.split("/") if item]
            session_id = parts[1]
            _SLAVE_STATE.ensure_session(session_id)
            _json_response(self, HTTPStatus.OK, {"ok": True, "session_id": session_id})
            return

        if parsed.path.startswith("/sessions/") and parsed.path.endswith("/cleanup"):
            parts = [item for item in parsed.path.split("/") if item]
            session_id = parts[1]
            _SLAVE_STATE.cleanup_session(session_id)
            _json_response(self, HTTPStatus.OK, {"ok": True})
            return

        if parsed.path.startswith("/sessions/") and parsed.path.endswith("/bundle"):
            parts = [item for item in parsed.path.split("/") if item]
            session_id = parts[1]
            session = _SLAVE_STATE.ensure_session(session_id)
            length = _parse_optional_positive_int(self.headers.get("Content-Length"))
            payload = self.rfile.read(length) if length > 0 else b""
            if payload:
                with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                    archive.extractall(path=REPO_ROOT)
            session.last_touch = time.time()
            _json_response(self, HTTPStatus.OK, {"ok": True})
            return

        if parsed.path.startswith("/sessions/") and parsed.path.endswith("/jobs"):
            parts = [item for item in parsed.path.split("/") if item]
            session_id = parts[1]
            session = _SLAVE_STATE.ensure_session(session_id)
            if session.current_job_id:
                current = session.jobs.get(session.current_job_id)
                if current is not None and current.status in {"queued", "running"}:
                    _json_response(
                        self,
                        HTTPStatus.CONFLICT,
                        {"error": f"current job still active: {session.current_job_id}"},
                    )
                    return
            data = _read_json_body(self)
            job_type = str(data.get("job_type", "")).strip()
            payload = data.get("payload")
            if not job_type or not isinstance(payload, dict):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "missing job_type/payload"})
                return
            job_id = uuid.uuid4().hex
            log_path = session.root / "job_logs" / f"{job_type}_{job_id}.log"
            job = _SlaveJob(
                job_id=job_id,
                session_id=session_id,
                job_type=job_type,
                payload=payload,
                log_path=log_path,
            )
            session.jobs[job_id] = job
            session.current_job_id = job_id
            thread = threading.Thread(target=_run_job, args=(job,), name=f"slave-job-{job_id}", daemon=True)
            job.thread = thread
            thread.start()
            _json_response(self, HTTPStatus.OK, {"ok": True, "job_id": job_id})
            return

        if parsed.path.startswith("/sessions/") and parsed.path.endswith("/cancel"):
            parts = [item for item in parsed.path.split("/") if item]
            if len(parts) != 5:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "invalid cancel path"})
                return
            session_id = parts[1]
            job_id = parts[3]
            session = _SLAVE_STATE.ensure_session(session_id)
            job = session.jobs.get(job_id)
            if job is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "job not found"})
                return
            job.cancelled = True
            if job.process_pid > 0:
                try:
                    os.killpg(job.process_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            _json_response(self, HTTPStatus.OK, {"ok": True})
            return

        if parsed.path.startswith("/sessions/") and parsed.path.endswith("/collect"):
            parts = [item for item in parsed.path.split("/") if item]
            session_id = parts[1]
            _SLAVE_STATE.ensure_session(session_id)
            data = _read_json_body(self)
            items = data.get("items")
            if not isinstance(items, list):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "missing items"})
                return
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    rel_path = str(item.get("path", "")).strip()
                    flatten = _to_bool(item.get("flatten"))
                    if not rel_path:
                        continue
                    abs_path = (REPO_ROOT / rel_path).resolve()
                    if not abs_path.exists():
                        continue
                    if abs_path.is_dir():
                        if flatten:
                            for child in sorted(abs_path.rglob("*")):
                                if not child.is_file():
                                    continue
                                arcname = child.relative_to(abs_path)
                                archive.add(str(child), arcname=str(arcname))
                        else:
                            archive.add(str(abs_path), arcname=Path(rel_path).name)
                    else:
                        arcname = abs_path.name if flatten else rel_path
                        archive.add(str(abs_path), arcname=str(arcname))
            _bytes_response(self, HTTPStatus.OK, buffer.getvalue(), "application/gzip")
            return

        _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_slave_server(config: ClusterConfig, log_info=print) -> None:
    runtime_helpers.configure_project_tmp_env(REPO_ROOT / "tmp" / "slave_bootstrap")
    server = ThreadingHTTPServer((config.listen_host, config.listen_port), SlaveRequestHandler)
    log_info(
        f"[slave] listening on {config.listen_host}:{config.listen_port} | "
        f"master_addr={config.master_addr} | configured_node_rank={config.node_rank}"
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()


def _main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] == "direct-job":
        return run_direct_job_payload(argv[2])
    print("usage: python -m utils.runtime.cluster_runtime direct-job <base64-json>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
