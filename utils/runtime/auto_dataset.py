#!/usr/bin/env python3
"""Automatic raw-dataset discovery and indexed dataset preparation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from pathlib import Path

from utils.task import runtime_helpers


RAW_DATASET_ENV = "LMSV_RAW_DATASET_PATH"
PREPARED_DATASET_ENV = "LMSV_PREPARED_DATA_PATH"
DEFAULT_DATASET_DIR = Path("assets") / "datasets"
DEFAULT_PREPARED_ROOT = Path("tmp") / "prepared_datasets"
DEFAULT_OUTPUT_BASENAME = "dataset_text_document"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_payload(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _normalize_path(text: str, repo_root: Path) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def _expand_shell_vars(text: str, variables: dict[str, str]) -> str:
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
    return pattern.sub(lambda m: variables.get(m.group(1), m.group(0)), text)


def _discover_raw_dataset(repo_root: Path) -> Path:
    override = os.environ.get(RAW_DATASET_ENV, "").strip()
    if override:
        path = _normalize_path(override, repo_root)
        if not path.exists():
            raise FileNotFoundError(f"{RAW_DATASET_ENV} 指向的数据集不存在: {path}")
        return path

    dataset_dir = (repo_root / DEFAULT_DATASET_DIR).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"未找到默认原始数据集目录: {dataset_dir}")

    candidates = sorted(
        [
            path
            for path in dataset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
        ]
    )
    if not candidates:
        raise FileNotFoundError(f"{dataset_dir} 下未发现可用的 .json/.jsonl 原始数据集")

    preferred_names = ("dataset_raw.json", "dataset_raw.jsonl")
    for name in preferred_names:
        for candidate in candidates:
            if candidate.name == name:
                return candidate

    if len(candidates) == 1:
        return candidates[0]

    names = ", ".join(path.name for path in candidates)
    raise RuntimeError(f"发现多个原始数据集，请仅保留一个或设置 {RAW_DATASET_ENV}: {names}")


def _extract_template_tokenizer(model_name: str, repo_root: Path) -> dict[str, str]:
    template_path = repo_root / "scripts" / "templates" / "pretrain_example" / f"pretrain_mutated_{model_name}.sh"
    if not template_path.exists():
        raise FileNotFoundError(f"未找到模型 {model_name} 的训练模板: {template_path}")

    content = template_path.read_text(encoding="utf-8", errors="ignore")
    variables: dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
            continue
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        variables[name] = _expand_shell_vars(value, variables)

    tokenizer_type_match = re.search(r"--tokenizer-type\s+([^\s\\]+)", content)
    tokenizer_name_match = re.search(r"--tokenizer-name-or-path\s+([^\s\\]+)", content)
    tokenizer_model_match = re.search(r"--tokenizer-model\s+([^\s\\]+)", content)

    if not tokenizer_type_match:
        raise RuntimeError(f"无法从模板中解析 tokenizer 类型: {template_path}")
    tokenizer_type = tokenizer_type_match.group(1).strip()

    raw_source = ""
    source_kind = ""
    if tokenizer_name_match:
        raw_source = tokenizer_name_match.group(1).strip()
        source_kind = "name_or_path"
    elif tokenizer_model_match:
        raw_source = tokenizer_model_match.group(1).strip()
        source_kind = "model"
    else:
        raise RuntimeError(f"无法从模板中解析 tokenizer 路径: {template_path}")

    expanded_source = _expand_shell_vars(raw_source, variables)
    if expanded_source.startswith("./"):
        expanded_source = str((repo_root / expanded_source[2:]).resolve())
    elif not os.path.isabs(expanded_source):
        expanded_source = str((repo_root / expanded_source).resolve())

    return {
        "template_path": str(template_path.resolve()),
        "tokenizer_type": tokenizer_type,
        "tokenizer_source": expanded_source,
        "tokenizer_source_kind": source_kind,
    }


def _run_prepare_command(
    *,
    repo_root: Path,
    pta_env: str,
    pta_path: str,
    raw_dataset_path: Path,
    output_prefix: Path,
    tokenizer_spec: dict[str, str],
    log_info,
    log_error,
) -> None:
    log_file = output_prefix.parent / "prepare.log"
    if not pta_path:
        raise ValueError("缺少 PTA_PATH/PTAPATH，无法执行自动数据预处理")
    cmd = f"""
    {runtime_helpers.build_conda_activate_block(pta_env, load_ascend=True)}
    export PTAPATH={shlex.quote(pta_path)}
    source scripts/envset/pta.sh
    python -m utils.runtime.prepare_indexed_dataset \
      --input {shlex.quote(str(raw_dataset_path))} \
      --output-prefix {shlex.quote(str(output_prefix))} \
      --tokenizer-type {shlex.quote(tokenizer_spec["tokenizer_type"])} \
      --tokenizer-source {shlex.quote(tokenizer_spec["tokenizer_source"])} \
      --model-name {shlex.quote(output_prefix.parent.name)}
    """
    log_info(f"开始自动预处理原始数据集: {raw_dataset_path.name}")
    result = runtime_helpers.run_shell_to_file(
        cmd,
        log_file,
        repo_root,
        log_error,
        check=False,
    )
    if result is None or result.returncode != 0:
        raise RuntimeError(f"原始数据集预处理失败，请查看日志: {log_file}")


def ensure_task1_data_path(config: dict, repo_root: Path, log_info, log_error) -> str:
    task_config = ((config.get("tasks") or {}).get("1")) or {}
    model_name = str(task_config.get("MODEL_NAME") or "").strip()
    if not model_name:
        raise ValueError("任务1缺少 MODEL_NAME，无法自动准备数据集")

    raw_dataset_path = _discover_raw_dataset(repo_root)

    tokenizer_spec = _extract_template_tokenizer(model_name, repo_root)
    raw_hash = _hash_file(raw_dataset_path)
    fingerprint = _hash_payload(
        {
            "raw_dataset_path": str(raw_dataset_path),
            "raw_dataset_sha256": raw_hash,
            "model_name": model_name,
            "tokenizer_type": tokenizer_spec["tokenizer_type"],
            "tokenizer_source": tokenizer_spec["tokenizer_source"],
        }
    )
    cache_dir = (repo_root / DEFAULT_PREPARED_ROOT / model_name / f"{raw_dataset_path.stem}-{fingerprint[:12]}").resolve()
    output_prefix = cache_dir / DEFAULT_OUTPUT_BASENAME
    metadata_path = cache_dir / "metadata.json"
    output_bin = output_prefix.with_suffix(".bin")
    output_idx = output_prefix.with_suffix(".idx")

    if output_bin.exists() and output_idx.exists():
        log_info(f"复用已缓存的训练数据前缀: {output_prefix}")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        pta_env = str(config.get("PTA_NAME") or os.environ.get("PTA_NAME") or "").strip()
        if not pta_env:
            raise ValueError("缺少 PTA_NAME，无法在 PTA 环境中执行自动数据预处理")
        pta_path = str(config.get("PTA_PATH") or os.environ.get("PTA_PATH") or os.environ.get("PTAPATH") or "").strip()
        if not pta_path:
            raise ValueError("缺少 PTA_PATH/PTAPATH，无法执行自动数据预处理")
        _run_prepare_command(
            repo_root=repo_root,
            pta_env=pta_env,
            pta_path=pta_path,
            raw_dataset_path=raw_dataset_path,
            output_prefix=output_prefix,
            tokenizer_spec=tokenizer_spec,
            log_info=log_info,
            log_error=log_error,
        )

    metadata = {
        "raw_dataset_path": str(raw_dataset_path),
        "raw_dataset_sha256": raw_hash,
        "model_name": model_name,
        "prepared_prefix": str(output_prefix),
        "prepared_bin": str(output_bin),
        "prepared_idx": str(output_idx),
        "fingerprint": fingerprint,
        "tokenizer": tokenizer_spec,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    os.environ[RAW_DATASET_ENV] = str(raw_dataset_path)
    os.environ[PREPARED_DATASET_ENV] = str(output_prefix)
    return str(output_prefix)
