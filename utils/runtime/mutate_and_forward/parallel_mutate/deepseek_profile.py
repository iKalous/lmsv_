#!/usr/bin/env python3
"""DeepSeekV3-specific low-memory adjustments for generated PTA bash scripts."""

from __future__ import annotations

import re
from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _extract_int(script_path: Path, param_key: str, default: int) -> int:
    try:
        content = _read_text(script_path)
        match = re.search(rf"{re.escape(param_key)}\s+([0-9]+)", content)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return int(default)


def _get_first_k_dense_replace_limit(script_path: Path) -> int:
    num_layers = max(1, _extract_int(script_path, "--num-layers", 8))
    pipeline_parallel_size = max(1, _extract_int(script_path, "--pipeline-model-parallel-size", 1))
    layers_per_stage = max(1, (num_layers + pipeline_parallel_size - 1) // pipeline_parallel_size)
    return max(0, layers_per_stage - 1)


def _replace_or_insert_value(script_path: Path, flag_key: str, pattern: str, new_value: str) -> bool:
    content = _read_text(script_path)
    updated, count = re.subn(pattern, rf"\g<1>{new_value}", content, count=1)
    if count > 0:
        if updated != content:
            _write_text(script_path, updated)
        return True

    lines = content.splitlines()
    insert_at = len(lines)
    for index, line in enumerate(lines):
        if "--train-iters" in line:
            insert_at = index + 1
            break
    lines.insert(insert_at, f"    {flag_key} {new_value} \\")
    _write_text(script_path, "\n".join(lines) + "\n")
    return True


def _replace_or_insert_flag(script_path: Path, flag_key: str) -> bool:
    content = _read_text(script_path)
    if re.search(rf"(^|\s){re.escape(flag_key)}(\s|\\|$)", content, flags=re.MULTILINE):
        return True

    lines = content.splitlines()
    insert_at = len(lines)
    for index, line in enumerate(lines):
        if "--train-iters" in line:
            insert_at = index + 1
            break
    lines.insert(insert_at, f"    {flag_key} \\")
    _write_text(script_path, "\n".join(lines) + "\n")
    return True


def _remove_script_flag(script_path: Path, flag_key: str) -> bool:
    content = _read_text(script_path)
    lines = content.splitlines(keepends=True)
    filtered_lines = []
    removed = False
    for line in lines:
        if re.search(rf"(^|\s){re.escape(flag_key)}(\s|\\|$)", line):
            removed = True
            continue
        filtered_lines.append(line)

    if not removed:
        return True

    new_content = "".join(filtered_lines)
    if new_content != content:
        _write_text(script_path, new_content)
    return True


def _ensure_global_batch_divisible(script_path: Path, world_size: int = 8) -> bool:
    micro_bs = max(1, _extract_int(script_path, "--micro-batch-size", 1))
    global_bs = max(1, _extract_int(script_path, "--global-batch-size", 1))
    content = _read_text(script_path)

    tp_match = re.search(r"--tensor-model-parallel-size\s+([0-9]+)", content)
    pp_match = re.search(r"--pipeline-model-parallel-size\s+([0-9]+)", content)
    cp_match = re.search(r"--context-parallel-size\s+([0-9]+)", content)

    if not (tp_match and pp_match and cp_match):
        return True

    tp = max(1, int(tp_match.group(1)))
    pp = max(1, int(pp_match.group(1)))
    cp = max(1, int(cp_match.group(1)))
    denom = max(1, tp * pp * cp)
    dp = max(1, int(world_size) // denom)
    divisor = max(1, micro_bs * dp)

    if global_bs % divisor == 0:
        return True

    adjusted_global = ((global_bs + divisor - 1) // divisor) * divisor
    return _replace_or_insert_value(
        script_path,
        "--global-batch-size",
        r"(--global-batch-size\s+)[0-9]+",
        str(adjusted_global),
    )


def apply_deepseekv3_unified_low_memory_profile(script_path: str | Path) -> bool:
    """Apply the DeepSeekV3 low-memory profile to a generated PTA bash script."""
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"脚本不存在: {path}")

    ok = True

    ok = _remove_script_flag(path, "--group-query-attention") and ok
    ok = _replace_or_insert_flag(path, "--moe-router-enable-expert-bias") and ok
    ok = _replace_or_insert_flag(path, "--no-check-for-nan-in-loss-and-grad") and ok
    ok = _replace_or_insert_value(path, "--moe-router-score-function", r"(--moe-router-score-function\s+)[^\s\\]+", "sigmoid") and ok
    ok = _replace_or_insert_value(path, "--micro-batch-size", r"(--micro-batch-size\s+)[0-9]+", "1") and ok
    ok = _replace_or_insert_value(path, "--global-batch-size", r"(--global-batch-size\s+)[0-9]+", "8") and ok
    ok = _replace_or_insert_value(path, "--num-layers", r"(--num-layers\s+)[0-9]+", "8") and ok
    ok = _replace_or_insert_value(path, "--hidden-size", r"(--hidden-size\s+)[0-9]+", "1024") and ok
    ok = _replace_or_insert_value(path, "--ffn-hidden-size", r"(--ffn-hidden-size\s+)[0-9]+", "2048") and ok
    ok = _replace_or_insert_value(path, "--num-attention-heads", r"(--num-attention-heads\s+)[0-9]+", "16") and ok
    ok = _replace_or_insert_value(path, "--q-lora-rank", r"(--q-lora-rank\s+)[0-9]+", "192") and ok
    ok = _replace_or_insert_value(path, "--kv-lora-rank", r"(--kv-lora-rank\s+)[0-9]+", "64") and ok
    ok = _replace_or_insert_value(path, "--moe-intermediate-size", r"(--moe-intermediate-size\s+)[0-9]+", "768") and ok
    ok = _replace_or_insert_value(path, "--num-experts", r"(--num-experts\s+)[0-9]+", "16") and ok
    ok = _replace_or_insert_value(path, "--n-shared-experts", r"(--n-shared-experts\s+)[0-9]+", "1") and ok
    ok = _replace_or_insert_value(path, "--moe-router-topk", r"(--moe-router-topk\s+)[0-9]+", "2") and ok
    ok = _replace_or_insert_value(path, "--seq-length", r"(--seq-length\s+)[0-9]+", "1024") and ok
    ok = _replace_or_insert_value(path, "--max-position-embeddings", r"(--max-position-embeddings\s+)[0-9]+", "1024") and ok
    ok = _replace_or_insert_value(path, "--moe-layer-freq", r"(--moe-layer-freq\s+)[0-9]+", "1") and ok
    first_k_dense_replace = _get_first_k_dense_replace_limit(path)
    ok = _replace_or_insert_value(
        path,
        "--first-k-dense-replace",
        r"(--first-k-dense-replace\s+)[0-9]+",
        str(first_k_dense_replace),
    ) and ok
    ok = _ensure_global_batch_divisible(path) and ok
    ok = _remove_script_flag(path, "--num-layers-per-virtual-pipeline-stage") and ok

    if not ok:
        raise RuntimeError(f"DeepSeekV3统一减配失败: {path}")

    return True
