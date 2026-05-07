#!/usr/bin/env python3
"""Export torch .pth state dict tensors to .npz (torch-only environment)."""

import argparse
import re
from pathlib import Path

import numpy as np
import torch


def _normalize_state_dict(obj):
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Unsupported pth payload: expected dict or dict with 'state_dict'.")


def _map_pta_key_to_mf(key: str):
    # Drop framework metadata tensors that are not real model parameters.
    if key.endswith("._extra_state"):
        return None

    if key.startswith("shared.embedding.0."):
        key = "node_block_0." + key[len("shared.embedding.0."):]

    if key.startswith("shared.decoder.0."):
        key = "node_block_1." + key[len("shared.decoder.0."):]

    match = re.match(r"^nodes\.(\d+)\.(.+)$", key)
    if match:
        node_idx = int(match.group(1)) - 1
        key = f"node_block_{node_idx}.{match.group(2)}"

    # MindSpore LayerNorm parameter names use gamma/beta while PTA uses weight/bias.
    key = re.sub(r"(layernorm)\.weight$", r"\1.gamma", key)
    key = re.sub(r"(layernorm)\.bias$", r"\1.beta", key)

    # Keep bias tensors and let the downstream loader decide by net parameter match.
    # This is required for cases where MF graph enables qkv bias alignment.

    return key


def export_pth_to_npz(pth_path: Path, npz_path: Path) -> None:
    payload = torch.load(str(pth_path), map_location="cpu")
    state_dict = _normalize_state_dict(payload)

    arrays = {}
    skipped = 0
    remapped = 0
    duplicated = 0
    for name, value in state_dict.items():
        mapped_name = _map_pta_key_to_mf(name)
        if mapped_name is None:
            skipped += 1
            continue

        if not hasattr(value, "detach"):
            skipped += 1
            continue

        if mapped_name != name:
            remapped += 1

        tensor = value.detach().cpu()
        if mapped_name in arrays:
            duplicated += 1
            continue
        arrays[mapped_name] = tensor.numpy()

    if not arrays:
        raise RuntimeError("No tensor parameters were exported from pth.")

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(npz_path), **arrays)
    print(
        f"[export] tensors={len(arrays)} skipped={skipped} remapped={remapped} "
        f"duplicated={duplicated} npz={npz_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export torch .pth to numpy .npz")
    parser.add_argument("--pth", required=True, help="Input torch pth path")
    parser.add_argument("--npz", required=True, help="Output npz path")
    args = parser.parse_args()

    export_pth_to_npz(Path(args.pth), Path(args.npz))
