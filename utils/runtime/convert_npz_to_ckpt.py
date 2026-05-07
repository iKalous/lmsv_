#!/usr/bin/env python3
"""Convert .npz tensors to MindSpore .ckpt (mindspore-only environment)."""

import argparse
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore.train.serialization import save_checkpoint


def convert_npz_to_ckpt(npz_path: Path, ckpt_path: Path) -> None:
    data = np.load(str(npz_path), allow_pickle=False)
    params = []
    for key in data.files:
        params.append({"name": key, "data": ms.Tensor(data[key])})

    if not params:
        raise RuntimeError("No tensors found in npz.")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(params, str(ckpt_path))
    print(f"[convert] tensors={len(params)} ckpt={ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert npz to MindSpore ckpt")
    parser.add_argument("--npz", required=True, help="Input npz path")
    parser.add_argument("--ckpt", required=True, help="Output ckpt path")
    args = parser.parse_args()

    convert_npz_to_ckpt(Path(args.npz), Path(args.ckpt))
