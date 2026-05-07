#!/usr/bin/env python3
"""Checkpoint conversion entry for LMSV runtime.

This is migrated from the legacy conversion utility and constrained by
the local support list.
"""

import argparse
import logging
import os
import sys
import time

# Ensure repository root is importable when invoked from nested runtime shells.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.runtime.model_support import TASK1_WEIGHT_CONVERT_ACCEPTED_MODELS
from utils.runtime.model_support import resolve_task1_weight_convert_model_alias


def _patch_qwen3_mg2hf_router_compat(converter, model_type_hf):
    """Tolerate old MindSpeed-LLM qwen3 mg2hf maps without router HF keys."""
    if str(model_type_hf).strip().lower() != "qwen3":
        return

    method = getattr(converter, "set_model_layer_mlp", None)
    if method is None or getattr(method, "_lmsv_router_compat", False):
        return

    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except KeyError as exc:
            if exc.args and exc.args[0] == "layers_mlp_router":
                logging.warning(
                    "Compatibility: skipped qwen3 router weight export because "
                    "MindSpeed-LLM mg2hf mapping lacks layers_mlp_router"
                )
                return None
            raise

    wrapper._lmsv_router_compat = True
    setattr(converter, "set_model_layer_mlp", wrapper)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load-model-type",
        type=str,
        nargs="?",
        default="hf",
        const=None,
        choices=["hf", "mg"],
        help="Type of the converter",
    )
    parser.add_argument(
        "--save-model-type",
        type=str,
        default="mg",
        choices=["mg", "hf"],
        help="Save model type",
    )
    parser.add_argument(
        "--load-dir",
        type=str,
        required=True,
        help="Directory to load model checkpoint from",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save model checkpoint to",
    )
    parser.add_argument(
        "--model-type-hf",
        type=str,
        default="qwen3",
        choices=list(TASK1_WEIGHT_CONVERT_ACCEPTED_MODELS),
        help="model type of huggingface",
    )
    parser.add_argument(
        "--target-tensor-parallel-size",
        type=int,
        default=1,
        help="Target tensor model parallel size, defaults to 1.",
    )
    parser.add_argument(
        "--target-pipeline-parallel-size",
        type=int,
        default=1,
        help="Target pipeline model parallel size, defaults to 1.",
    )
    parser.add_argument(
        "--target-expert-parallel-size",
        type=int,
        default=1,
        help="Target expert model parallel size, defaults to 1.",
    )
    parser.add_argument(
        "--expert-tensor-parallel-size",
        type=int,
        default=None,
        help=(
            "Degree of expert model parallelism, Currentley it is support "
            "to be set to 1 or None. Default is None, which will be set "
            "to the value of --target-tensor-parallel-size"
        ),
    )
    parser.add_argument(
        "--num-layers-per-virtual-pipeline-stage",
        type=int,
        default=None,
        help="Number of layers per virtual pipeline stage",
    )
    parser.add_argument("--moe-grouped-gemm", action="store_true", help="Use moe grouped gemm.")
    parser.add_argument("--noop-layers", type=str, default=None, help="Specity the noop layers.")
    parser.add_argument("--mtp-num-layers", type=int, default=0, help="Multi-Token prediction layer num")
    parser.add_argument(
        "--num-layer-list",
        type=str,
        help="a list of number of layers, separated by comma; e.g., 4,4,4,4",
    )
    parser.add_argument(
        "--moe-tp-extend-ep",
        action="store_true",
        help="use tp group to extend experts parallism instead of sharding weight tensor of experts in tp group",
    )
    parser.add_argument("--mla-mm-split", action="store_true", default=False, help="Split 2 up-proj matmul into 4 in MLA")
    parser.add_argument(
        "--schedules-method",
        type=str,
        default=None,
        choices=["dualpipev"],
        help="An innovative bidirectional pipeline parallelism algorithm.",
    )
    parser.add_argument("--first-k-dense-replace", type=int, default=None, help="Customizing the number of dense layers.")
    parser.add_argument("--num-layers", type=int, default=None, help="Specify the number of transformer layers to use.")
    parser.add_argument(
        "--transformer-impl",
        default="local",
        choices=["local", "transformer_engine"],
        help="Which Transformer implementation to use.",
    )

    args, _ = parser.parse_known_args()
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = get_args()
    args.model_type_hf = resolve_task1_weight_convert_model_alias(args.model_type_hf)
    logging.info("Arguments: %s", args)

    from mindspeed_llm.tasks.checkpoint.convert_hf2mg import Hf2MgConvert
    from mindspeed_llm.tasks.checkpoint.convert_mg2hf import Mg2HfConvert

    if args.load_model_type == "hf" and args.save_model_type == "mg":
        converter = Hf2MgConvert(args)
    elif args.load_model_type == "mg" and args.save_model_type == "hf":
        converter = Mg2HfConvert(args)
    else:
        raise ValueError("This conversion scheme is not supported")

    _patch_qwen3_mg2hf_router_compat(converter, args.model_type_hf)

    # Compatibility shim:
    # Some upstream mg2hf paths gate DSA-indexer weight loading with hasattr()
    # instead of the boolean value. If the attribute exists but is False, it can
    # still wrongly attempt to read missing dsa_indexer tensors from checkpoint.
    load_model = getattr(converter, "load_model", None)
    if load_model is not None and hasattr(load_model, "enable_dsa_indexer"):
        if not bool(getattr(load_model, "enable_dsa_indexer")):
            delattr(load_model, "enable_dsa_indexer")
            logging.info("Compatibility: removed disabled enable_dsa_indexer flag on load_model")

    start_time = time.time()
    converter.run()
    end_time = time.time()
    logging.info("time-consuming: %.2fs", end_time - start_time)


if __name__ == "__main__":
    main()
