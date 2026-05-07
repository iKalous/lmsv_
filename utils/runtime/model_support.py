"""Model support constraints for runtime conversion flows."""

from pathlib import Path

TASK1_WEIGHT_CONVERT_SUPPORTED_MODELS = ("qwen3","deepseekv3")
TASK1_WEIGHT_CONVERT_MODEL_ALIASES = {
    "deepseekv3": "deepseek3",
}
TASK1_WEIGHT_CONVERT_ACCEPTED_MODELS = tuple(
    sorted(set(TASK1_WEIGHT_CONVERT_SUPPORTED_MODELS) | set(TASK1_WEIGHT_CONVERT_MODEL_ALIASES.values()))
)


def list_task1_template_supported_models(templates_dir):
    root = Path(templates_dir)
    if not root.exists():
        return tuple()
    models = []
    for path in sorted(root.glob("pretrain_mutated_*.sh")):
        name = path.stem
        prefix = "pretrain_mutated_"
        if name.startswith(prefix):
            models.append(name[len(prefix):].strip().lower())
    return tuple(models)


def ensure_task1_model_supported_for_mode(model_name, compare_mode, templates_dir):
    model = "" if model_name is None else str(model_name).strip().lower()
    mode = "" if compare_mode is None else str(compare_mode).strip().lower()

    if mode == "pta_mf":
        return ensure_task1_weight_convert_model_supported(model)

    if mode != "pta_msa":
        raise ValueError(f"不支持的COMPARE_MODE: {compare_mode}，仅支持 pta_msa/pta_mf")

    supported = list_task1_template_supported_models(templates_dir)
    if model in supported:
        return model
    supported_text = ", ".join(supported) if supported else "<empty>"
    raise ValueError(
        f"不支持的模型类型: {model_name}，在 pta_msa 模式下仅支持 scripts/templates/pretrain_example 中已有模板模型: {supported_text}"
    )


def is_task1_weight_convert_model_supported(model_name):
    if model_name is None:
        return False
    return str(model_name).strip().lower() in TASK1_WEIGHT_CONVERT_SUPPORTED_MODELS


def ensure_task1_weight_convert_model_supported(model_name):
    model = "" if model_name is None else str(model_name).strip().lower()
    if model in TASK1_WEIGHT_CONVERT_SUPPORTED_MODELS:
        return model
    supported = ", ".join(TASK1_WEIGHT_CONVERT_SUPPORTED_MODELS)
    raise ValueError(f"不支持的模型类型: {model_name}，当前仅支持: {supported}")


def resolve_task1_weight_convert_model_alias(model_name):
    """Map LMSV model names to converter-accepted model names."""
    model = "" if model_name is None else str(model_name).strip().lower()
    if model in TASK1_WEIGHT_CONVERT_SUPPORTED_MODELS:
        return TASK1_WEIGHT_CONVERT_MODEL_ALIASES.get(model, model)
    if model in TASK1_WEIGHT_CONVERT_MODEL_ALIASES.values():
        return model
    supported = ", ".join(TASK1_WEIGHT_CONVERT_ACCEPTED_MODELS)
    raise ValueError(f"不支持的模型类型: {model_name}，当前仅支持: {supported}")
