#!/usr/bin/env python3
"""Convert raw JSON/JSONL text samples into Megatron indexed dataset files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _load_records(path: Path) -> list[object]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if isinstance(data.get("data"), list):
                return list(data["data"])
            return [data]
        raise ValueError(f"不支持的 JSON 顶层结构: {type(data).__name__}")

    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    records.append(json.loads(text))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL 第 {line_no} 行解析失败: {exc}") from exc
        return records

    raise ValueError(f"暂不支持的数据集后缀: {suffix}")


def _record_to_text(record: object) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        if isinstance(record.get("text"), str):
            return record["text"]
        if isinstance(record.get("content"), str):
            return record["content"]
        messages = record.get("messages")
        if isinstance(messages, list):
            chunks = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "unknown").strip()
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    chunks.append(f"{role}: {content}")
            if chunks:
                return "\n".join(chunks)
        instruction = record.get("instruction")
        output = record.get("output")
        input_text = record.get("input")
        if isinstance(instruction, str) and isinstance(output, str):
            parts = [instruction]
            if isinstance(input_text, str) and input_text.strip():
                parts.append(input_text)
            parts.append(output)
            return "\n\n".join(parts)
    raise ValueError(f"无法从记录中提取文本内容: {type(record).__name__}")


def _load_tokenizer(tokenizer_type: str, tokenizer_source: str):
    if tokenizer_type == "PretrainedFromHF":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

        def encode_text(text: str) -> list[int]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not isinstance(ids, list):
                ids = list(ids)
            return [int(token) for token in ids]

        vocab_size = getattr(tokenizer, "vocab_size", None)
        return encode_text, vocab_size

    if tokenizer_type == "Llama2Tokenizer":
        import sentencepiece as spm

        processor = spm.SentencePieceProcessor()
        if not processor.load(tokenizer_source):
            raise RuntimeError(f"无法加载 sentencepiece tokenizer: {tokenizer_source}")

        def encode_text(text: str) -> list[int]:
            ids = processor.encode(text, out_type=int)
            return [int(token) for token in ids]

        return encode_text, int(processor.vocab_size())

    raise ValueError(f"暂不支持的 tokenizer 类型: {tokenizer_type}")


def _build_indexed_dataset(
    records: list[object],
    output_prefix: Path,
    encode_text,
    vocab_size,
) -> int:
    from megatron.core.datasets import indexed_dataset

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = str(output_prefix.with_suffix(".bin"))
    idx_path = str(output_prefix.with_suffix(".idx"))

    builder_name = ""
    if hasattr(indexed_dataset, "IndexedDatasetBuilder"):
        builder = indexed_dataset.IndexedDatasetBuilder(bin_path, dtype=np.int32)
        builder_name = "IndexedDatasetBuilder"
    elif hasattr(indexed_dataset, "MMapIndexedDatasetBuilder"):
        kwargs = {}
        if vocab_size is not None:
            kwargs["vocab_size"] = int(vocab_size)
        builder = indexed_dataset.MMapIndexedDatasetBuilder(bin_path, **kwargs)
        builder_name = "MMapIndexedDatasetBuilder"
    elif hasattr(indexed_dataset, "make_builder"):
        builder = indexed_dataset.make_builder(
            bin_path,
            impl="mmap",
            vocab_size=vocab_size,
        )
        builder_name = "make_builder"
    else:
        available = sorted(name for name in dir(indexed_dataset) if "Builder" in name or "builder" in name)
        raise RuntimeError(
            "当前 megatron.core.datasets.indexed_dataset 不包含可用的 builder 实现: "
            f"{available}"
        )

    written = 0
    for record in records:
        text = _record_to_text(record).strip()
        if not text:
            continue
        token_ids = encode_text(text)
        if not token_ids:
            continue
        tensor = torch.tensor(token_ids, dtype=torch.int32)
        if hasattr(builder, "add_document"):
            builder.add_document(tensor, [len(token_ids)])
        elif hasattr(builder, "add_item") and hasattr(builder, "end_document"):
            builder.add_item(tensor)
            builder.end_document()
        else:
            raise RuntimeError(f"builder {type(builder).__name__} 不支持 add_document / add_item")
        written += 1

    if written == 0:
        raise RuntimeError("没有可写入的有效样本，无法生成 indexed dataset")

    builder.finalize(idx_path)
    print(
        f"[prepare_indexed_dataset] builder={builder_name} bin={bin_path} idx={idx_path}",
        flush=True,
    )
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw dataset path (.json or .jsonl)")
    parser.add_argument("--output-prefix", required=True, help="Output dataset prefix without suffix")
    parser.add_argument("--tokenizer-type", required=True, help="Tokenizer type from training template")
    parser.add_argument("--tokenizer-source", required=True, help="Tokenizer path or model file")
    parser.add_argument("--model-name", default="", help="Model name for logging")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    records = _load_records(input_path)
    encode_text, vocab_size = _load_tokenizer(args.tokenizer_type, args.tokenizer_source)
    written = _build_indexed_dataset(records, output_prefix, encode_text, vocab_size)
    print(
        f"[prepare_indexed_dataset] model={args.model_name or '-'} input={input_path} "
        f"output={output_prefix} records={written}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
