#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Benchmark Dense Attention vs. TAM probabilistic sparse attention.

This script is intended for the revision response. It reports:
1) learnable parameter counts,
2) theoretical dominant attention MACs, and
3) measured inference time.

By default, it randomly selects two datasets from the five datasets used in the
paper, tries to load one real batch for each selected dataset, and falls back to
synthetic inputs with the same signal length if local dataset files are absent.

Example:
    python benchmark_tam_efficiency.py --device cuda --batch-size 64
    python benchmark_tam_efficiency.py --datasets CWRU PU --data-root dataset
    python benchmark_tam_efficiency.py --synthetic-only
"""

import argparse
import csv
import importlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.TAM import (  # noqa: E402
    AttentionLayer,
    ConvLayer,
    DataEmbedding,
    Encoder,
    EncoderLayer,
    InformerEncoder,
)


DATASET_META: Dict[str, Dict[str, object]] = {
    "CWRU": {
        "module_name": "CWRU",
        "data_subdir": "CWRU",
        "transfer_task": [[0], [1]],
        "signal_length": 1024,
        "input_channels": 1,
    },
    "PU": {
        "module_name": "PU",
        "data_subdir": "PU",
        "transfer_task": [[0], [1]],
        "signal_length": 1024,
        "input_channels": 1,
    },
    "JNU": {
        "module_name": "JNU",
        "data_subdir": "JNU",
        "transfer_task": [[0], [1]],
        "signal_length": 1024,
        "input_channels": 1,
    },
    "PHM": {
        "module_name": "PHM",
        "data_subdir": "PHM2009",
        "transfer_task": [[0], [3]],
        "signal_length": 1024,
        "input_channels": 1,
    },
    "SEU": {
        "module_name": "SEU",
        "data_subdir": "SEU",
        "transfer_task": [[0], [1]],
        "signal_length": 1024,
        "input_channels": 1,
    },
}


class FullAttention(nn.Module):
    """Dense scaled dot-product attention with the same interface as ProbAttention."""

    def __init__(
        self,
        mask_flag: bool = False,
        factor: int = 5,
        scale: Optional[float] = None,
        attention_dropout: float = 0.0,
        output_attention: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        # queries: [B, L, H, D_h], keys/values: [B, S, H, D_h]
        _, _, _, d_head = queries.shape
        scale = self.scale or 1.0 / math.sqrt(d_head)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                raise ValueError("FullAttention received mask_flag=True but attn_mask=None")
            scores.masked_fill_(attn_mask.mask, -float("inf"))
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values)
        if self.output_attention:
            return out.contiguous(), attn
        return out.contiguous(), None


class DenseInformerEncoder(nn.Module):
    """Informer encoder using dense full attention for fair comparison."""

    def __init__(
        self,
        enc_in: int,
        factor: int = 5,
        d_model: int = 16,
        n_heads: int = 8,
        e_layers: int = 6,
        d_ff: int = 256,
        dropout: float = 0.0,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        distil: bool = True,
    ):
        super().__init__()
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
        )

    def forward(self, x_enc, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = enc_out.view(len(enc_out), -1)
        return enc_out


@dataclass
class BenchmarkResult:
    dataset: str
    input_source: str
    method: str
    params: int
    params_m: float
    attention_macs: int
    attention_macs_m: float
    attention_reduction_vs_dense_pct: Optional[float]
    batch_size: int
    input_shape: str
    device: str
    warmup: int
    repeats: int
    inference_ms_per_batch: float
    inference_ms_per_sample: float
    speedup_vs_dense: Optional[float]
    peak_memory_mb: Optional[float]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Dense Attention vs TAM.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        choices=sorted(DATASET_META.keys()),
        help="Datasets to benchmark. If omitted, two datasets are randomly selected.",
    )
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=2,
        help="Number of datasets to randomly select when --datasets is omitted.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Kept for compatibility. Real data is attempted by default unless --synthetic-only is set.",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip dataset loading and benchmark synthetic inputs with the same paper signal length.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Root directory containing dataset subfolders. Used unless --synthetic-only is set.",
    )
    parser.add_argument("--normlizetype", type=str, default="0-1")
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--factor", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    return parser.parse_args()


def select_datasets(args) -> List[str]:
    if args.datasets:
        return args.datasets
    rng = random.Random(args.seed)
    dataset_names = sorted(DATASET_META.keys())
    n = min(max(args.num_datasets, 1), len(dataset_names))
    return rng.sample(dataset_names, n)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pooled_lengths(length: int, e_layers: int, distil: bool = True) -> List[int]:
    lengths = []
    current = length
    for layer_idx in range(e_layers):
        lengths.append(current)
        if distil and layer_idx < e_layers - 1:
            # MaxPool1d(kernel_size=3, stride=2, padding=1) gives ceil(L / 2).
            current = math.ceil(current / 2)
    return lengths


def attention_macs(
    length: int,
    d_model: int,
    e_layers: int,
    factor: int,
    method: str,
    distil: bool = True,
) -> int:
    total = 0
    for layer_len in pooled_lengths(length, e_layers, distil=distil):
        if method == "DenseAttention":
            # QK^T and A*V, per sample. H * D_h = d_model.
            total += 2 * d_model * layer_len * layer_len
        elif method == "TAM":
            u_part = min(factor * math.ceil(math.log(layer_len)), layer_len)
            n_top = min(factor * math.ceil(math.log(layer_len)), layer_len)
            # sampled QK + selected-query full QK + selected-query A*V.
            total += d_model * (layer_len * u_part + 2 * n_top * layer_len)
        else:
            raise ValueError(f"Unknown method: {method}")
    return int(total)


def build_model(method: str, input_channels: int, args) -> nn.Module:
    common_kwargs = dict(
        enc_in=input_channels,
        factor=args.factor,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        output_attention=False,
        distil=True,
    )
    if method == "TAM":
        return InformerEncoder(**common_kwargs)
    if method == "DenseAttention":
        return DenseInformerEncoder(**common_kwargs)
    raise ValueError(f"Unknown method: {method}")


def try_real_batch(dataset_name: str, args) -> Tuple[Optional[torch.Tensor], str]:
    meta = DATASET_META[dataset_name]
    data_dir = Path(args.data_root) / str(meta["data_subdir"])
    try:
        datasets_module = importlib.import_module("datasets")
        dataset_cls = getattr(datasets_module, str(meta["module_name"]))
        dataset_obj = dataset_cls(
            str(data_dir),
            meta["transfer_task"],
            args.normlizetype,
        )
        splits = dataset_obj.data_split(transfer_learning=True)
        target_val = splits[-1]
        loader = torch.utils.data.DataLoader(
            target_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        batch, _ = next(iter(loader))
        batch = batch.float()
        if batch.size(0) < args.batch_size:
            repeat = math.ceil(args.batch_size / batch.size(0))
            batch = batch.repeat((repeat,) + (1,) * (batch.dim() - 1))[: args.batch_size]
        return batch, f"real:{data_dir}"
    except Exception as exc:  # noqa: BLE001 - fallback is intentional for missing local data.
        return None, f"synthetic:fallback_real_data_error={type(exc).__name__}: {exc}"


def make_input(dataset_name: str, args, device: torch.device) -> Tuple[torch.Tensor, str]:
    meta = DATASET_META[dataset_name]
    if not args.synthetic_only:
        batch, source = try_real_batch(dataset_name, args)
        if batch is not None:
            return batch.to(device), source
    length = int(meta["signal_length"])
    channels = int(meta["input_channels"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + sum(ord(c) for c in dataset_name))
    batch = torch.randn(args.batch_size, channels, length, generator=generator)
    source = "synthetic:same_shape_as_paper"
    return batch.to(device), source


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_inference(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Tuple[float, Optional[float]]:
    model.eval()
    model.to(device)
    batch = batch.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(batch)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(batch)
        synchronize(device)
        elapsed = time.perf_counter() - start
    peak_memory = None
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    return elapsed * 1000.0 / repeats, peak_memory


def format_shape(tensor: torch.Tensor) -> str:
    return "x".join(str(x) for x in tensor.shape)


def run_benchmark_for_dataset(dataset_name: str, args, device: torch.device) -> List[BenchmarkResult]:
    batch, input_source = make_input(dataset_name, args, device)
    length = int(batch.shape[-1])
    input_channels = int(batch.shape[1])

    methods = ["DenseAttention", "TAM"]
    models = {method: build_model(method, input_channels, args) for method in methods}
    params = {method: count_params(model) for method, model in models.items()}
    macs = {
        method: attention_macs(
            length=length,
            d_model=args.d_model,
            e_layers=args.e_layers,
            factor=args.factor,
            method=method,
            distil=True,
        )
        for method in methods
    }

    timings = {}
    memories = {}
    for method in methods:
        try:
            timings[method], memories[method] = measure_inference(
                models[method],
                batch,
                device,
                args.warmup,
                args.repeats,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(
                f"Failed while benchmarking {method} on {dataset_name}. "
                f"Try a smaller --batch-size if this is an out-of-memory error."
            ) from exc

    dense_time = timings["DenseAttention"]
    dense_macs = macs["DenseAttention"]
    results = []
    for method in methods:
        reduction = None
        speedup = None
        if method == "TAM":
            reduction = 100.0 * (1.0 - macs[method] / dense_macs)
            speedup = dense_time / timings[method] if timings[method] > 0 else None
        results.append(
            BenchmarkResult(
                dataset=dataset_name,
                input_source=input_source,
                method=method,
                params=params[method],
                params_m=params[method] / 1_000_000,
                attention_macs=macs[method],
                attention_macs_m=macs[method] / 1_000_000,
                attention_reduction_vs_dense_pct=reduction,
                batch_size=args.batch_size,
                input_shape=format_shape(batch),
                device=str(device),
                warmup=args.warmup,
                repeats=args.repeats,
                inference_ms_per_batch=timings[method],
                inference_ms_per_sample=timings[method] / args.batch_size,
                speedup_vs_dense=speedup,
                peak_memory_mb=memories[method],
            )
        )
    return results


def value_to_csv(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_csv(results: Sequence[BenchmarkResult], path: Path):
    rows = [asdict(result) for result in results]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value_to_csv(value) for key, value in row.items()})


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def fmt_float(value: Optional[float], digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}{suffix}"


def write_markdown(results: Sequence[BenchmarkResult], path: Path):
    rows = []
    for r in results:
        rows.append(
            [
                r.dataset,
                r.method,
                f"{r.params_m:.4f}",
                f"{r.attention_macs_m:.3f}",
                fmt_float(r.attention_reduction_vs_dense_pct, 2, "%"),
                f"{r.inference_ms_per_batch:.3f}",
                f"{r.inference_ms_per_sample:.5f}",
                fmt_float(r.speedup_vs_dense, 3, "x"),
                fmt_float(r.peak_memory_mb, 2),
                r.input_source,
            ]
        )
    content = []
    content.append("# TAM Efficiency Benchmark\n")
    content.append(
        "DenseAttention and TAM use the same embedding, projection, encoder, "
        "convolution, and feed-forward layers. Only the attention kernel is changed.\n"
    )
    content.append(
        "Therefore, the parameter counts are expected to be almost identical; "
        "TAM mainly reduces the dominant attention computation and inference cost.\n"
    )
    content.append(
        markdown_table(
            [
                "Dataset",
                "Method",
                "Params (M)",
                "Attention MACs (M/sample)",
                "MAC reduction",
                "ms/batch",
                "ms/sample",
                "Speedup",
                "Peak mem (MB)",
                "Input",
            ],
            rows,
        )
    )
    content.append("\n")
    content.append(
        "Attention MACs count the dominant QK and AV matrix multiplications. "
        "For DenseAttention, MACs are estimated as 2*d_model*L^2 per encoder layer. "
        "For TAM, MACs are estimated as d_model*(L*U_part + 2*n_top*L), where "
        "U_part = n_top = factor*ceil(log L), clipped by L.\n"
    )
    path.write_text("\n".join(content), encoding="utf-8")


def print_summary(results: Sequence[BenchmarkResult]):
    rows = []
    for r in results:
        rows.append(
            [
                r.dataset,
                r.method,
                f"{r.params_m:.4f}",
                f"{r.attention_macs_m:.3f}",
                fmt_float(r.attention_reduction_vs_dense_pct, 2, "%"),
                f"{r.inference_ms_per_batch:.3f}",
                fmt_float(r.speedup_vs_dense, 3, "x"),
            ]
        )
    print(
        markdown_table(
            [
                "Dataset",
                "Method",
                "Params (M)",
                "Attention MACs (M/sample)",
                "MAC reduction",
                "ms/batch",
                "Speedup",
            ],
            rows,
        )
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    selected_datasets = select_datasets(args)

    all_results: List[BenchmarkResult] = []
    for dataset_name in selected_datasets:
        all_results.extend(run_benchmark_for_dataset(dataset_name, args, device))

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "tam_efficiency_" + "_".join(selected_datasets)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    write_csv(all_results, csv_path)
    write_markdown(all_results, md_path)
    json_path.write_text(
        json.dumps([asdict(r) for r in all_results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_summary(all_results)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
