#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Sensitivity benchmark for the TAM sparse sampling factor c.

The paper denotes the probabilistic sparse sampling factor as ``c``. In code,
the same hyperparameter is named ``factor`` in ``ProbAttention``.

This script sweeps c values, measures parameter counts, theoretical attention
MACs, and inference time, then writes publication-friendly CSV/Markdown/JSON
tables. It can also merge accuracy values from a user-provided CSV after the
corresponding trained models are available.

Examples:
    python benchmark_tam_factor_sensitivity.py --device cuda --batch-size 64
    python benchmark_tam_factor_sensitivity.py --datasets CWRU PU --factors 1 3 5 7 9
    python benchmark_tam_factor_sensitivity.py --synthetic-only
    python benchmark_tam_factor_sensitivity.py --accuracy-csv factor_accuracy.csv

Expected accuracy CSV columns:
    dataset,factor,accuracy
or:
    dataset,factor,accuracy_pct
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

from models.TAM import InformerEncoder  # noqa: E402


DATASET_META: Dict[str, Dict[str, object]] = {
    "CWRU": {
        "module_name": "CWRU",
        "data_subdir": "CWRU",
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
    "PU": {
        "module_name": "PU",
        "data_subdir": "PU",
        "transfer_task": [[0], [1]],
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


@dataclass
class SensitivityResult:
    dataset: str
    input_source: str
    c: int
    first_layer_sampled_keys: int
    first_layer_selected_queries: int
    params: int
    params_m: float
    attention_macs: int
    attention_macs_m: float
    mac_reduction_vs_dense_pct: float
    mac_change_vs_c5_pct: float
    batch_size: int
    input_shape: str
    device: str
    warmup: int
    repeats: int
    inference_ms_per_batch: float
    inference_ms_per_sample: float
    speed_vs_c5: Optional[float]
    peak_memory_mb: Optional[float]
    accuracy_pct: Optional[float]


def parse_args():
    parser = argparse.ArgumentParser(description="TAM c/factor sensitivity benchmark.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        choices=sorted(DATASET_META.keys()),
        help="Datasets to benchmark. If omitted, all five paper datasets are used.",
    )
    parser.add_argument("--factors", nargs="+", type=int, default=[1, 3, 5, 7, 9])
    parser.add_argument("--baseline-factor", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
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
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--accuracy-csv",
        type=str,
        default=None,
        help="Optional CSV with columns dataset,factor,accuracy or dataset,factor,accuracy_pct.",
    )
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    return parser.parse_args()


def select_datasets(args) -> List[str]:
    if args.datasets:
        return args.datasets
    return sorted(DATASET_META.keys())


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


def dense_attention_macs(length: int, d_model: int, e_layers: int, distil: bool = True) -> int:
    total = 0
    for layer_len in pooled_lengths(length, e_layers, distil=distil):
        # QK^T and A*V, per sample. H * D_h = d_model.
        total += 2 * d_model * layer_len * layer_len
    return int(total)


def tam_attention_macs(
    length: int,
    d_model: int,
    e_layers: int,
    factor: int,
    distil: bool = True,
) -> int:
    total = 0
    for layer_len in pooled_lengths(length, e_layers, distil=distil):
        u_part = min(factor * math.ceil(math.log(layer_len)), layer_len)
        n_top = min(factor * math.ceil(math.log(layer_len)), layer_len)
        # sampled QK + selected-query full QK + selected-query A*V.
        total += d_model * (layer_len * u_part + 2 * n_top * layer_len)
    return int(total)


def first_layer_sparse_sizes(length: int, factor: int) -> Tuple[int, int]:
    sampled = min(factor * math.ceil(math.log(length)), length)
    selected = min(factor * math.ceil(math.log(length)), length)
    return int(sampled), int(selected)


def build_tam_model(input_channels: int, factor: int, args) -> nn.Module:
    return InformerEncoder(
        enc_in=input_channels,
        factor=factor,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        output_attention=False,
        distil=True,
    )


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
    return batch.to(device), "synthetic:same_shape_as_paper"


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


def load_accuracy_map(path: Optional[str]) -> Dict[Tuple[str, int], float]:
    if not path:
        return {}
    accuracy_path = Path(path)
    if not accuracy_path.exists():
        raise FileNotFoundError(f"Accuracy CSV not found: {accuracy_path}")
    result = {}
    with accuracy_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["dataset"]
            factor = int(row.get("factor", row.get("c", "")))
            if "accuracy_pct" in row and row["accuracy_pct"] != "":
                accuracy_pct = float(row["accuracy_pct"])
            else:
                accuracy = float(row["accuracy"])
                accuracy_pct = accuracy if accuracy > 1.0 else accuracy * 100.0
            result[(dataset, factor)] = accuracy_pct
    return result


def run_dataset(
    dataset_name: str,
    args,
    device: torch.device,
    accuracy_map: Dict[Tuple[str, int], float],
) -> List[SensitivityResult]:
    batch, input_source = make_input(dataset_name, args, device)
    length = int(batch.shape[-1])
    input_channels = int(batch.shape[1])
    dense_macs = dense_attention_macs(length, args.d_model, args.e_layers, distil=True)
    c5_macs = tam_attention_macs(length, args.d_model, args.e_layers, args.baseline_factor, distil=True)

    timings: Dict[int, float] = {}
    memories: Dict[int, Optional[float]] = {}
    params_by_factor: Dict[int, int] = {}
    macs_by_factor: Dict[int, int] = {}
    sparse_sizes: Dict[int, Tuple[int, int]] = {}
    sorted_factors = sorted(dict.fromkeys(args.factors))

    for factor in sorted_factors:
        model = build_tam_model(input_channels, factor, args)
        params_by_factor[factor] = count_params(model)
        macs_by_factor[factor] = tam_attention_macs(length, args.d_model, args.e_layers, factor, distil=True)
        sparse_sizes[factor] = first_layer_sparse_sizes(length, factor)
        timings[factor], memories[factor] = measure_inference(
            model,
            batch,
            device,
            args.warmup,
            args.repeats,
        )

    baseline_time = timings.get(args.baseline_factor)
    results = []
    for factor in sorted_factors:
        sampled_keys, selected_queries = sparse_sizes[factor]
        speed_vs_c5 = None
        if baseline_time is not None and timings[factor] > 0:
            speed_vs_c5 = baseline_time / timings[factor]
        results.append(
            SensitivityResult(
                dataset=dataset_name,
                input_source=input_source,
                c=factor,
                first_layer_sampled_keys=sampled_keys,
                first_layer_selected_queries=selected_queries,
                params=params_by_factor[factor],
                params_m=params_by_factor[factor] / 1_000_000,
                attention_macs=macs_by_factor[factor],
                attention_macs_m=macs_by_factor[factor] / 1_000_000,
                mac_reduction_vs_dense_pct=100.0 * (1.0 - macs_by_factor[factor] / dense_macs),
                mac_change_vs_c5_pct=100.0 * (macs_by_factor[factor] / c5_macs - 1.0),
                batch_size=args.batch_size,
                input_shape=format_shape(batch),
                device=str(device),
                warmup=args.warmup,
                repeats=args.repeats,
                inference_ms_per_batch=timings[factor],
                inference_ms_per_sample=timings[factor] / args.batch_size,
                speed_vs_c5=speed_vs_c5,
                peak_memory_mb=memories[factor],
                accuracy_pct=accuracy_map.get((dataset_name, factor)),
            )
        )
    return results


def value_to_csv(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_csv(results: Sequence[SensitivityResult], path: Path):
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


def write_markdown(results: Sequence[SensitivityResult], path: Path):
    has_accuracy = any(r.accuracy_pct is not None for r in results)
    headers = [
        "Dataset",
        "c",
        "Sampled keys",
        "Selected queries",
        "Params (M)",
        "Attention MACs (M/sample)",
        "MAC reduction vs dense",
        "MAC change vs c=5",
        "ms/batch",
        "Speed vs c=5",
    ]
    if has_accuracy:
        headers.append("Accuracy (%)")
    headers.append("Input")

    rows = []
    for r in results:
        row = [
            r.dataset,
            r.c,
            r.first_layer_sampled_keys,
            r.first_layer_selected_queries,
            f"{r.params_m:.4f}",
            f"{r.attention_macs_m:.3f}",
            fmt_float(r.mac_reduction_vs_dense_pct, 2, "%"),
            fmt_float(r.mac_change_vs_c5_pct, 2, "%"),
            f"{r.inference_ms_per_batch:.3f}",
            fmt_float(r.speed_vs_c5, 3, "x"),
        ]
        if has_accuracy:
            row.append(fmt_float(r.accuracy_pct, 2))
        row.append(r.input_source)
        rows.append(row)

    content = []
    content.append("# TAM Sparse Sampling Factor Sensitivity\n")
    content.append(
        "The hyperparameter c controls both U_part = c*ceil(log L_K) and "
        "n_top = c*ceil(log L_Q). Larger c values use more sampled keys and "
        "selected queries, increasing computation while potentially retaining "
        "more temporal dependency information.\n"
    )
    content.append(
        markdown_table(headers, rows)
    )
    content.append("\n")
    content.append(
        "Accuracy is included only when --accuracy-csv is provided. Without "
        "that file, this script reports the efficiency side of the c sensitivity "
        "analysis: parameter count, attention MACs, and inference time.\n"
    )
    path.write_text("\n".join(content), encoding="utf-8")


def print_summary(results: Sequence[SensitivityResult]):
    has_accuracy = any(r.accuracy_pct is not None for r in results)
    headers = [
        "Dataset",
        "c",
        "Sampled keys",
        "Selected queries",
        "Params (M)",
        "Attention MACs (M/sample)",
        "MAC reduction",
        "ms/batch",
        "Speed vs c=5",
    ]
    if has_accuracy:
        headers.append("Accuracy (%)")
    rows = []
    for r in results:
        row = [
            r.dataset,
            r.c,
            r.first_layer_sampled_keys,
            r.first_layer_selected_queries,
            f"{r.params_m:.4f}",
            f"{r.attention_macs_m:.3f}",
            fmt_float(r.mac_reduction_vs_dense_pct, 2, "%"),
            f"{r.inference_ms_per_batch:.3f}",
            fmt_float(r.speed_vs_c5, 3, "x"),
        ]
        if has_accuracy:
            row.append(fmt_float(r.accuracy_pct, 2))
        rows.append(row)
    print(markdown_table(headers, rows))


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    selected_datasets = select_datasets(args)
    accuracy_map = load_accuracy_map(args.accuracy_csv)

    all_results: List[SensitivityResult] = []
    for dataset_name in selected_datasets:
        all_results.extend(run_dataset(dataset_name, args, device, accuracy_map))

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_tag = "c" + "_".join(str(c) for c in sorted(dict.fromkeys(args.factors)))
    stem = "tam_factor_sensitivity_" + "_".join(selected_datasets) + "_" + factor_tag
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
