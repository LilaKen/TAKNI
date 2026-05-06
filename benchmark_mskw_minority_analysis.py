#!/usr/bin/env python
"""Analyze minority-class performance for MSKWM-related experiments.

The script is intentionally separated from the training code. It consumes saved
prediction results or confusion-matrix counts and reports class-wise and
minority-class metrics in ready-to-copy CSV/Markdown/JSON files.

Prediction CSV format:
    dataset,method,y_true,y_pred
    CWRU,TAKNI,0,0
    CWRU,w/o MSKWM,0,1

Long confusion CSV format:
    dataset,method,true_label,pred_label,count
    CWRU,TAKNI,0,0,31
    CWRU,TAKNI,0,1,2

Example:
    python benchmark_mskw_minority_analysis.py \
        --predictions-csv outputs/predictions.csv \
        --baseline-method "w/o MSKWM" \
        --target-method TAKNI
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


CountTable = Dict[str, Dict[str, Dict[str, Dict[str, int]]]]


@dataclass
class ClassMetric:
    dataset: str
    method: str
    class_label: str
    support: int
    predicted: int
    precision: float
    recall: float
    f1: float
    is_minority: bool


@dataclass
class SummaryMetric:
    dataset: str
    method: str
    samples: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    minority_classes: str
    minority_support: int
    minority_recall: Optional[float]
    minority_f1: Optional[float]
    majority_recall: Optional[float]
    worst_class_recall: float


@dataclass
class ComparisonMetric:
    dataset: str
    baseline_method: str
    target_method: str
    delta_accuracy: Optional[float]
    delta_balanced_accuracy: Optional[float]
    delta_macro_f1: Optional[float]
    delta_minority_recall: Optional[float]
    delta_minority_f1: Optional[float]
    delta_worst_class_recall: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute class-wise and minority-class metrics for MSKWM experiments."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--predictions-csv",
        type=Path,
        help="CSV with columns dataset,method,y_true,y_pred.",
    )
    input_group.add_argument(
        "--confusion-csv",
        type=Path,
        help="Long-format confusion CSV with columns dataset,method,true_label,pred_label,count.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/minority_analysis"),
        help="Directory for CSV, Markdown, and JSON outputs.",
    )
    parser.add_argument(
        "--baseline-method",
        default="w/o MSKWM",
        help="Baseline method used in the comparison table.",
    )
    parser.add_argument(
        "--target-method",
        default="TAKNI",
        help="Target method used in the comparison table.",
    )
    parser.add_argument(
        "--minority-mode",
        choices=["below-mean", "below-median", "bottom-k", "explicit"],
        default="below-mean",
        help="How to define minority classes from target-domain support.",
    )
    parser.add_argument(
        "--bottom-k",
        type=int,
        default=2,
        help="Number of lowest-support classes used when --minority-mode bottom-k.",
    )
    parser.add_argument(
        "--minority-classes",
        default="",
        help=(
            "Explicit minority classes, e.g. 'CWRU:7,8;JNU:2,5'. "
            "Use only with --minority-mode explicit."
        ),
    )
    return parser.parse_args()


def nested_counts() -> CountTable:
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))


def read_predictions(path: Path) -> CountTable:
    counts = nested_counts()
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"dataset", "method", "y_true", "y_pred"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            dataset = clean(row["dataset"])
            method = clean(row["method"])
            y_true = clean(row["y_true"])
            y_pred = clean(row["y_pred"])
            counts[dataset][method][y_true][y_pred] += 1
    return counts


def read_confusion_long(path: Path) -> CountTable:
    counts = nested_counts()
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"dataset", "method", "true_label", "pred_label", "count"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            dataset = clean(row["dataset"])
            method = clean(row["method"])
            y_true = clean(row["true_label"])
            y_pred = clean(row["pred_label"])
            counts[dataset][method][y_true][y_pred] += int(float(row["count"]))
    return counts


def clean(value: object) -> str:
    text = "" if value is None else str(value)
    return text.strip()


def parse_explicit_minority(spec: str) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {}
    if not spec.strip():
        return result
    for item in spec.split(";"):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(
                "Invalid --minority-classes format. Expected entries like 'CWRU:7,8'."
            )
        dataset, labels = item.split(":", 1)
        result[clean(dataset)] = {clean(label) for label in labels.split(",") if clean(label)}
    return result


def all_labels(method_counts: Mapping[str, Mapping[str, int]]) -> List[str]:
    labels = set(method_counts.keys())
    for pred_counts in method_counts.values():
        labels.update(pred_counts.keys())
    return sorted(labels, key=natural_key)


def natural_key(value: str) -> Tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def class_supports_for_dataset(dataset_counts: Mapping[str, Mapping[str, Mapping[str, int]]]) -> Dict[str, int]:
    supports: Dict[str, int] = defaultdict(int)
    for method_counts in dataset_counts.values():
        for true_label, pred_counts in method_counts.items():
            supports[true_label] += sum(pred_counts.values())
    return dict(supports)


def choose_minority_classes(
    supports: Mapping[str, int],
    mode: str,
    bottom_k: int,
    explicit: Optional[Set[str]] = None,
) -> Set[str]:
    if explicit is not None:
        return set(explicit)
    if not supports:
        return set()

    labels = sorted(supports, key=lambda label: (supports[label], natural_key(label)))
    values = [supports[label] for label in labels]

    if mode == "bottom-k":
        return set(labels[: max(0, min(bottom_k, len(labels)))])

    if min(values) == max(values):
        return set()

    if mode == "below-median":
        threshold = median(values)
    else:
        threshold = sum(values) / len(values)
    chosen = {label for label, support in supports.items() if support < threshold}

    if not chosen and min(values) < max(values):
        chosen = {label for label, support in supports.items() if support == min(values)}
    return chosen


def median(values: Sequence[int]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def compute_class_metrics(
    counts: CountTable,
    minority_by_dataset: Mapping[str, Set[str]],
) -> List[ClassMetric]:
    rows: List[ClassMetric] = []
    for dataset in sorted(counts):
        for method in sorted(counts[dataset]):
            method_counts = counts[dataset][method]
            labels = all_labels(method_counts)
            for label in labels:
                support = sum(method_counts.get(label, {}).values())
                predicted = sum(method_counts.get(true_label, {}).get(label, 0) for true_label in labels)
                tp = method_counts.get(label, {}).get(label, 0)
                precision = safe_div(tp, predicted)
                recall = safe_div(tp, support)
                f1 = safe_div(2 * precision * recall, precision + recall)
                rows.append(
                    ClassMetric(
                        dataset=dataset,
                        method=method,
                        class_label=label,
                        support=support,
                        predicted=predicted,
                        precision=precision,
                        recall=recall,
                        f1=f1,
                        is_minority=label in minority_by_dataset.get(dataset, set()),
                    )
                )
    return rows


def compute_summary(class_rows: Sequence[ClassMetric], counts: CountTable) -> List[SummaryMetric]:
    class_by_key: Dict[Tuple[str, str], List[ClassMetric]] = defaultdict(list)
    for row in class_rows:
        class_by_key[(row.dataset, row.method)].append(row)

    summaries: List[SummaryMetric] = []
    for dataset in sorted(counts):
        for method in sorted(counts[dataset]):
            rows = class_by_key[(dataset, method)]
            total = sum(row.support for row in rows)
            correct = sum(
                counts[dataset][method].get(row.class_label, {}).get(row.class_label, 0)
                for row in rows
            )
            minority_rows = [row for row in rows if row.is_minority and row.support > 0]
            majority_rows = [row for row in rows if not row.is_minority and row.support > 0]
            summaries.append(
                SummaryMetric(
                    dataset=dataset,
                    method=method,
                    samples=total,
                    accuracy=safe_div(correct, total),
                    balanced_accuracy=mean([row.recall for row in rows if row.support > 0]),
                    macro_f1=mean([row.f1 for row in rows if row.support > 0]),
                    minority_classes=", ".join(row.class_label for row in minority_rows),
                    minority_support=sum(row.support for row in minority_rows),
                    minority_recall=mean_or_none([row.recall for row in minority_rows]),
                    minority_f1=mean_or_none([row.f1 for row in minority_rows]),
                    majority_recall=mean_or_none([row.recall for row in majority_rows]),
                    worst_class_recall=min([row.recall for row in rows if row.support > 0], default=float("nan")),
                )
            )
    return summaries


def compute_comparison(
    summaries: Sequence[SummaryMetric],
    baseline_method: str,
    target_method: str,
) -> List[ComparisonMetric]:
    by_key = {(row.dataset, row.method): row for row in summaries}
    datasets = sorted({row.dataset for row in summaries})
    comparisons: List[ComparisonMetric] = []
    for dataset in datasets:
        baseline = by_key.get((dataset, baseline_method))
        target = by_key.get((dataset, target_method))
        if baseline is None or target is None:
            continue
        comparisons.append(
            ComparisonMetric(
                dataset=dataset,
                baseline_method=baseline_method,
                target_method=target_method,
                delta_accuracy=diff(target.accuracy, baseline.accuracy),
                delta_balanced_accuracy=diff(target.balanced_accuracy, baseline.balanced_accuracy),
                delta_macro_f1=diff(target.macro_f1, baseline.macro_f1),
                delta_minority_recall=diff_optional(target.minority_recall, baseline.minority_recall),
                delta_minority_f1=diff_optional(target.minority_f1, baseline.minority_f1),
                delta_worst_class_recall=diff(target.worst_class_recall, baseline.worst_class_recall),
            )
        )
    return comparisons


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else float("nan")


def mean_or_none(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def diff(a: float, b: float) -> Optional[float]:
    if math.isnan(a) or math.isnan(b):
        return None
    return a - b


def diff_optional(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def write_csv(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    dict_rows = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dict_rows)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, title: str, rows: Sequence[object], columns: Sequence[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = markdown_table(title, rows, columns)
    path.write_text(text, encoding="utf-8")
    return text


def markdown_table(title: str, rows: Sequence[object], columns: Sequence[str]) -> str:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows were generated.")
        return "\n".join(lines) + "\n"
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        data = asdict(row)
        lines.append("| " + " | ".join(format_value(data[col]) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def format_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value * 100:.2f}%"
    return str(value)


def main() -> None:
    args = parse_args()
    if args.predictions_csv:
        counts = read_predictions(args.predictions_csv)
    else:
        counts = read_confusion_long(args.confusion_csv)

    explicit_map = parse_explicit_minority(args.minority_classes)
    minority_by_dataset: Dict[str, Set[str]] = {}
    for dataset, dataset_counts in counts.items():
        supports = class_supports_for_dataset(dataset_counts)
        explicit = explicit_map.get(dataset) if args.minority_mode == "explicit" else None
        minority_by_dataset[dataset] = choose_minority_classes(
            supports=supports,
            mode=args.minority_mode,
            bottom_k=args.bottom_k,
            explicit=explicit,
        )

    class_rows = compute_class_metrics(counts, minority_by_dataset)
    summary_rows = compute_summary(class_rows, counts)
    comparison_rows = compute_comparison(summary_rows, args.baseline_method, args.target_method)

    out_dir = args.output_dir
    write_csv(out_dir / "minority_class_metrics_per_class.csv", class_rows)
    write_csv(out_dir / "minority_class_metrics_summary.csv", summary_rows)
    write_csv(out_dir / "minority_class_metrics_comparison.csv", comparison_rows)
    write_json(
        out_dir / "minority_class_metrics.json",
        {
            "minority_by_dataset": {k: sorted(v, key=natural_key) for k, v in minority_by_dataset.items()},
            "per_class": [asdict(row) for row in class_rows],
            "summary": [asdict(row) for row in summary_rows],
            "comparison": [asdict(row) for row in comparison_rows],
        },
    )

    summary_md = write_markdown(
        out_dir / "minority_class_metrics_summary.md",
        "Minority-Class Summary Metrics",
        summary_rows,
        [
            "dataset",
            "method",
            "samples",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "minority_classes",
            "minority_recall",
            "minority_f1",
            "majority_recall",
            "worst_class_recall",
        ],
    )
    comparison_md = write_markdown(
        out_dir / "minority_class_metrics_comparison.md",
        "Minority-Class Comparison Metrics",
        comparison_rows,
        [
            "dataset",
            "baseline_method",
            "target_method",
            "delta_accuracy",
            "delta_balanced_accuracy",
            "delta_macro_f1",
            "delta_minority_recall",
            "delta_minority_f1",
            "delta_worst_class_recall",
        ],
    )

    print(summary_md)
    print(comparison_md)
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
