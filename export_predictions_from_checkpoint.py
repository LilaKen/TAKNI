#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export target-domain predictions from a saved TAKNI checkpoint.

The loading flow follows the visualization scripts in UDTL-master:
1) rebuild the same feature extractor, bottleneck, and classifier modules;
2) load the saved ``model_all.state_dict()`` checkpoint;
3) run inference on a selected dataset split;
4) export prediction rows for downstream confusion-matrix or minority-class
   analysis.

Example:
    python export_predictions_from_checkpoint.py \
        --checkpoint checkpoint/tam_features_1d_CWRU_0_to_1_2023_True_CDA+E_True_MSKW_adam_step/120-0.9900-best_model.pth \
        --model-name tam_features_1d \
        --data-name CWRU \
        --data-dir dataset/CWRU \
        --transfer-task "[[0],[1]]" \
        --method TAKNI \
        --output-csv outputs/predictions.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datasets  # noqa: E402
import models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a TAKNI checkpoint and export target-domain predictions."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a saved *-best_model.pth file.",
    )
    parser.add_argument(
        "--model-name",
        default="tam_features_1d",
        help="Model name registered in models/__init__.py.",
    )
    parser.add_argument(
        "--data-name",
        required=True,
        help="Dataset name registered in datasets/__init__.py, e.g. CWRU, CWRUFFT, PHM.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root directory used by the dataset class.",
    )
    parser.add_argument(
        "--transfer-task",
        default="[[0],[1]]",
        help='Python-style transfer task, for example "[[0],[1]]" or "[[0],[3]]".',
    )
    parser.add_argument(
        "--normlizetype",
        default="0-1",
        help="Normalization type used by the original training script.",
    )
    parser.add_argument(
        "--split",
        choices=["source_train", "source_val", "target_train", "target_val"],
        default="target_val",
        help="Dataset split used for inference.",
    )
    parser.add_argument(
        "--method",
        default="TAKNI",
        help="Method name written into the exported CSV.",
    )
    parser.add_argument(
        "--dataset-alias",
        default=None,
        help="Optional dataset name written into CSV. Defaults to --data-name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Pass pretrained=True when constructing the feature extractor.",
    )
    parser.add_argument(
        "--no-bottleneck",
        action="store_true",
        help="Use this only if the checkpoint was trained without the bottleneck layer.",
    )
    parser.add_argument(
        "--bottleneck-num",
        type=int,
        default=256,
        help="Bottleneck feature dimension used during training.",
    )
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Load with strict=False. Use only for debugging architecture mismatches.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/predictions.csv"),
        help="Prediction CSV path. Rows are appended unless --overwrite is set.",
    )
    parser.add_argument(
        "--output-confusion-csv",
        type=Path,
        default=None,
        help="Optional long-form confusion-count CSV path.",
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files instead of appending.",
    )
    return parser.parse_args()


def parse_transfer_task(raw: str) -> Sequence[Sequence[int]]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid --transfer-task value: {raw}") from exc
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("--transfer-task must be like [[0],[1]].")
    return value


def build_model(args: argparse.Namespace) -> nn.Module:
    dataset_cls = getattr(datasets, args.data_name)
    feature_extractor = getattr(models, args.model_name)(args.pretrained)

    if args.no_bottleneck:
        classifier_layer = nn.Linear(feature_extractor.output_num(), dataset_cls.num_classes)
        return nn.Sequential(feature_extractor, classifier_layer)

    bottleneck_layer = nn.Sequential(
        nn.Linear(feature_extractor.output_num(), args.bottleneck_num),
        nn.ReLU(inplace=True),
        nn.Dropout(),
    )
    classifier_layer = nn.Linear(args.bottleneck_num, dataset_cls.num_classes)
    return nn.Sequential(feature_extractor, bottleneck_layer, classifier_layer)


def extract_state_dict(raw_checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(raw_checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = raw_checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        return raw_checkpoint
    raise TypeError("The checkpoint does not contain a state_dict mapping.")


def normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        new_key = new_key.replace(".module.", ".")
        normalized[new_key] = value
    return normalized


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device, allow_partial: bool) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=not allow_partial)
    if allow_partial:
        if incompatible.missing_keys:
            print("Missing keys:", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print("Unexpected keys:", incompatible.unexpected_keys)


def build_dataloader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    transfer_task = parse_transfer_task(args.transfer_task)
    dataset_cls = getattr(datasets, args.data_name)
    split_data = dataset_cls(str(args.data_dir), transfer_task, args.normlizetype).data_split(
        transfer_learning=True
    )
    split_names = ["source_train", "source_val", "target_train", "target_val"]
    datasets_by_name = dict(zip(split_names, split_data))
    selected_dataset = datasets_by_name[args.split]
    return torch.utils.data.DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )


def append_prediction_rows(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite or not path.exists() else "a"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "method", "y_true", "y_pred"])
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def write_confusion_counts(
    path: Path,
    dataset_name: str,
    method_name: str,
    pairs: Iterable[tuple[int, int]],
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(pairs)
    mode = "w" if overwrite or not path.exists() else "a"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "method", "true_label", "pred_label", "count"],
        )
        if mode == "w":
            writer.writeheader()
        for (true_label, pred_label), count in sorted(counts.items()):
            writer.writerow(
                {
                    "dataset": dataset_name,
                    "method": method_name,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": count,
                }
            )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset_name = args.dataset_alias or args.data_name

    model = build_model(args).to(device)
    load_checkpoint(model, args.checkpoint, device, args.allow_partial_load)
    model.eval()

    dataloader = build_dataloader(args)

    rows = []
    label_pred_pairs = []
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            for true_label, pred_label in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "method": args.method,
                        "y_true": int(true_label),
                        "y_pred": int(pred_label),
                    }
                )
                label_pred_pairs.append((int(true_label), int(pred_label)))

    append_prediction_rows(args.output_csv, rows, args.overwrite)
    if args.output_confusion_csv is not None:
        write_confusion_counts(
            args.output_confusion_csv,
            dataset_name,
            args.method,
            label_pred_pairs,
            args.overwrite,
        )

    summary: MutableMapping[str, object] = {
        "dataset": dataset_name,
        "method": args.method,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "samples": total,
        "accuracy": correct / total if total else None,
        "output_csv": str(args.output_csv),
    }
    if args.output_confusion_csv is not None:
        summary["output_confusion_csv"] = str(args.output_confusion_csv)
    if args.output_summary_json is not None:
        args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
