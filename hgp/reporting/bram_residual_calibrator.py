from __future__ import annotations

import argparse
import csv
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier

from hgp.reporting.checkpoint_prediction_stats import summarize_error_buckets, summarize_values
from hgp.reporting.compare_training_graphs import HGBO_ROOT, ORIGINAL_TARGET_SPECS, _load_split, _target_values


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return HGBO_ROOT / "img" / "training" / f"bram_residual_calibrator_{stamp}"


def graph_features(data, hls_index: int) -> np.ndarray:
    hls_attr = data["hls_attr"].view(-1).detach().cpu().numpy().astype(float)
    if hls_index < 0 or hls_index >= hls_attr.shape[0]:
        raise ValueError(f"hls_index {hls_index} is out of range for hls_attr width {hls_attr.shape[0]}")
    node_features = data.x.detach().cpu().numpy().astype(float)
    num_edges = float(data.edge_index.shape[1])
    return np.concatenate(
        [
            hls_attr,
            node_features.sum(axis=0),
            node_features.max(axis=0),
            node_features.mean(axis=0),
            np.array([float(node_features.shape[0]), num_edges, hls_attr[hls_index]], dtype=float),
        ]
    )


def build_feature_matrix(dataset, hls_index: int = 3):
    spec = ORIGINAL_TARGET_SPECS["bram"]
    rows = []
    true_values = []
    hls_values = []
    residuals = []
    for data in dataset:
        features = graph_features(data, hls_index)
        true_bram = float(_target_values(data, spec).view(-1)[0].item())
        hls_bram = float(data["hls_attr"].view(-1)[hls_index].item())
        rows.append(features)
        true_values.append(true_bram)
        hls_values.append(hls_bram)
        residuals.append(int(round(true_bram - hls_bram)))
    return (
        np.vstack(rows),
        np.asarray(true_values, dtype=float),
        np.asarray(hls_values, dtype=float),
        np.asarray(residuals, dtype=int),
    )


def train_classifier(features: np.ndarray, residual: np.ndarray, *, seed: int, n_estimators: int, n_jobs: int):
    classifier = ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=n_jobs,
        class_weight="balanced",
        min_samples_leaf=1,
    )
    classifier.fit(features, residual)
    return classifier


def evaluate_predictions(*, true: np.ndarray, hls: np.ndarray, residual: np.ndarray, predicted_residual: np.ndarray) -> dict:
    predicted_bram = hls + predicted_residual
    abs_error = np.abs(true - predicted_bram)
    true_tensor = torch.as_tensor(true, dtype=torch.float64)
    pred_tensor = torch.as_tensor(predicted_bram, dtype=torch.float64)
    abs_error_tensor = torch.as_tensor(abs_error, dtype=torch.float64)
    signed_error_tensor = pred_tensor - true_tensor
    return {
        "mae": float(abs_error.mean()),
        "residual_accuracy": float(np.mean(predicted_residual == residual)),
        "true": summarize_values(true_tensor),
        "pred": summarize_values(pred_tensor),
        "abs_error": summarize_values(abs_error_tensor),
        "signed_error": summarize_values(signed_error_tensor),
        "true_buckets": summarize_error_buckets(true_tensor, abs_error_tensor),
        "residual_classes": {
            str(int(value)): int(count) for value, count in zip(*np.unique(residual, return_counts=True))
        },
        "predicted_residual_classes": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(predicted_residual, return_counts=True))
        },
    }


def write_predictions_csv(path: Path, *, true: np.ndarray, hls: np.ndarray, residual: np.ndarray, predicted_residual: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "true_bram", "hls_bram", "true_residual", "predicted_residual", "predicted_bram", "abs_error"],
        )
        writer.writeheader()
        for index, (true_value, hls_value, residual_value, predicted_residual_value) in enumerate(
            zip(true, hls, residual, predicted_residual)
        ):
            predicted_bram = hls_value + predicted_residual_value
            writer.writerow(
                {
                    "index": index,
                    "true_bram": true_value,
                    "hls_bram": hls_value,
                    "true_residual": int(residual_value),
                    "predicted_residual": int(predicted_residual_value),
                    "predicted_bram": predicted_bram,
                    "abs_error": abs(true_value - predicted_bram),
                }
            )


def run(args):
    train_ds, test_ds = _load_split("std", args.seed)
    train_features, train_true, train_hls, train_residual = build_feature_matrix(train_ds, hls_index=args.hls_index)
    test_features, test_true, test_hls, test_residual = build_feature_matrix(test_ds, hls_index=args.hls_index)
    classifier = train_classifier(
        train_features,
        train_residual,
        seed=args.seed,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )
    predicted_residual = classifier.predict(test_features)
    test_result = evaluate_predictions(
        true=test_true,
        hls=test_hls,
        residual=test_residual,
        predicted_residual=predicted_residual,
    )
    train_predicted_residual = classifier.predict(train_features)
    train_result = evaluate_predictions(
        true=train_true,
        hls=train_hls,
        residual=train_residual,
        predicted_residual=train_predicted_residual,
    )
    result = {
        "settings": {
            "seed": args.seed,
            "hls_index": args.hls_index,
            "n_estimators": args.n_estimators,
            "n_jobs": args.n_jobs,
            "train_size": len(train_ds),
            "test_size": len(test_ds),
            "method": "ExtraTreesClassifier on true_bram - hls_attr[hls_index]",
        },
        "train": train_result,
        "test": test_result,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "predictions.csv"
    model_path = output_dir / "bram_residual_calibrator.pkl"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_predictions_csv(
        predictions_path,
        true=test_true,
        hls=test_hls,
        residual=test_residual,
        predicted_residual=predicted_residual,
    )
    with model_path.open("wb") as handle:
        pickle.dump(classifier, handle)
    print(json.dumps({"summary": str(summary_path), "predictions": str(predictions_path), "model": str(model_path)}, indent=2))
    print(f"bram residual calibrator: test MAE = {test_result['mae']:.6g}")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Train a from-scratch BRAM residual calibrator.")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--hls-index", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
