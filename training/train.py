"""Train, evaluate, track, version, and register the Jigsaw toxicity model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/train.csv")
    parser.add_argument("--output", default="artifacts/toxic_comment_model.joblib")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-features", type=int, default=75_000)
    parser.add_argument("--ngram-max", type=int, choices=[1, 2], default=2)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--no-registry-link", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            "Training must run inside the Git repository so the required code version can be logged."
        ) from exc


def load_training_data(path: Path, sample_fraction: float, seed: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download Jigsaw train.csv and place it at data/raw/train.csv."
        )
    frame = pd.read_csv(path)
    required = {"comment_text", *LABELS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in the interval (0, 1].")
    if sample_fraction < 1:
        frame = frame.sample(frac=sample_fraction, random_state=seed).reset_index(drop=True)
    frame["comment_text"] = frame["comment_text"].fillna("").astype(str)
    return frame


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    class_weight = None if args.class_weight == "none" else args.class_weight
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    sublinear_tf=True,
                    min_df=2,
                    max_df=0.98,
                    max_features=args.max_features,
                    ngram_range=(1, args.ngram_max),
                    dtype=np.float32,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        C=args.c,
                        max_iter=1000,
                        solver="liblinear",
                        class_weight=class_weight,
                        random_state=args.seed,
                    ),
                    n_jobs=1,
                ),
            ),
        ]
    )


def calculate_metrics(y_true: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    y_array = y_true.to_numpy(dtype=int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_array, predictions, average=None, zero_division=0
    )

    metrics: dict[str, float] = {
        "validation_exact_match_accuracy": float(accuracy_score(y_array, predictions)),
        "validation_micro_f1": float(f1_score(y_array, predictions, average="micro", zero_division=0)),
        "validation_macro_f1": float(f1_score(y_array, predictions, average="macro", zero_division=0)),
    }

    auc_values = []
    for index, label in enumerate(LABELS):
        metrics[f"validation_{label}_precision"] = float(precision[index])
        metrics[f"validation_{label}_recall"] = float(recall[index])
        metrics[f"validation_{label}_f1"] = float(f1[index])
        if len(np.unique(y_array[:, index])) == 2:
            auc = float(roc_auc_score(y_array[:, index], probabilities[:, index]))
            metrics[f"validation_{label}_roc_auc"] = auc
            auc_values.append(auc)

    if auc_values:
        metrics["validation_mean_roc_auc"] = float(np.mean(auc_values))
    return metrics


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    load_dotenv()
    args = parse_args()
    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    commit = git_commit()
    dataset_hash = sha256_file(data_path)
    frame = load_training_data(data_path, args.sample_fraction, args.seed)

    x_train, x_val, y_train, y_val = train_test_split(
        frame["comment_text"],
        frame[LABELS],
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )

    pipeline = build_pipeline(args)
    pipeline.fit(x_train, y_train)
    probabilities = np.asarray(pipeline.predict_proba(x_val), dtype=float)
    metrics = calculate_metrics(y_val, probabilities, args.threshold)

    trained_at = datetime.now(UTC).isoformat()
    bundle = {
        "pipeline": pipeline,
        "labels": LABELS,
        "threshold": args.threshold,
        "git_commit": commit,
        "dataset_sha256": dataset_hash,
        "trained_at": trained_at,
        "sklearn_version": sklearn_version,
        "metrics": metrics,
    }
    joblib.dump(bundle, output_path)

    dataset_manifest = {
        "name": "Jigsaw Toxic Comment Classification Challenge",
        "source_file": data_path.name,
        "sha256": dataset_hash,
        "rows_used": int(len(frame)),
        "sample_fraction": args.sample_fraction,
        "labels": LABELS,
        "split_seed": args.seed,
        "test_size": args.test_size,
    }
    metrics_path = output_path.parent / "latest_metrics.json"
    manifest_path = output_path.parent / "dataset_manifest.json"
    write_json(metrics_path, metrics)
    write_json(manifest_path, dataset_manifest)

    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT", "comp4450-toxic-comment-mlops")
    registry = os.getenv("WANDB_REGISTRY", "Models")
    collection = os.getenv("WANDB_COLLECTION", "toxic-comment-classifier")
    if not entity:
        raise RuntimeError("WANDB_ENTITY must be set to the W&B team/entity that owns the Registry.")

    import wandb

    run_name = args.run_name or (
        f"tfidf-logreg-c{args.c}-ngram{args.ngram_max}-cw{args.class_weight}"
    )
    config = {
        "model_family": "TF-IDF + OneVsRest LogisticRegression",
        "git_commit": commit,
        "dataset_sha256": dataset_hash,
        "dataset_rows_used": int(len(frame)),
        "test_size": args.test_size,
        "split_seed": args.seed,
        "C": args.c,
        "max_features": args.max_features,
        "ngram_range": [1, args.ngram_max],
        "threshold": args.threshold,
        "class_weight": args.class_weight,
        "sklearn_version": sklearn_version,
    }

    with wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        job_type="training",
        config=config,
    ) as run:
        run.log(metrics)

        dataset_artifact = wandb.Artifact(
            name="jigsaw-toxic-comment-dataset-metadata",
            type="dataset-metadata",
            metadata=dataset_manifest,
        )
        dataset_artifact.add_file(str(manifest_path))
        run.log_artifact(dataset_artifact)

        model_artifact = wandb.Artifact(
            name="toxic-comment-model",
            type="model",
            metadata={
                "git_commit": commit,
                "dataset_sha256": dataset_hash,
                "validation_micro_f1": metrics["validation_micro_f1"],
                "validation_macro_f1": metrics["validation_macro_f1"],
                "trained_at": trained_at,
            },
        )
        model_artifact.add_file(str(output_path))
        model_artifact.add_file(str(metrics_path))
        logged_artifact = run.log_artifact(model_artifact)

        if not args.no_registry_link:
            run.link_artifact(
                artifact=logged_artifact,
                target_path=f"wandb-registry-{registry}/{collection}",
            )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved model bundle: {output_path}")
    print(f"Git commit: {commit}")
    print(f"Dataset SHA-256: {dataset_hash}")
    if not args.no_registry_link:
        print(f"Linked artifact to W&B Registry collection: {registry}/{collection}")
        print("Promote the best linked version by assigning the 'production' alias in W&B Registry.")


if __name__ == "__main__":
    main()
