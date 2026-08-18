# COMP-4450 Final Course Project
## Production-Grade Toxic Comment Moderation MLOps System

> **Selected problem:** Toxic Comment Moderation  
> **Dataset:** Jigsaw Toxic Comment Classification Challenge  
> **Model:** TF-IDF + One-vs-Rest Logistic Regression

## Project objective

This repository implements an end-to-end MLOps system that tracks experiments, versions and registers
models, serves a selected Registry model through FastAPI, persists production prediction telemetry in
Amazon DynamoDB, exposes a user-facing Streamlit interface, provides a separate database-backed monitoring
dashboard, validates changes with pytest/Ruff in GitHub Actions, and deploys the three application components
as Docker containers on separate Amazon EC2 instances.

## Architecture

```text
Jigsaw train.csv
      |
      v
Training / evaluation
      |
      +--> W&B experiment runs
      |      - Git commit
      |      - hyperparameters
      |      - data SHA-256
      |      - metrics
      |
      v
W&B model artifact --> W&B Registry --> production alias
                                      |
                                      v
                               EC2 #1 FastAPI
                               /health /predict /feedback
                                      |
                                      v
                                   DynamoDB
                                  /        \
                                 /          \
                                v            v
                    EC2 #2 Streamlit    EC2 #3 Streamlit
                    user frontend       monitoring dashboard
```

The final architecture deliberately does **not** exchange prediction data through JSON files or shared Docker
volumes. FastAPI writes prediction records to DynamoDB; the separate monitoring application reads DynamoDB.

## Repository structure

```text
training/                   model training + W&B tracking/Registry linking
backend/                    FastAPI service and DynamoDB persistence
frontend/                   user-facing Streamlit app
monitoring/                 separate Streamlit monitoring dashboard
tests/unit/                 preprocessing/persistence unit tests
tests/integration/          FastAPI endpoint tests
infra/                      DynamoDB creation + IAM policy templates
scripts/                    smoke test and demo-traffic generator
.github/workflows/ci.yml    PR lint/test workflow
docs/screenshots/           submission evidence
PROJECT_WALKTHROUGH.md      full implementation/deployment walkthrough
PROJECT_REQUIREMENTS_CHECKLIST.md rubric audit
```

## Model

The model is a six-label multilabel classifier for:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

A scikit-learn pipeline performs TF-IDF vectorization followed by six binary Logistic Regression classifiers
through `OneVsRestClassifier`. The design keeps model training lightweight so the project effort stays focused
on the required MLOps lifecycle.

Tracked validation metrics include exact-match accuracy, micro/macro F1, per-label precision/recall/F1, and
per-label/mean ROC-AUC where defined.

## Data setup

Download the Jigsaw Toxic Comment Classification Challenge data and place:

```text
data/raw/train.csv
```

The dataset is intentionally excluded from Git.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r training/requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env
```

Configure the W&B values in `.env`, then:

```bash
wandb login
```

## Train and track experiments

Example baseline:

```bash
python training/train.py \
  --run-name baseline-unigram \
  --ngram-max 1 \
  --c 1.0 \
  --max-features 50000
```

Example candidates:

```bash
python training/train.py --run-name bigram-c1 --ngram-max 2 --c 1.0 --max-features 75000
python training/train.py --run-name bigram-balanced --ngram-max 2 --c 1.0 \
  --max-features 75000 --class-weight balanced
```

Every run records the current Git SHA, dataset SHA-256, training configuration, and validation metrics. The
trained joblib bundle is logged as a W&B model artifact and linked to the configured Registry collection.

After comparing runs, assign the best linked artifact version the Registry alias `production`.

## FastAPI

Required endpoints:

- `GET /health`
- `POST /predict`

Additional human-review endpoint:

- `POST /feedback`

Prediction example:

```bash
curl -X POST http://HOST:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"You are being extremely rude."}'
```

Example response shape:

```json
{
  "prediction_id": "uuid",
  "predicted_labels": ["toxic", "insult"],
  "is_toxic": true,
  "probabilities": {
    "toxic": 0.91,
    "severe_toxic": 0.03,
    "obscene": 0.10,
    "threat": 0.01,
    "insult": 0.82,
    "identity_hate": 0.02
  },
  "inference_latency_ms": 12.5,
  "model_ref": "wandb-registry-Models/toxic-comment-classifier:production",
  "model_version": "v3"
}
```

Each successful prediction is written once to DynamoDB with its timestamp, request text, text length,
predicted labels, probabilities, model identity, latency, and feedback state.

## DynamoDB

Production table:

```text
comp4450-toxic-comment-predictions
```

Partition key:

```text
prediction_id (String)
```

Create it with:

```bash
python infra/create_dynamodb_table.py
```

or through the AWS console using on-demand capacity.

## User frontend

The user-facing Streamlit app:

1. accepts comment text;
2. sends it to FastAPI;
3. displays labels/probabilities;
4. records human feedback through FastAPI.

It does not load the model or write DynamoDB directly.

## Monitoring dashboard

The separate monitoring app connects directly to DynamoDB and displays:

- total predictions;
- average/P95 inference latency;
- latency over time;
- predicted-label distribution;
- feedback coverage;
- live feedback-derived accuracy;
- cumulative accuracy over time;
- earlier-vs-recent input-length drift;
- model-version distribution.

## Tests

```bash
pytest
```

Tests use dependency injection/fakes, so CI does not need AWS credentials, W&B credentials, or a live model
Registry.

## Lint

```bash
ruff check .
```

## CI

`.github/workflows/ci.yml` runs automatically for pull requests targeting `main` and executes both Ruff and
the full pytest suite. Configure the repository's branch protection/ruleset so the `lint-and-test` check is
required before merge.

## Docker

```bash
docker build -t comp4450-toxic-api ./backend
docker build -t comp4450-toxic-frontend ./frontend
docker build -t comp4450-toxic-monitoring ./monitoring
```

`docker-compose.local.yml` exists only for local integration work. The final AWS implementation uses separate
EC2 instances.

## AWS deployment

Production layout:

| Component | Deployment |
|---|---|
| FastAPI backend | EC2 #1 + Docker |
| User Streamlit frontend | EC2 #2 + Docker |
| Monitoring Streamlit dashboard | EC2 #3 + Docker |
| Prediction/feedback persistence | Amazon DynamoDB |
| Experiment tracking/model registry | Weights & Biases |

Attach DynamoDB permissions to EC2 with IAM roles/instance profiles rather than embedding AWS access keys in
containers. The backend's W&B API key is supplied as a runtime secret/environment variable and is never
committed.

See **[PROJECT_WALKTHROUGH.md](PROJECT_WALKTHROUGH.md)** for exact deployment order and commands.

## Required submission links

- **Public GitHub repository:** `REPLACE_WITH_PUBLIC_GITHUB_URL`
- **Public W&B project dashboard:** `REPLACE_WITH_PUBLIC_WANDB_URL`

## Submission evidence

Add screenshots under `docs/screenshots/` showing the AWS console and live system, W&B experiments/Registry,
GitHub CI, and working user/monitoring applications.

## Limitations and responsible-use notes

This baseline model learns lexical/statistical correlations from historical labeled comments. It can produce false
positives and false negatives, and toxicity systems can behave unevenly around identity-related language. Human
feedback in this project is a monitoring signal, not automatically trusted training ground truth. Production systems
should apply stronger privacy, retention, access-control, fairness evaluation, and abuse-review processes than this
course demonstration.
