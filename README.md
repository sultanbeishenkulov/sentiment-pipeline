# Real-Time Social Media Sentiment Pipeline

Streaming pipeline that pulls live posts from Bluesky, scores them with a fine-tuned DistilBERT model, and serves the results through a dashboard. The whole stack runs with one command via Docker Compose.

Started this project to learn Kafka and streaming pipelines hands-on. Originally planned to use Reddit's API but their access process is effectively broken post-2023, so I pivoted to Bluesky's AT Protocol firehose — which turned out to be a better fit anyway (higher volume, no auth needed, true real-time stream via websocket). The ML side uses a DistilBERT model I fine-tuned on tweet_eval; accuracy isn't state-of-the-art but the model is honest about its confidence and the pipeline is transparent about its limitations.

## Architecture

```
Bluesky Firehose (AT Protocol)
        │
        ▼
  ┌──────────┐     ┌─────────┐     ┌────────────┐     ┌────────────┐
  │ Producer  │────▶│  Kafka  │────▶│  Consumer   │────▶│ PostgreSQL │
  │ (Python)  │     │ (Broker)│     │ (DistilBERT)│     │            │
  └──────────┘     └─────────┘     └────────────┘     └─────┬──────┘
                                                            │
                                                            ▼
                                                    ┌──────────────┐
                                                    │   FastAPI     │
                                                    │  (REST API)   │
                                                    └──────┬───────┘
                                                           │
                                                           ▼
                                                    ┌──────────────┐
                                                    │  Streamlit    │
                                                    │ (Dashboard)   │
                                                    └──────────────┘
```

## Quick Start

You need [Docker](https://docs.docker.com/get-docker/) installed. Then:

```bash
git clone https://github.com/sultanbeishenkulov/sentiment-pipeline.git
cd sentiment-pipeline
docker compose up --build
```

Seven containers start up. Once you see logs from the producer and consumer, open:
- **Dashboard**: http://localhost:8501
- **API docs**: http://localhost:8000/docs

Posts start flowing within seconds. The dashboard auto-refreshes every 10 seconds.

## How It Works

**Producer** connects to Bluesky's public firehose via websocket. The firehose sends every event on the network (posts, likes, follows, everything) as CBOR-encoded binary. The producer decodes these, filters for English-only text posts, and publishes them to Kafka at ~30 posts/sec. The websocket runs on a background thread with a bounded queue between it and the Kafka producer — if Kafka falls behind, we drop overflow rather than blocking the websocket (which would disconnect us).

**Consumer** reads from Kafka in batches of 32 (or a 2-second timeout, whichever comes first) and runs them through DistilBERT in a single forward pass. Batching is the single biggest throughput lever — 32 messages in one pass is dramatically faster than 32 individual predictions. After scoring, it writes to Postgres with `ON CONFLICT DO NOTHING` and only then commits the Kafka offsets. That ordering matters: if the process crashes between the DB write and the commit, we'll reprocess those messages on restart, and the unique constraint silently absorbs the duplicates. This is the at-least-once delivery pattern.

**API** is a FastAPI service with three endpoints: `/stats` (overall distribution), `/trends` (time-bucketed counts), and `/recent` (latest posts with optional label filtering). Uses a psycopg connection pool so we're not opening a new Postgres connection on every request.

**Dashboard** is a Streamlit app that polls the API. Metric cards, a donut chart for distribution, a line chart for trends over time, and a scrollable feed of recent posts with their sentiment scores.

## Model

DistilBERT fine-tuned on [tweet_eval](https://huggingface.co/datasets/tweet_eval) sentiment (3-class: positive, neutral, negative). Trained with HuggingFace's Trainer API, early stopping on validation macro-F1 with patience of 2, learning rate 2e-5, batch size 32.

Test set results (12,284 examples):

```
              precision    recall    f1-score
    negative     0.75        0.57      0.65
     neutral     0.67        0.70      0.68
    positive     0.57        0.76      0.65

    accuracy                           0.667
    macro F1                           0.662
```

The negative class has about half as many training examples as the other two (~10k vs ~17k), which shows in the results — the model is cautious about predicting negative (high precision, lower recall). When it says something is negative, it's usually right, but it misses a lot of actual negatives by bucketing them as neutral.

These numbers are in line with published tweet_eval benchmarks for DistilBERT. Could push F1 up a few points with class weighting or a larger base model (RoBERTa), but that wasn't the point of this project.

To retrain:

```bash
cd model
pip install -r requirements.txt
python train.py       # ~10-15 min on Apple Silicon, ~30-40 min on CPU
python evaluate.py    # prints classification report + saves confusion matrix
```

The consumer automatically picks up the trained model from `model/artifacts/` on startup. If artifacts are missing, it falls back to a pretrained SST-2 model from HuggingFace so the pipeline is runnable even before training.

## Project Structure

```
sentiment-pipeline/
├── producer/              # Bluesky firehose → Kafka
├── consumer/              # Kafka → DistilBERT → Postgres
├── model/                 # Training, evaluation, artifacts
├── api/                   # FastAPI endpoints
├── dashboard/             # Streamlit UI
├── db/init/               # Postgres schema
├── docker-compose.yml     # Full stack (7 containers)
└── README.md
```

Each service has its own `requirements.txt` and `Dockerfile`. In Docker, services talk to each other by name (`kafka:9092`, `postgres:5432`, `api:8000`). For local development, swap those for `localhost` and run services individually — see below.

## Local Development

If you want to iterate on a single service without rebuilding Docker images every time:

```bash
# Start just the infrastructure
docker compose up -d kafka zookeeper postgres

# Activate venv and install whichever service you're working on
source venv/bin/activate
pip install -r producer/requirements.txt

# Run from the service directory (relative paths depend on working directory)
cd producer
python producer.py
```

When running locally, services connect to `localhost:9092` / `localhost:5432`. The `.env.example` in each service folder shows the available config.

## Things I'd Improve

The language filter isn't perfect — Bluesky relies on self-declared tags from client apps, so Japanese or Portuguese posts occasionally slip through tagged as English. Adding `fasttext-langdetect` as a second-stage filter would catch these.

The model was trained on tweets, which are structurally similar to Bluesky posts but not identical. Fine-tuning on actual Bluesky data would probably help, but I'd need to label a dataset first.

Docker on macOS runs inside a Linux VM without GPU access, so inference in containers is CPU-only (~15-30 msg/sec). On bare metal with a GPU, throughput would be much higher. Running locally on Apple Silicon (MPS) gets ~75-100 msg/sec.

The API has no authentication — fine for local use, not for public deployment.

## Tech Stack

Kafka (Confluent 7.5.0), DistilBERT via HuggingFace Transformers + PyTorch, PostgreSQL 16, FastAPI + psycopg3, Streamlit + Plotly, Docker Compose. Data source is Bluesky's AT Protocol firehose via the atproto Python SDK.

## License

MIT
