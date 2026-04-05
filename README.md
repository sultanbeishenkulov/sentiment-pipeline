# Real-Time Social Media Sentiment Pipeline

A production-style streaming pipeline that ingests Reddit posts in real time, classifies sentiment using a fine-tuned DistilBERT model, stores results in PostgreSQL, and visualizes trends through a live Streamlit dashboard.

## Architecture

```
┌──────────────┐     ┌─────────────┐     ┌──────────────┐     ┌────────────┐
│  Reddit API  │────▶│    Kafka     │────▶│   Consumer   │────▶│ PostgreSQL │
│  (Producer)  │     │   Broker    │     │  + NLP Model │     │            │
└──────────────┘     └─────────────┘     └──────────────┘     └─────┬──────┘
                                                                     │
                                                              ┌──────▼──────┐
                                                              │   FastAPI   │
                                                              │   Server    │
                                                              └──────┬──────┘
                                                                     │
                                                              ┌──────▼──────┐
                                                              │  Streamlit  │
                                                              │  Dashboard  │
                                                              └─────────────┘
```

## Tech Stack

- **Data Ingestion**: Reddit API (PRAW) → Kafka producer
- **Streaming**: Apache Kafka (Zookeeper + Broker)
- **NLP Model**: Fine-tuned DistilBERT (HuggingFace Transformers) — 3-class sentiment (positive / negative / neutral)
- **Storage**: PostgreSQL with time-series indexing
- **API**: FastAPI with aggregated sentiment endpoints
- **Dashboard**: Streamlit with real-time polling and interactive charts
- **Infrastructure**: Docker Compose (all services containerized)

## Project Structure

```
sentiment-pipeline/
├── producer/           # Reddit API ingestion → Kafka
├── consumer/           # Kafka consumer → NLP inference → PostgreSQL
├── model/              # DistilBERT fine-tuning, evaluation, inference
├── api/                # FastAPI endpoints for sentiment trends
├── dashboard/          # Streamlit real-time visualization
├── docker/             # Dockerfiles and docker-compose.yml
├── data/               # Training data and sample datasets
├── .env.example        # Environment variable template
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Getting Started

> Setup instructions will be added as services are built.

## License

MIT
