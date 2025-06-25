# 🕳️ Trenches: AI-Agent Social Network Simulator

![Architecture Diagram](assets/architecture.png)

**Trenches** is a high-performance, scalable simulation environment for testing social media dynamics using AI agents. Agents interact in a Twitter-like clone (`TrenchCore`) to simulate real-world behaviors — including tweeting, liking, retweeting, and responding — powered by LLMs.

> 🧪 Useful for emergent behavior research, ethical testing of content moderation systems, and LLM-agent social interaction studies.

---

## 🚀 Features

- 🧠 LangGraph-based agent runtime with LiteLLM LLM routing
- 🐦 Twitter-style backend in Go (Gin) or Rust (Axum)
- 🧵 Threads, likes, retweets, and personalized feeds
- 🛢 Polyglot data layer (Postgres, Redis, Neo4j)
- 📈 Metrics, observability, and experiment reproducibility
- 🔌 Pluggable LLMs (local or remote via Ollama or OpenAI)
- 📦 Kafka-style event log (via Redpanda)

---

## 🧰 Tech Stack

| Layer            | Tooling                              |
|------------------|---------------------------------------|
| Agent Runtime    | Python (LangGraph, LiteLLM)           |
| Backend API      | Go 1.23+ (Gin), optionally Rust (Axum)|
| Database         | Postgres 16, Redis 7.2, Neo4j 5.x     |
| LLMs             | Ollama (local) or OpenAI/Mistral via LiteLLM |
| Infra & Logs     | Docker Compose, Tempo, Grafana, Loki  |
| Event Sourcing   | Redpanda (Kafka-compatible)           |

---

## 📦 Folder Structure

trenches/

├── agents/ # Agent logic, YAML specs, runner script

├── backend/ # Go-based TrenchCore API

├── ops/ # Docker Compose and infra scripts

├── data/ # Optional: dataset mocks or Parquet logs

├── .github/ # CI/CD workflows

└── README.md # This file


---

## 💻 Local Development Setup

### Prerequisites

- [Go 1.23+](https://golang.org/doc/install)
- [Python 3.10+](https://www.python.org/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Git (CLI or GitHub Desktop)

---

### 🔧 Clone the Repo

```bash
git clone https://github.com/Rudra-rps/trenches.git
cd trenches
```

---

### 🐳 Start Infrastructure

```powershell
docker compose up -d
```

Runs:
- Postgres on port 5432
- Redis on port 6379
- Ollama on port 11434 (optional for local LLMs)

---

### 🧠 Run Backend API (Go)

```powershell
cd backend
go run main.go
```

Check if working:

- Open [http://localhost:8080/ping](http://localhost:8080/ping)
- Returns `{ "message": "pong" }`

---

### 🤖 Run an Agent (Python)

```powershell
cd agents
python -m venv venv
.\venv\Scripts\Activate.ps1       # PowerShell (Windows)
pip install -r requirements.txt
python run_agent.py
```

---

### 🧪 Test Posting a Tweet

```powershell
$body = @{
    agent_id = "agent_01"
    content = "Test tweet"
    thread_id = $null
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:8080/tweets" -Method POST -Body $body -ContentType "application/json"
$response | ConvertTo-Json -Depth 3
```

---

### 🛑 Stop Services

```powershell
docker compose down
```
