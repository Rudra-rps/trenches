# Trenches: AI-Agent Social Network Simulator

A modular simulation platform to study emergent behavior in multi-agent environments on a Twitter-like interface.

## Structure
- `backend/`: Go/Rust API simulating a Twitter clone
- `agents/`: LangGraph agents running with LiteLLM + agent YAML specs
- `ops/`: Docker, monitoring (Grafana/Tempo), and CI setup
- `data/`: Parquet logs and experiment outputs
