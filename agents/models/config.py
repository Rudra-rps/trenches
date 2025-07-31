"""Configuration models for the Trenches agent system."""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    """Dynamic simulation configuration"""
    rounds: int = 3
    backend_url: str = "http://localhost:8080"
    backend_timeout: int = 5
    round_delay_range: List[int] = field(default_factory=lambda: [8, 15])
    agent_delay_range: List[float] = field(default_factory=lambda: [1.0, 3.0])
    context_tweets_limit: int = 5
    max_concurrent_agents: int = 10
    health_check_timeout: int = 3
    request_timeout: int = 5

    @classmethod
    def from_file(cls, path: Path) -> 'SimulationConfig':
        """Load configuration from file"""
        if path.exists():
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
                return cls(**data)
        return cls()

    def save_to_file(self, path: Path):
        """Save configuration to file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


@dataclass
class LLMConfig:
    """Dynamic LLM configuration for Groq API"""
    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"
    default_model: str = "llama-3.1-8b-instant"
    default_temperature: float = 0.7
    default_max_tokens: int = 100
    request_timeout: int = 30
    retry_attempts: int = 3
    available_models: List[str] = field(default_factory=lambda: [
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma-7b-it"
    ])
    default_headers: Dict[str, str] = field(default_factory=lambda: {
        "Content-Type": "application/json"
    })

    @classmethod
    def from_env_and_file(cls, config_path: Path = None) -> 'LLMConfig':
        """Load LLM config from environment and file"""
        config = cls()

        # Load from environment
        config.api_key = os.getenv('GROQ_API_KEY', '')
        config.base_url = os.getenv('LLM_BASE_URL', config.base_url)
        config.default_model = os.getenv('DEFAULT_LLM_MODEL', config.default_model)
        config.default_temperature = float(os.getenv('DEFAULT_TEMPERATURE', config.default_temperature))
        config.default_max_tokens = int(os.getenv('DEFAULT_MAX_TOKENS', config.default_max_tokens))

        # Override with file if exists
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
                for key, value in file_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

        return config

    def discover_models_from_agents(self, agents: Dict[str, Dict]) -> List[str]:
        """Dynamically discover available models from agent configurations"""
        models = set(self.available_models)
        models.add(self.default_model)

        # Extract models from all agents
        for agent in agents.values():
            agent_model = agent.get('llm', {}).get('model')
            if agent_model:
                models.add(agent_model)

        self.available_models = list(models)
        return self.available_models
