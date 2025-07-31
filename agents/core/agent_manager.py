"""Agent manager for loading and validating agent configurations."""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from models.entities import Tweet, Profile


class AgentManager:
    """Manager for agent configurations and operations"""

    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path("config")
        self.config_path.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)
        self.agents_cache = {}

    def load_all_agents(self, agent_dir: Path = None) -> Dict[str, Dict]:
        """Dynamically load all agent configurations"""
        if agent_dir is None:
            agent_dir = Path("agent_spec")

        if not agent_dir.exists():
            self.logger.warning(f"Agent directory {agent_dir} not found")
            return {}

        agents = {}
        for config_file in agent_dir.glob("*.yaml"):
            try:
                agent = self._load_and_validate_agent(config_file)
                agents[agent['id']] = agent
                self.logger.debug(f"Loaded agent: {agent['id']}")
            except Exception as e:
                self.logger.error(f"Failed to load {config_file}: {e}")

        self.logger.info(f"Loaded {len(agents)} agents from {agent_dir}")
        self.agents_cache = agents
        return agents

    def _load_and_validate_agent(self, path: Path) -> Dict:
        """Load and validate agent with dynamic schema"""
        with open(path, 'r', encoding='utf-8') as f:
            agent = yaml.safe_load(f)

        if not agent:
            raise ValueError(f"Empty configuration file: {path}")

        # Dynamic validation based on schema file
        schema_path = self.config_path / "agent_schema.yaml"
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema = yaml.safe_load(f)
                self._validate_against_schema(agent, schema, path)
        else:
            # Minimal validation
            if 'id' not in agent:
                raise ValueError(f"Agent missing required 'id' field in {path}")

        # Apply dynamic defaults
        self._apply_dynamic_defaults(agent)
        return agent

    def _validate_against_schema(self, agent: Dict, schema: Dict, path: Path):
        """Validate agent against dynamic schema"""
        required_fields = schema.get('required', ['id'])
        for field in required_fields:
            if field not in agent:
                raise ValueError(f"Agent missing required field '{field}' in {path}")

        # Validate field types if specified
        field_types = schema.get('field_types', {})
        for field, expected_type in field_types.items():
            if field in agent:
                actual_type = type(agent[field]).__name__
                if actual_type != expected_type:
                    raise ValueError(f"Field '{field}' should be {expected_type}, got {actual_type} in {path}")

    def _apply_dynamic_defaults(self, agent: Dict):
        """Apply dynamic defaults to agent configuration"""
        defaults_path = self.config_path / "agent_defaults.yaml"
        if defaults_path.exists():
            with open(defaults_path, 'r') as f:
                defaults = yaml.safe_load(f) or {}
                self._deep_merge_defaults(agent, defaults)

    def _deep_merge_defaults(self, agent: Dict, defaults: Dict):
        """Deep merge defaults into agent config"""
        for key, value in defaults.items():
            if key not in agent:
                agent[key] = value
            elif isinstance(value, dict) and isinstance(agent[key], dict):
                self._deep_merge_defaults(agent[key], value)

    def get_agent_profile(self, agent: Dict) -> Profile:
        """Generate a profile for an agent"""
        return Profile(
            username=agent.get('id', 'unknown'),
            avatar=agent.get('avatar', ''),
            metadata={
                'personality': agent.get('personality', {}),
                'llm_model': agent.get('llm', {}).get('model', 'unknown'),
                'temperament': agent.get('personality', {}).get('temperament', 'neutral')
            }
        )

    def select_active_agents(self, agents: Dict, context: Any, max_agents: int = 10) -> List[Dict]:
        """Select which agents should be active based on dynamic criteria"""
        import random
        import time

        active = []
        current_hour = time.localtime().tm_hour

        for agent in agents.values():
            activity = agent.get('activity', {})

            # Check time-based activity
            active_hours = activity.get('active_hours', list(range(24)))
            if current_hour not in active_hours:
                continue

            # Check activity probability
            base_prob = activity.get('activity_probability', 0.7)

            # Modify probability based on context
            if hasattr(context, 'activity_level'):
                if context.activity_level == 'high':
                    base_prob *= activity.get('high_activity_modifier', 1.2)
                elif context.activity_level == 'low':
                    base_prob *= activity.get('low_activity_modifier', 0.8)

            if random.random() < base_prob:
                active.append(agent)

        # Limit concurrent agents
        if len(active) > max_agents:
            active = random.sample(active, max_agents)

        return active

    def get_agent_constraints(self, agent: Dict) -> Dict[str, Any]:
        """Get constraints for an agent"""
        return agent.get('constraints', {
            'max_tweet_length': 280,
            'min_delay_between_actions': 1.0,
            'max_actions_per_round': 3
        })

    def get_agent_by_id(self, agent_id: str) -> Optional[Dict]:
        """Get agent by ID from cache"""
        return self.agents_cache.get(agent_id)

    def get_agent_stats_summary(self, agents: Dict) -> Dict[str, Any]:
        """Get summary statistics about loaded agents"""
        if not agents:
            return {}

        temperaments = {}
        models = {}

        for agent in agents.values():
            # Count temperaments
            temp = agent.get('personality', {}).get('temperament', 'unknown')
            temperaments[temp] = temperaments.get(temp, 0) + 1

            # Count models
            model = agent.get('llm', {}).get('model', 'unknown')
            models[model] = models.get(model, 0) + 1

        return {
            'total_agents': len(agents),
            'temperament_distribution': temperaments,
            'model_distribution': models,
            'agent_ids': list(agents.keys())
        }
