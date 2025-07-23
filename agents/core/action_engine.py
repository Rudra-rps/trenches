"""Action probability engine for dynamic agent behavior."""

import random
import yaml
from pathlib import Path
from typing import Dict, List, Any

from models.entities import SimulationContext


class ActionProbabilityEngine:
    """Dynamic action probability calculation based on agent traits"""

    def __init__(self, config_path: Path = None):
        self.base_probabilities = self._load_base_probabilities(config_path)
        self.personality_modifiers = self._load_personality_modifiers(config_path)
        self.context_modifiers = self._load_context_modifiers(config_path)

    def _load_base_probabilities(self, config_path: Path) -> Dict[str, float]:
        """Load base action probabilities from config"""
        default_probs = {
            "tweet": 0.6,
            "like": 0.2,
            "retweet": 0.15,
            "reply": 0.05
        }

        if config_path and (config_path / "action_probabilities.yaml").exists():
            with open(config_path / "action_probabilities.yaml", 'r') as f:
                return yaml.safe_load(f) or default_probs

        return default_probs

    def _load_personality_modifiers(self, config_path: Path) -> Dict[str, Dict[str, float]]:
        """Load personality-based probability modifiers"""
        default_modifiers = {
            "analytical": {"tweet": 1.3, "like": 0.5, "retweet": 0.3, "reply": 0.8},
            "sarcastic": {"tweet": 1.2, "like": 0.5, "retweet": 0.7, "reply": 1.5},
            "optimistic": {"tweet": 0.8, "like": 1.5, "retweet": 1.2, "reply": 0.8},
            "playful": {"tweet": 1.3, "like": 0.8, "retweet": 0.5, "reply": 1.2},
            "contemplative": {"tweet": 1.5, "like": 0.3, "retweet": 0.2, "reply": 0.4},
            "neutral": {"tweet": 1.0, "like": 1.0, "retweet": 1.0, "reply": 1.0}
        }

        if config_path and (config_path / "personality_modifiers.yaml").exists():
            with open(config_path / "personality_modifiers.yaml", 'r') as f:
                return yaml.safe_load(f) or default_modifiers

        return default_modifiers

    def _load_context_modifiers(self, config_path: Path) -> Dict[str, Dict[str, float]]:
        """Load context-based probability modifiers"""
        default_modifiers = {
            "high_activity": {"like": 1.3, "reply": 1.2, "tweet": 0.8},
            "low_activity": {"tweet": 1.4, "like": 0.8, "reply": 0.9},
            "positive_sentiment": {"like": 1.2, "retweet": 1.1, "reply": 1.1},
            "negative_sentiment": {"reply": 1.3, "tweet": 0.9, "like": 0.8},
            "trending_topics": {"tweet": 1.2, "retweet": 1.5, "reply": 1.1}
        }

        if config_path and (config_path / "context_modifiers.yaml").exists():
            with open(config_path / "context_modifiers.yaml", 'r') as f:
                return yaml.safe_load(f) or default_modifiers

        return default_modifiers

    def calculate_probabilities(self, agent: Dict, context: SimulationContext = None) -> Dict[str, float]:
        """Calculate dynamic action probabilities for an agent"""
        probs = self.base_probabilities.copy()

        # Apply personality modifiers
        personality = agent.get('personality', {})
        temperament = personality.get('temperament', 'neutral')

        if temperament in self.personality_modifiers:
            modifiers = self.personality_modifiers[temperament]
            for action, modifier in modifiers.items():
                if action in probs:
                    probs[action] *= modifier

        # Apply context modifiers if available
        if context:
            # Activity level modifiers
            activity_key = f"{context.activity_level}_activity"
            if activity_key in self.context_modifiers:
                modifiers = self.context_modifiers[activity_key]
                for action, modifier in modifiers.items():
                    if action in probs:
                        probs[action] *= modifier

            # Sentiment modifiers
            sentiment_key = f"{context.sentiment}_sentiment"
            if sentiment_key in self.context_modifiers:
                modifiers = self.context_modifiers[sentiment_key]
                for action, modifier in modifiers.items():
                    if action in probs:
                        probs[action] *= modifier

            # Trending topics modifiers
            if context.trending_topics and "trending_topics" in self.context_modifiers:
                modifiers = self.context_modifiers["trending_topics"]
                for action, modifier in modifiers.items():
                    if action in probs:
                        probs[action] *= modifier

        # Apply agent-specific activity modifiers
        activity_config = agent.get('activity', {})
        if 'action_preferences' in activity_config:
            preferences = activity_config['action_preferences']
            for action, preference in preferences.items():
                if action in probs:
                    probs[action] *= preference

        # Normalize probabilities
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}

        return probs

    def select_action(self, agent: Dict, context: SimulationContext = None) -> str:
        """Select an action based on calculated probabilities"""
        probabilities = self.calculate_probabilities(agent, context)

        actions = list(probabilities.keys())
        weights = list(probabilities.values())

        return random.choices(actions, weights=weights)[0]

    def should_agent_be_active(self, agent: Dict, context: SimulationContext = None) -> bool:
        """Determine if an agent should be active based on schedule and context"""
        import time

        activity = agent.get('activity', {})
        current_hour = time.localtime().tm_hour

        # Check time-based activity
        active_hours = activity.get('active_hours', list(range(24)))
        if current_hour not in active_hours:
            return False

        # Check activity probability
        base_prob = activity.get('activity_probability', 0.7)

        # Modify probability based on context
        if context:
            if context.activity_level == 'high':
                base_prob *= activity.get('high_activity_modifier', 1.2)
            elif context.activity_level == 'low':
                base_prob *= activity.get('low_activity_modifier', 0.8)

        return random.random() < base_prob
