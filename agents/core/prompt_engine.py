"""Dynamic prompt generation system for agent content creation."""

import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any

from models.entities import SimulationContext, Tweet


class DynamicPromptEngine:
    """Dynamic prompt generation system"""

    def __init__(self, config_path: Path = None):
        self.templates = self._load_prompt_templates(config_path)
        self.personality_traits = self._load_personality_traits(config_path)
        self.context_analyzers = self._load_context_analyzers(config_path)

    def _load_prompt_templates(self, config_path: Path) -> Dict[str, str]:
        """Load dynamic prompt templates"""
        default_templates = {
            "base": """You are {agent_id}, an AI agent with the following dynamic characteristics:

{personality_description}

{context_section}

{instruction_section}

{constraints_section}

Your response:""",

            "personality_trait": "- {trait_name}: {trait_value} ({trait_description})",
            "context_trending": "Current trending topics: {topics}",
            "context_activity": "Community activity level: {level}",
            "context_sentiment": "Community sentiment: {sentiment}",
            "constraint_length": "Keep response under {max_chars} characters",
            "context_time": "Current time context: {time_info}",
            "context_recent_tweets": "Recent community tweets:\n{tweets}"
        }

        if config_path and (config_path / "prompt_templates.yaml").exists():
            with open(config_path / "prompt_templates.yaml", 'r') as f:
                loaded = yaml.safe_load(f) or {}
                default_templates.update(loaded)

        return default_templates

    def _load_personality_traits(self, config_path: Path) -> Dict[str, Dict]:
        """Load personality trait descriptions and behaviors"""
        default_traits = {
            "temperament": {
                "analytical": "You approach topics with logical reasoning and data-driven insights",
                "sarcastic": "You often use wit and irony to make points",
                "optimistic": "You tend to see the positive side of situations",
                "playful": "You enjoy humor and lighthearted interactions",
                "contemplative": "You prefer deep, thoughtful discussions",
                "neutral": "You maintain a balanced perspective on topics"
            },
            "tone": {
                "formal": "You communicate in a professional, structured manner",
                "casual": "You use relaxed, conversational language",
                "energetic": "Your communication is vibrant and enthusiastic",
                "calm": "You maintain a peaceful, measured tone",
                "neutral": "You use a balanced, moderate tone",
                "philosophical": "You explore deeper meanings and implications",
                "technical": "You focus on precise, technical communication",
                "witty": "You use clever wordplay and humor",
                "cheerful": "You maintain an upbeat, positive tone"
            },
            "emotionality": {
                "high": "You express emotions freely and passionately",
                "medium": "You show appropriate emotional responses",
                "low": "You maintain emotional restraint and composure"
            },
            "decision_bias": {
                "optimistic": "You tend to see the best possible outcomes",
                "pessimistic": "You consider potential risks and downsides",
                "balanced": "You weigh both positive and negative aspects",
                "thoughtful": "You carefully consider all angles before responding",
                "humorous": "You find humor in most situations",
                "logical": "You rely on facts and reasoning",
                "positive": "You focus on positive aspects and solutions",
                "trendy": "You stay current with latest trends and topics"
            }
        }

        if config_path and (config_path / "personality_traits.yaml").exists():
            with open(config_path / "personality_traits.yaml", 'r') as f:
                loaded = yaml.safe_load(f) or {}
                default_traits.update(loaded)

        return default_traits

    def _load_context_analyzers(self, config_path: Path) -> Dict[str, Any]:
        """Load context analysis configuration"""
        default_analyzers = {
            "sentiment_keywords": {
                "positive": ["great", "awesome", "love", "amazing", "wonderful", "excellent", "fantastic", "brilliant"],
                "negative": ["terrible", "awful", "hate", "horrible", "disappointing", "failed", "worst", "bad"]
            },
            "activity_thresholds": {
                "high": 10,
                "medium": 5,
                "low": 2
            },
            "stop_words": {
                "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "a", "an", "is", "are", "was", "were"
            },
            "min_trend_frequency": 2,
            "max_trending_topics": 8,
            "max_recent_tweets_context": 3
        }

        if config_path and (config_path / "context_analyzers.yaml").exists():
            with open(config_path / "context_analyzers.yaml", 'r') as f:
                loaded = yaml.safe_load(f) or {}
                default_analyzers.update(loaded)

        return default_analyzers

    def build_dynamic_prompt(self, agent: Dict, action_type: str, context: SimulationContext = None) -> str:
        """Build completely dynamic prompt based on agent and context"""
        agent_id = agent.get('id', 'Agent')
        personality = agent.get('personality', {})

        # Build personality description
        personality_parts = []
        for trait_category, trait_value in personality.items():
            if trait_category in self.personality_traits:
                trait_info = self.personality_traits[trait_category].get(trait_value)
                if trait_info:
                    personality_parts.append(
                        self.templates["personality_trait"].format(
                            trait_name=trait_category.replace('_', ' ').title(),
                            trait_value=trait_value,
                            trait_description=trait_info
                        )
                    )

        personality_description = "\n".join(personality_parts) if personality_parts else "Standard AI agent personality"

        # Build context section
        context_parts = []
        if context:
            if context.trending_topics:
                context_parts.append(
                    self.templates["context_trending"].format(
                        topics=", ".join(context.trending_topics[:5])
                    )
                )
            if context.activity_level:
                context_parts.append(
                    self.templates["context_activity"].format(
                        level=context.activity_level
                    )
                )
            if context.sentiment:
                context_parts.append(
                    self.templates["context_sentiment"].format(
                        sentiment=context.sentiment
                    )
                )
            if context.time_context:
                context_parts.append(
                    self.templates["context_time"].format(
                        time_info=context.time_context
                    )
                )
            if context.recent_tweets:
                max_tweets = self.context_analyzers.get("max_recent_tweets_context", 3)
                tweet_summaries = []
                for tweet in context.recent_tweets[:max_tweets]:
                    tweet_summaries.append(f"@{tweet.agent_id}: {tweet.content[:100]}...")

                context_parts.append(
                    self.templates["context_recent_tweets"].format(
                        tweets="\n".join(tweet_summaries)
                    )
                )

        context_section = "\n".join(context_parts) if context_parts else ""

        # Build instruction section based on action type
        instruction_map = {
            "tweet": "Generate an original tweet that reflects your personality and current context. Be engaging and authentic.",
            "reply": "Generate a thoughtful reply to the conversation that adds value and reflects your personality.",
            "retweet": "Consider whether this content aligns with your personality and if you would share it with your followers.",
            "like": "Consider liking content that resonates with your personality and interests."
        }
        instruction_section = instruction_map.get(action_type, "Generate appropriate content")

        # Build constraints section
        constraints = []
        if action_type in ["tweet", "reply"]:
            max_chars = agent.get('constraints', {}).get('max_tweet_length', 280)
            constraints.append(
                self.templates["constraint_length"].format(max_chars=max_chars)
            )
        constraints_section = "\n".join(constraints)

        # Assemble final prompt
        return self.templates["base"].format(
            agent_id=agent_id,
            personality_description=personality_description,
            context_section=context_section,
            instruction_section=instruction_section,
            constraints_section=constraints_section
        )

    def analyze_context_from_tweets(self, tweets: List[Tweet]) -> SimulationContext:
        """Analyze context from recent tweets"""
        context = SimulationContext()

        if not tweets:
            return context

        context.recent_tweets = tweets
        context.trending_topics = self._extract_trending_topics(tweets)
        context.activity_level = self._calculate_activity_level(tweets)
        context.sentiment = self._analyze_sentiment(tweets)
        context.time_context = self._get_time_context()

        return context

    def _extract_trending_topics(self, tweets: List[Tweet]) -> List[str]:
        """Extract trending topics from recent tweets"""
        word_freq = {}
        stop_words = self.context_analyzers.get('stop_words', set())

        for tweet in tweets:
            content = tweet.content.lower()
            words = [w.strip('.,!?#@') for w in content.split()
                    if len(w) > 3 and w not in stop_words]

            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Return top trending words
        min_frequency = self.context_analyzers.get('min_trend_frequency', 2)
        max_trends = self.context_analyzers.get('max_trending_topics', 8)

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_trends] if freq >= min_frequency]

    def _calculate_activity_level(self, tweets: List[Tweet]) -> str:
        """Calculate activity level based on recent tweets"""
        thresholds = self.context_analyzers.get('activity_thresholds', {
            'high': 10, 'medium': 5, 'low': 2
        })

        tweet_count = len(tweets)

        if tweet_count >= thresholds['high']:
            return 'high'
        elif tweet_count >= thresholds['medium']:
            return 'medium'
        elif tweet_count >= thresholds['low']:
            return 'low'
        else:
            return 'very_low'

    def _analyze_sentiment(self, tweets: List[Tweet]) -> str:
        """Analyze overall sentiment of recent tweets"""
        sentiment_keywords = self.context_analyzers.get('sentiment_keywords', {
            'positive': ['great', 'awesome', 'love', 'amazing'],
            'negative': ['terrible', 'awful', 'hate', 'horrible']
        })

        positive_count = 0
        negative_count = 0

        for tweet in tweets:
            content = tweet.content.lower()

            for word in sentiment_keywords['positive']:
                if word in content:
                    positive_count += 1

            for word in sentiment_keywords['negative']:
                if word in content:
                    negative_count += 1

        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

    def _get_time_context(self) -> str:
        """Get time-based context"""
        current_hour = time.localtime().tm_hour

        time_periods = {
            'early_morning': range(5, 9),
            'morning': range(9, 12),
            'afternoon': range(12, 17),
            'evening': range(17, 21),
            'night': range(21, 24),
            'late_night': list(range(0, 5))
        }

        for period, hours in time_periods.items():
            if current_hour in hours:
                return period

        return 'unknown'
