import asyncio
import aiohttp
import time
import random
import yaml
import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI
import backoff

load_dotenv()

class ActionType(Enum):
    TWEET = "tweet"
    LIKE = "like"
    RETWEET = "retweet"
    REPLY = "reply"
    FOLLOW = "follow"

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
    """Dynamic LLM configuration"""
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "moonshotai/kimi-k2:free"
    default_temperature: float = 0.7
    default_max_tokens: int = 100
    request_timeout: int = 30
    retry_attempts: int = 3
    available_models: List[str] = field(default_factory=list)
    default_headers: Dict[str, str] = field(default_factory=lambda: {
        "HTTP-Referer": "https://trenches-social.com",
        "X-Title": "Trenches Social Sim"
    })

    @classmethod
    def from_env_and_file(cls, config_path: Path = None) -> 'LLMConfig':
        """Load LLM config from environment and file"""
        config = cls()

        # Load from environment
        config.api_key = os.getenv('OPENROUTER_API_KEY', '')
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
                models = set()

                # Add default model
                models.add(self.default_model)

                # Extract models from all agents
                for agent in agents.values():
                    agent_model = agent.get('llm', {}).get('model')
                    if agent_model:
                        models.add(agent_model)

                self.available_models = list(models)
                return self.available_models

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
            "analytical": {"tweet": 1.3, "like": 0.5, "retweet": 0.3, "reply": 0.5},
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
            "trending_topic": {"tweet": 1.2, "retweet": 1.5},
            "high_activity": {"like": 1.3, "reply": 1.2},
            "low_activity": {"tweet": 1.4, "like": 0.8},
            "positive_sentiment": {"like": 1.2, "retweet": 1.1},
            "negative_sentiment": {"reply": 1.3, "tweet": 0.9}
        }

        if config_path and (config_path / "context_modifiers.yaml").exists():
            with open(config_path / "context_modifiers.yaml", 'r') as f:
                return yaml.safe_load(f) or default_modifiers

        return default_modifiers

    def calculate_probabilities(self, agent: Dict, context: Dict = None) -> Dict[str, float]:
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
            for context_type, active in context.items():
                if active and context_type in self.context_modifiers:
                    modifiers = self.context_modifiers[context_type]
                    for action, modifier in modifiers.items():
                        if action in probs:
                            probs[action] *= modifier

        # Normalize probabilities
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}

        return probs

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
            "context_time": "Current time context: {time_info}"
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
                "neutral": "You use a balanced, moderate tone"
            },
            "emotionality": {
                "high": "You express emotions freely and passionately",
                "medium": "You show appropriate emotional responses",
                "low": "You maintain emotional restraint and composure"
            },
            "decision_bias": {
                "optimistic": "You tend to see the best possible outcomes",
                "pessimistic": "You consider potential risks and downsides",
                "balanced": "You weigh both positive and negative aspects"
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
                "positive": ["great", "awesome", "love", "amazing", "wonderful", "excellent"],
                "negative": ["terrible", "awful", "hate", "horrible", "disappointing", "failed"]
            },
            "activity_thresholds": {
                "high": 10,
                "medium": 5,
                "low": 2
            }
        }

        if config_path and (config_path / "context_analyzers.yaml").exists():
            with open(config_path / "context_analyzers.yaml", 'r') as f:
                loaded = yaml.safe_load(f) or {}
                default_analyzers.update(loaded)

        return default_analyzers

    def build_dynamic_prompt(self, agent: Dict, action_type: str, context: Dict = None) -> str:
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
            if context.get('trending_topics'):
                context_parts.append(
                    self.templates["context_trending"].format(
                        topics=", ".join(context['trending_topics'][:5])
                    )
                )
            if context.get('activity_level'):
                context_parts.append(
                    self.templates["context_activity"].format(
                        level=context['activity_level']
                    )
                )
            if context.get('sentiment'):
                context_parts.append(
                    self.templates["context_sentiment"].format(
                        sentiment=context['sentiment']
                    )
                )
            if context.get('time_context'):
                context_parts.append(
                    self.templates["context_time"].format(
                        time_info=context['time_context']
                    )
                )

        context_section = "\n".join(context_parts) if context_parts else ""

        # Build instruction section based on action type
        instruction_map = {
            "tweet": "Generate an original tweet that reflects your personality and current context",
            "reply": "Generate a thoughtful reply to the conversation",
            "retweet": "Decide whether to retweet and optionally add commentary",
            "like": "Consider liking content that resonates with your personality"
        }
        instruction_section = instruction_map.get(action_type, "Generate appropriate content")

        # Build constraints section
        constraints = []
        if action_type == "tweet":
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

class TrenchesAgent:
    def __init__(self, config_path: Path = None):
        # Set up configuration path
        if config_path is None:
            config_path = Path(os.getenv('CONFIG_DIR', 'config'))
        self.config_path = config_path
        self.config_path.mkdir(exist_ok=True)

        # Load all dynamic configurations
        self.sim_config = SimulationConfig.from_file(config_path / "simulation.yaml")
        self.llm_config = LLMConfig.from_env_and_file(config_path / "llm.yaml")
        self.action_engine = ActionProbabilityEngine(config_path)
        self.prompt_engine = DynamicPromptEngine(config_path)

        # Runtime state
        self.session = None
        self.agents_cache = {}
        self.context_cache = {}

        # Setup logging with dynamic level
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def run_dynamic_simulation(self):
        """Run completely dynamic simulation"""
        agents = self.load_all_agents()
        if not agents:
            raise ValueError("No agents loaded for simulation")

        # Discover models dynamically from agent configurations
        self.llm_config.discover_models_from_agents(agents)
        self.logger.info(f"🤖 Discovered models: {', '.join(self.llm_config.available_models)}")

        self.logger.info(f"🚀 Starting dynamic simulation with {len(agents)} agents")

        for round_num in range(self.sim_config.rounds):
            self.logger.info(f"\n🔄 Round {round_num + 1}/{self.sim_config.rounds}")

            # Analyze current context dynamically
            context = await self.analyze_dynamic_context()
            self.logger.info(f"📊 Context: {len(context)} factors analyzed")

            # Select active agents dynamically
            active_agents = self._select_dynamic_agents(agents, context)
            self.logger.info(f"👥 {len(active_agents)} agents selected for this round")

            # Run agents concurrently
            tasks = [
                self.simulate_agent_dynamically(agent, context)
                for agent in active_agents
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

            # Dynamic delay between rounds
            if round_num < self.sim_config.rounds - 1:
                delay = random.randint(*self.sim_config.round_delay_range)
                self.logger.info(f"⏳ Waiting {delay}s before next round...")
                await asyncio.sleep(delay)

        self.logger.info("✅ Dynamic simulation complete!")

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=self.sim_config.max_concurrent_agents,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        timeout = aiohttp.ClientTimeout(total=self.sim_config.backend_timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def load_all_agents(self, agent_dir: Path = None) -> Dict[str, Dict]:
        """Dynamically load all agent configurations"""
        if agent_dir is None:
            agent_dir = Path(os.getenv('AGENT_SPEC_DIR', 'agent_spec'))

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

    async def check_backend_health_async(self) -> bool:
        """Check if Go backend is running asynchronously"""
        try:
            health_endpoint = f"{self.sim_config.backend_url}/ping"
            timeout = aiohttp.ClientTimeout(total=self.sim_config.health_check_timeout)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_endpoint) as response:
                    if response.status == 200:
                        data = await response.json()
                        return (data == "pong" or
                               (isinstance(data, dict) and data.get("message") == "pong"))
            return False
        except Exception as e:
            self.logger.error(f"Backend health check failed: {e}")
            return False

    async def get_recent_tweets_async(self, limit: int = None) -> List[Dict]:
        """Fetch recent tweets for context asynchronously"""
        if limit is None:
            limit = self.sim_config.context_tweets_limit

        try:
            tweets_endpoint = f"{self.sim_config.backend_url}/tweets"
            async with self.session.get(tweets_endpoint) as response:
                response.raise_for_status()
                tweets = await response.json()
                return tweets[:limit] if tweets else []
        except Exception as e:
            self.logger.error(f"Failed to fetch tweets: {e}")
            return []

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def post_tweet_async(self, agent_id: str, content: str, thread_id: Optional[int] = None) -> bool:
        """Post a tweet asynchronously with retry logic"""
        payload = {
            "agent_id": agent_id,
            "content": content,
            "timestamp": time.time()
        }
        if thread_id:
            payload["thread_id"] = thread_id

        try:
            tweets_endpoint = f"{self.sim_config.backend_url}/tweets"
            async with self.session.post(
                tweets_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                self.logger.info(f"✅ Tweet posted by {agent_id}: {content[:50]}...")
                return True
        except Exception as e:
            self.logger.error(f"❌ Failed to post tweet for {agent_id}: {e}")
            return False

    async def analyze_dynamic_context(self) -> Dict[str, Any]:
        """Analyze current context dynamically"""
        context = {}

        # Get recent tweets for analysis
        recent_tweets = await self.get_recent_tweets_async(
            limit=self.sim_config.context_tweets_limit * 2
        )

        if recent_tweets:
            # Analyze trending topics
            context['trending_topics'] = self._extract_trending_topics(recent_tweets)

            # Analyze activity level
            context['activity_level'] = self._calculate_activity_level(recent_tweets)

            # Analyze sentiment
            context['sentiment'] = self._analyze_sentiment(recent_tweets)

        # Time-based context
        context['time_context'] = self._get_time_context()

        return context

    def _extract_trending_topics(self, tweets: List[Dict]) -> List[str]:
        """Extract trending topics from recent tweets"""
        word_freq = {}
        stop_words_config = self.prompt_engine.context_analyzers.get('stop_words', {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        })

        for tweet in tweets:
            content = tweet.get('content', '').lower()
            words = [w.strip('.,!?#@') for w in content.split()
                    if len(w) > 3 and w not in stop_words_config]

            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Return top trending words
        min_frequency = self.prompt_engine.context_analyzers.get('min_trend_frequency', 1)
        max_trends = self.prompt_engine.context_analyzers.get('max_trending_topics', 10)

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_trends] if freq > min_frequency]

    def _calculate_activity_level(self, tweets: List[Dict]) -> str:
        """Calculate activity level based on recent tweets"""
        thresholds = self.prompt_engine.context_analyzers.get('activity_thresholds', {
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

    def _analyze_sentiment(self, tweets: List[Dict]) -> str:
        """Analyze overall sentiment of recent tweets"""
        sentiment_keywords = self.prompt_engine.context_analyzers.get('sentiment_keywords', {
            'positive': ['great', 'awesome', 'love', 'amazing'],
            'negative': ['terrible', 'awful', 'hate', 'horrible']
        })

        positive_count = 0
        negative_count = 0

        for tweet in tweets:
            content = tweet.get('content', '').lower()

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

    def _generate_with_openrouter(self, agent: Dict, prompt: str) -> str:
        """Generate using OpenRouter API via OpenAI SDK"""
        # Get configuration
        agent_llm = agent.get('llm', {})
        model = agent_llm.get('model', self.llm_config.default_model)
        temperature = agent_llm.get('temperature', self.llm_config.default_temperature)
        max_tokens = agent_llm.get('max_tokens', self.llm_config.default_max_tokens)

        # Get API key
        api_key = self.llm_config.api_key or os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OpenRouter API key not found")

        api_key = api_key.strip()

        try:
            # Initialize OpenAI client
            client = OpenAI(
                base_url=self.llm_config.base_url,
                api_key=api_key,
                default_headers=self.llm_config.default_headers
            )

            # Make the completion request
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Extract and format content
            content = completion.choices[0].message.content.strip()
            max_length = agent.get('constraints', {}).get('max_tweet_length', 280)
            return content[:max_length] if len(content) > max_length else content

        except Exception as e:
            error_message = str(e)
            self.logger.error(f"OpenRouter API error: {error_message}")

            if "401" in error_message:
                raise ValueError(f"OpenRouter authentication failed: {error_message}")
            raise

    async def generate_content_async(self, agent: Dict, action_type: str, context: Dict = None) -> str:
        """Generate content asynchronously using dynamic prompts"""
        agent_id = agent.get('id', 'unknown')

        try:
            prompt = self.prompt_engine.build_dynamic_prompt(agent, action_type, context)

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None,
                self._generate_with_openrouter,
                agent,
                prompt
            )
            return content
        except Exception as e:
            self.logger.error(f"❌ [{agent_id}] Failed to generate content: {e}")
            raise Exception(f"Content generation failed for {agent_id}: {e}")

    def _select_dynamic_agents(self, agents: Dict, context: Dict) -> List[Dict]:
        """Select which agents are active based on dynamic criteria"""
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
            if context.get('activity_level') == 'high':
                base_prob *= activity.get('high_activity_modifier', 1.2)
            elif context.get('activity_level') == 'low':
                base_prob *= activity.get('low_activity_modifier', 0.8)

            if random.random() < base_prob:
                active.append(agent)

        # Limit concurrent agents
        max_agents = activity.get('max_concurrent_agents', self.sim_config.max_concurrent_agents)
        if len(active) > max_agents:
            active = random.sample(active, max_agents)

        return active

    async def _execute_dynamic_action(self, agent: Dict, action: str, context: Dict):
        """Execute a dynamic action for an agent"""
        agent_id = agent.get('id')

        if action == 'tweet':
            try:
                content = await self.generate_content_async(agent, action, context)
                success = await self.post_tweet_async(agent_id, content)
                if success:
                    self.logger.info(f"🐦 [{agent_id}] tweeted")
            except Exception as e:
                self.logger.error(f"❌ [{agent_id}] Failed to tweet: {e}")
        else:
            # For MVP, just log other actions
            self.logger.info(f"📱 [{agent_id}] {action}d something...")

    async def simulate_agent_dynamically(self, agent: Dict, context: Dict):
        """Simulate agent with completely dynamic behavior"""
        agent_id = agent.get('id')

        # Calculate dynamic action probabilities
        probabilities = self.action_engine.calculate_probabilities(agent, context)

        # Determine number of actions dynamically
        activity = agent.get('activity', {})
        action_range = activity.get('actions_per_awake', [1, 2])

        if len(action_range) != 2 or action_range[0] > action_range[1]:
            action_range = [1, 2]

        num_actions = random.randint(action_range[0], action_range[1])

        for _ in range(num_actions):
            # Choose action based on dynamic probabilities
            action = random.choices(
                list(probabilities.keys()),
                weights=list(probabilities.values())
            )[0]

            await self._execute_dynamic_action(agent, action, context)

            # Dynamic delay between actions
            if num_actions > 1:
                delay_range = activity.get('action_delay_range', self.sim_config.agent_delay_range)
                delay = random.uniform(delay_range[0], delay_range[1])
                await asyncio.sleep(delay)

    async def run_dynamic_simulation(self):
        """Run completely dynamic simulation"""
        agents = self.load_all_agents()
        if not agents:
            raise ValueError("No agents loaded for simulation")

        self.logger.info(f"🚀 Starting dynamic simulation with {len(agents)} agents")

        for round_num in range(self.sim_config.rounds):
            self.logger.info(f"\n🔄 Round {round_num + 1}/{self.sim_config.rounds}")

            # Analyze current context dynamically
            context = await self.analyze_dynamic_context()
            self.logger.info(f"📊 Context: {len(context)} factors analyzed")

            # Select active agents dynamically
            active_agents = self._select_dynamic_agents(agents, context)
            self.logger.info(f"👥 {len(active_agents)} agents selected for this round")

            # Run agents concurrently
            tasks = [
                self.simulate_agent_dynamically(agent, context)
                for agent in active_agents
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

            # Dynamic delay between rounds
            if round_num < self.sim_config.rounds - 1:
                delay = random.randint(*self.sim_config.round_delay_range)
                self.logger.info(f"⏳ Waiting {delay}s before next round...")
                await asyncio.sleep(delay)

        self.logger.info("✅ Dynamic simulation complete!")

async def main():
    """Completely dynamic main function"""
    # Load configuration dynamically
    config_dir = Path(os.getenv('CONFIG_DIR', 'config'))

    # Debug API key loading
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        logging.error("❌ OpenRouter API key not found in environment")
        return
    else:
        # Only show first few chars for security
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        logging.info(f"✅ API key found: {masked_key}")

    try:
        async with TrenchesAgent(config_dir) as agent:
            # Dynamic health check
            if not await agent.check_backend_health_async():
                backend_url = agent.sim_config.backend_url
                agent.logger.error(f"❌ Backend unavailable at {backend_url}")
                agent.logger.info("💡 Start the backend with: cd backend && go run main.go")
                return

            agent.logger.info("✅ Connected to Trenches backend")
            agent.logger.info("🤖 Dynamic LLM mode enabled")

            # Run dynamic simulation
            await agent.run_dynamic_simulation()

    except KeyboardInterrupt:
        logging.info("⏹️ Simulation interrupted by user")
    except Exception as e:
        logging.error(f"💥 Simulation failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())