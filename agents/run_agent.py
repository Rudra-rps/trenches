import time
import random
import yaml
import requests
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrenchesAgent:
    def __init__(self, backend_url: str = "http://localhost:8080", llm_config: Dict = None):
        self.backend_url = backend_url
        self.llm_config = llm_config or {}

    def load_agent_config(self, path: Path) -> Dict:
        """Load agent configuration from YAML file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if not config:
                    raise ValueError(f"Empty configuration file: {path}")
                return config
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {path}: {e}")
            raise
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {path}")
            raise

    def post_tweet(self, agent_id: str, content: str, thread_id: Optional[int] = None) -> bool:
        """Post a tweet via the Go backend"""
        payload = {
            "agent_id": agent_id,
            "content": content
        }
        if thread_id:
            payload["thread_id"] = thread_id

        try:
            response = requests.post(f"{self.backend_url}/tweets", json=payload, timeout=5)
            response.raise_for_status()
            logger.info(f"✅ Tweet posted by {agent_id}: {content[:50]}...")
            return True
        except requests.RequestException as e:
            logger.error(f"❌ Failed to post tweet for {agent_id}: {e}")
            return False

    def get_recent_tweets(self, limit: int = 10) -> List[Dict]:
        """Fetch recent tweets for context"""
        try:
            response = requests.get(f"{self.backend_url}/tweets", timeout=5)
            response.raise_for_status()
            tweets = response.json()
            return tweets[:limit] if tweets else []
        except requests.RequestException as e:
            logger.error(f"Failed to fetch tweets: {e}")
            return []

    def _build_personality_prompt(self, agent: Dict, recent_tweets: List[Dict] = None) -> str:
        """Build personality-aware prompt for LLM"""
        personality = agent.get('personality', {})
        agent_id = agent.get('id', 'unknown')

        # Extract all personality traits from the agent config
        temperament = personality.get('temperament', 'neutral')
        tone = personality.get('tone', 'neutral')
        emotionality = personality.get('emotionality', 'medium')
        decision_bias = personality.get('decision_bias', 'balanced')

        prompt = f"""You are {agent_id}, with these personality traits:
    - Temperament: {temperament}
    - Tone: {tone}
    - Emotionality: {emotionality}
    - Decision Bias: {decision_bias}

    Generate a tweet (max 280 characters) that reflects your personality.
    You can tweet about any topic that interests you - technology, life observations,
    current events, personal thoughts, humor, philosophy, or anything else.
    Be authentic to your character traits and speak in your unique voice.
    Use appropriate emojis and hashtags when relevant.
    """

        if recent_tweets:
            tweet_context = "\n".join([f"- {t.get('content', '')[:50]}" for t in recent_tweets[:3]])
            prompt += f"\n\nRecent community tweets for context:\n{tweet_context}\n"

        prompt += "\nYour tweet:"
        return prompt

    def _generate_with_openrouter(self, agent: Dict, prompt: str) -> str:
        """Generate using OpenRouter API via OpenAI SDK"""
        # Get configuration
        agent_llm = agent.get('llm', {})
        model = agent_llm.get('model', self.llm_config.get('model', 'openai/gpt-3.5-turbo'))
        temperature = agent_llm.get('temperature', self.llm_config.get('temperature', 0.7))

        # Get API key and ensure it's properly formatted
        api_key = self.llm_config.get('api_key') or os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OpenRouter API key not found")

        # Make sure the API key is properly formatted (no whitespace)
        api_key = api_key.strip()

        # Log connection attempt (limited key visibility for security)
        if len(api_key) > 8:
            visible_part = api_key[:4] + '...' + api_key[-4:]
            logger.info(f"Connecting to OpenRouter with key starting with: {visible_part}")

        try:
            # Initialize OpenAI client with proper headers
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": "https://trenches-social.com",
                    "X-Title": "Trenches Social Sim"
                }
            )

            # Make the completion request
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=100
            )

            # Extract and format content
            content = completion.choices[0].message.content.strip()
            return content[:280] if len(content) > 280 else content

        except Exception as e:
            error_message = str(e)
            logger.error(f"OpenRouter API error: {error_message}")

            # More helpful error for authentication issues
            if "401" in error_message:
                raise ValueError(f"OpenRouter authentication failed. Please check your API key format and permissions. Error: {error_message}")
            raise

    def generate_content(self, agent: Dict, recent_tweets: List[Dict] = None) -> str:
        """Generate content using LLM only - no templates"""
        agent_id = agent.get('id', 'unknown')

        try:
            prompt = self._build_personality_prompt(agent, recent_tweets)
            content = self._generate_with_openrouter(agent, prompt)
            return content
        except Exception as e:
            logger.error(f"❌ [{agent_id}] Failed to generate content: {e}")
            raise Exception(f"Content generation failed for {agent_id}: {e}")

    def simulate_action(self, agent: Dict, recent_tweets: List[Dict] = None):
        """Simulate various agent actions"""
        agent_id = agent.get('id')
        if not agent_id:
            logger.error("Agent missing required 'id' field")
            return

        # Get activity configuration
        activity_config = agent.get('activity', {})
        actions_range = activity_config.get('actions_per_awake', [1, 2])

        # Validate actions_range
        if len(actions_range) != 2 or actions_range[0] > actions_range[1]:
            logger.warning(f"Invalid actions_per_awake range for {agent_id}, using default [1, 2]")
            actions_range = [1, 2]

        # Determine number of actions for this agent
        num_actions = random.randint(actions_range[0], actions_range[1])

        for _ in range(num_actions):
            # Define action probabilities based on personality
            personality_info = agent.get('personality', {})
            temperament = personality_info.get('temperament', 'neutral')

            # Adjust probabilities based on temperament
            if temperament == 'analytical':
                actions = {'tweet': 0.8, 'like': 0.1, 'retweet': 0.05, 'reply': 0.05}
            elif temperament == 'sarcastic':
                actions = {'tweet': 0.7, 'like': 0.1, 'retweet': 0.1, 'reply': 0.1}
            elif temperament == 'optimistic':
                actions = {'tweet': 0.5, 'like': 0.3, 'retweet': 0.15, 'reply': 0.05}
            elif temperament == 'playful':
                actions = {'tweet': 0.8, 'like': 0.1, 'retweet': 0.05, 'reply': 0.05}
            elif temperament == 'contemplative':
                actions = {'tweet': 0.9, 'like': 0.05, 'retweet': 0.03, 'reply': 0.02}
            else:
                actions = {'tweet': 0.6, 'like': 0.2, 'retweet': 0.15, 'reply': 0.05}

            # Choose action based on probabilities
            action = random.choices(list(actions.keys()), weights=list(actions.values()))[0]

            if action == 'tweet':
                try:
                    content = self.generate_content(agent, recent_tweets)
                    success = self.post_tweet(agent_id, content)
                    if success:
                        logger.info(f"🐦 [{agent_id}] tweeted")
                except Exception as e:
                    logger.error(f"❌ [{agent_id}] Failed to tweet: {e}")
            else:
                # For MVP, just log other actions
                logger.info(f"📱 [{agent_id}] {action}d something...")

            # Small delay between actions from same agent
            if num_actions > 1:
                time.sleep(random.uniform(0.5, 1.5))

    def check_backend_health(self) -> bool:
        """Check if Go backend is running"""
        try:
            response = requests.get(f"{self.backend_url}/ping", timeout=3)
            if response.status_code == 200:
                response_data = response.json()
                return (response_data == "pong" or
                       (isinstance(response_data, dict) and response_data.get("message") == "pong"))
            return False
        except requests.RequestException:
            return False

def main():
    # Debug API key loading
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("❌ OpenRouter API key not found in environment")
        return
    else:
        # Only show first few chars for security
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        logger.info(f"✅ API key found: {masked_key}")

        # Verify API key format
        if not api_key.startswith("sk-"):
            logger.warning("⚠️ API key doesn't start with 'sk-', may not be correctly formatted")

    # LLM configuration for OpenRouter - global defaults
    llm_config = {
        'api_key': api_key,
        'model': 'openai/gpt-3.5-turbo',  # Using a widely available model
        'temperature': 0.7  # Default fallback temperature
    }

    agent_runner = TrenchesAgent(llm_config=llm_config)

    # Check backend connection
    if not agent_runner.check_backend_health():
        logger.error("❌ Cannot connect to Go backend. Make sure it's running on :8080")
        logger.info("💡 Start the backend with: cd backend && go run main.go")
        return

    logger.info("✅ Connected to Trenches backend")
    logger.info("🤖 LLM-only mode enabled - using agent-specific models and configs")

    # Load all agent configs
    config_path = Path("agent_spec")
    if not config_path.exists():
        logger.error(f"Agent spec directory '{config_path}' not found")
        logger.info("💡 Create agent_spec/ directory with YAML files")
        return

    config_files = list(config_path.glob("*.yaml"))
    if not config_files:
        logger.warning(f"No YAML files found in '{config_path}'")
        return

    logger.info(f"📁 Found {len(config_files)} agent configurations")

    # Simulation loop
    try:
        for round_num in range(3):  # 3 rounds of activity
            logger.info(f"\n🔄 === Round {round_num + 1}/3 ===")

            # Get recent tweets for context
            recent_tweets = agent_runner.get_recent_tweets(limit=5)
            if recent_tweets:
                logger.info(f"📰 Context: {len(recent_tweets)} recent tweets available")

            # Simulate each agent
            for config_file in config_files:
                try:
                    agent = agent_runner.load_agent_config(config_file)
                    agent_runner.simulate_action(agent, recent_tweets)

                    # Random delay between agents (1-3 seconds)
                    time.sleep(random.uniform(1, 3))

                except Exception as e:
                    logger.error(f"Error processing {config_file}: {e}")

            # Longer delay between rounds
            if round_num < 2:  # Don't sleep after last round
                logger.info("⏳ Waiting before next round...")
                time.sleep(random.randint(8, 15))

        logger.info("\n✅ Simulation complete!")

    except KeyboardInterrupt:
        logger.info("\n⏹️  Simulation stopped by user")

if __name__ == "__main__":
    main()