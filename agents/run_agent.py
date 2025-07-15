import time
import random
import yaml
import requests
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv('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')

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

        prompt = f"""You are {agent_id}, a software engineer with these personality traits:
- Temperament: {personality.get('temperament', 'neutral')}
- Tone: {personality.get('tone', 'neutral')}
- Emotionality: {personality.get('emotionality', 'medium')}

Generate a tweet (max 280 characters) that reflects your personality.
Focus on software engineering, debugging, or tech insights.
Be authentic to your character traits.
Use appropriate emojis and hashtags when relevant.
"""

        if recent_tweets:
            tweet_context = "\n".join([f"- {t.get('content', '')[:50]}" for t in recent_tweets[:3]])
            prompt += f"\n\nRecent community tweets for context:\n{tweet_context}\n"

        prompt += "\nYour tweet:"
        return prompt

    def _generate_with_openrouter(self, prompt: str) -> str:
        """Generate using OpenRouter API"""
        api_key = self.llm_config.get('api_key') or os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OpenRouter API key not found")

        headers = {
            "Authorization": f"Bearer {"OPENROUTER_API_KEY"}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8080",  # Optional
            "X-Title": "Trenches Social Sim"  # Optional
        }

        payload = {
            "model": self.llm_config.get('model', 'openai/gpt-3.5-turbo'),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": self.llm_config.get('temperature', 0.7)
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        return result['choices'][0]['message']['content'].strip()

    def _generate_with_templates(self, agent: Dict, recent_tweets: List[Dict] = None) -> str:
        """Original template-based generation as fallback"""
        personality_info = agent.get('personality', {})
        temperament = personality_info.get('temperament', 'neutral')
        tone = personality_info.get('tone', 'neutral')
        emotionality = personality_info.get('emotionality', 'medium')

        agent_id = agent.get('id', 'unknown')

        # Content templates based on temperament and tone
        content_templates = {
            'curious_analytical': [
                "Interesting pattern emerging in today's data... 🤔 #DataDriven",
                "Question: What's the most underrated debugging technique? 🔍",
                "Analyzing the correlation between code complexity and bug density...",
                "Deep dive into performance metrics reveals some fascinating insights 📊"
            ],
            'curious_neutral': [
                "Wonder what causes this particular edge case... 🤔",
                "Exploring new approaches to this problem space",
                "What's everyone's take on this architectural decision?",
                "Investigating the root cause of today's production hiccup 🔍"
            ],
            'analytical_low': [
                "Performance optimization complete: 23% improvement in latency",
                "Static analysis reveals 12 potential code smell patterns",
                "Metrics suggest we should refactor the authentication module",
                "Code review findings: 3 critical, 7 minor issues identified"
            ],
            'cautious_analytical': [
                "Before we deploy, let's double-check the rollback strategy...",
                "Running additional tests to validate edge case handling",
                "Careful consideration needed for this database migration",
                "Triple-checking the security implications of this change 🔐"
            ]
        }

        # Create composite key for personality matching
        composite_key = f"{temperament}_{tone}"
        if emotionality == 'low':
            composite_key += f"_{emotionality}"

        # Find best matching template
        for key, templates in content_templates.items():
            if composite_key in key or any(trait in key for trait in [temperament, tone]):
                return random.choice(templates)

        # Default templates based on individual traits
        if temperament == 'curious':
            defaults = [
                "Hmm, this is an interesting problem to solve... 🤔",
                "What's the best approach here? Looking for insights 💭",
                "Exploring different solutions to today's challenge"
            ]
        elif tone == 'analytical':
            defaults = [
                "Breaking down the problem into smaller components",
                "Data suggests we need to reconsider our approach",
                "Systematic analysis of the current implementation"
            ]
        else:
            defaults = [
                f"Another day, another challenge in the trenches... #{agent_id}",
                "Working through today's technical problems step by step",
                "Progress update: steady improvements across the board 📈"
            ]

        return random.choice(defaults)

    def generate_content(self, agent: Dict, recent_tweets: List[Dict] = None) -> str:
        """Generate content using LLM or fallback to templates"""
        use_llm = self.llm_config.get('enabled', False)

        if use_llm:
            try:
                prompt = self._build_personality_prompt(agent, recent_tweets)
                return self._generate_with_openrouter(prompt)
            except Exception as e:
                logger.warning(f"LLM generation failed, using templates: {e}")

        # Fallback to existing template system
        return self._generate_with_templates(agent, recent_tweets)

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
            # Define action probabilities (adjusted for analytical agents)
            personality_info = agent.get('personality', {})
            if personality_info.get('tone') == 'analytical':
                actions = {'tweet': 0.8, 'like': 0.1, 'retweet': 0.05, 'reply': 0.05}
            else:
                actions = {'tweet': 0.6, 'like': 0.2, 'retweet': 0.15, 'reply': 0.05}

            # Choose action based on probabilities
            action = random.choices(list(actions.keys()), weights=list(actions.values()))[0]

            if action == 'tweet':
                content = self.generate_content(agent, recent_tweets)
                success = self.post_tweet(agent_id, content)
                if success:
                    logger.info(f"🐦 [{agent_id}] tweeted")
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
                # Handle both possible response formats
                response_data = response.json()
                return (response_data == "pong" or
                       (isinstance(response_data, dict) and response_data.get("message") == "pong"))
            return False
        except requests.RequestException:
            return False

def main():
    # LLM configuration for OpenRouter
    llm_config = {
        'enabled': True,  # Set to False to use templates only
        'api_key': os.getenv('OPENROUTER_API_KEY'),  # Set this in your .env file
        'model': 'openai/gpt-3.5-turbo',  # or 'anthropic/claude-3-sonnet', etc.
        'temperature': 0.7
    }

    agent_runner = TrenchesAgent(llm_config=llm_config)

    # Check backend connection
    if not agent_runner.check_backend_health():
        logger.error("❌ Cannot connect to Go backend. Make sure it's running on :8080")
        logger.info("💡 Start the backend with: cd backend && go run main.go")
        return

    logger.info("✅ Connected to Trenches backend")

    # Check LLM configuration
    if llm_config['enabled']:
        if llm_config['api_key']:
            logger.info(f"🤖 LLM enabled: {llm_config['model']}")
        else:
            logger.warning("⚠️ LLM enabled but no API key found, falling back to templates")
            llm_config['enabled'] = False
    else:
        logger.info("📝 Using template-based content generation")

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