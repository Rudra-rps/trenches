"""LLM client for content generation using Groq API."""

import asyncio
import logging
import os
from typing import Dict, Optional

from openai import OpenAI
import backoff

from models.config import LLMConfig


class LLMClient:
    """Client for LLM content generation via Groq API"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _generate_with_groq(self, agent: Dict, prompt: str) -> str:
        """Generate content using Groq API via OpenAI SDK"""
        # Get configuration
        agent_llm = agent.get('llm', {})
        model = agent_llm.get('model', self.config.default_model)
        temperature = agent_llm.get('temperature', self.config.default_temperature)
        max_tokens = agent_llm.get('max_tokens', self.config.default_max_tokens)

        # Get API key
        api_key = self.config.api_key or os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("Groq API key not found")

        api_key = api_key.strip()

        try:
            # Initialize OpenAI client for Groq
            client = OpenAI(
                base_url=self.config.base_url,
                api_key=api_key
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
            self.logger.error(f"Groq API error: {error_message}")

            if "401" in error_message:
                raise ValueError(f"Groq authentication failed: {error_message}")
            raise

    async def generate_content_async(self, agent: Dict, prompt: str) -> str:
        """Generate content asynchronously"""
        agent_id = agent.get('id', 'unknown')

        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None,
                self._generate_with_groq,
                agent,
                prompt
            )
            return content
        except Exception as e:
            self.logger.error(f"[{agent_id}] Failed to generate content: {e}")
            raise Exception(f"Content generation failed for {agent_id}: {e}")

    def validate_api_key(self) -> bool:
        """Validate that the API key is working"""
        try:
            api_key = self.config.api_key or os.getenv('GROQ_API_KEY')
            if not api_key:
                return False

            client = OpenAI(
                base_url=self.config.base_url,
                api_key=api_key.strip()
            )

            # Test with a simple request
            completion = client.chat.completions.create(
                model=self.config.default_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )

            return completion.choices[0].message.content is not None
        except Exception as e:
            self.logger.error(f"API key validation failed: {e}")
            return False
