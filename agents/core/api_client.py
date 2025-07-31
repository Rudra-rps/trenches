"""API client for Trenches backend communication."""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
import backoff

from models.config import SimulationConfig
from models.entities import Tweet, Profile, TweetStats, AgentStats


class TrenchesAPIClient:
    """Async API client for Trenches backend"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.base_url = config.backend_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        """Initialize session with connection pooling"""
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent_agents,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        timeout = aiohttp.ClientTimeout(total=self.config.backend_timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close session"""
        if self.session:
            await self.session.close()

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def health_check(self) -> bool:
        """Check if backend is healthy"""
        try:
            endpoint = f"{self.config.backend_url}/ping"
            timeout = aiohttp.ClientTimeout(total=self.config.health_check_timeout)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(endpoint) as response:
                    if response.status == 200:
                        data = await response.json()
                        return (data == "pong" or
                               (isinstance(data, dict) and data.get("message") == "pong"))
            return False
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    # Tweet operations
    async def post_tweet(self, tweet: Tweet) -> Optional[Tweet]:
        """Post a new tweet"""
        try:
            endpoint = f"{self.config.backend_url}/tweets"
            async with self.session.post(endpoint, json=tweet.to_dict()) as response:
                response.raise_for_status()
                data = await response.json()

                # Extract tweet from response
                tweet_data = data.get('tweet', {})
                return Tweet(
                    id=tweet_data.get('id'),
                    agent_id=tweet_data.get('agent_id', tweet.agent_id),
                    content=tweet_data.get('content', tweet.content),
                    thread_id=tweet_data.get('thread_id'),
                    likes=tweet_data.get('likes', 0),
                    retweets=tweet_data.get('retweets', 0)
                )
        except Exception as e:
            self.logger.error(f"Failed to post tweet: {e}")
            return None

    async def get_tweets(self, limit: Optional[int] = None) -> List[Tweet]:
        """Get recent tweets"""
        try:
            endpoint = f"{self.config.backend_url}/tweets"
            async with self.session.get(endpoint) as response:
                response.raise_for_status()
                tweets_data = await response.json()

                tweets = []
                for tweet_data in tweets_data:
                    tweets.append(Tweet(
                        id=tweet_data.get('id'),
                        agent_id=tweet_data.get('agent_id'),
                        content=tweet_data.get('content'),
                        thread_id=tweet_data.get('thread_id'),
                        likes=tweet_data.get('likes', 0),
                        retweets=tweet_data.get('retweets', 0)
                    ))

                if limit:
                    return tweets[:limit]
                return tweets
        except Exception as e:
            self.logger.error(f"Failed to get tweets: {e}")
            return []

    async def get_timeline(self, limit: int = 20) -> List[Tweet]:
        """Get timeline with limit"""
        try:
            endpoint = f"{self.config.backend_url}/timeline"
            params = {"limit": limit}
            async with self.session.get(endpoint, params=params) as response:
                response.raise_for_status()
                tweets_data = await response.json()

                tweets = []
                for tweet_data in tweets_data:
                    tweets.append(Tweet(
                        id=tweet_data.get('id'),
                        agent_id=tweet_data.get('agent_id'),
                        content=tweet_data.get('content'),
                        thread_id=tweet_data.get('thread_id'),
                        likes=tweet_data.get('likes', 0),
                        retweets=tweet_data.get('retweets', 0)
                    ))
                return tweets
        except Exception as e:
            self.logger.error(f"Failed to get timeline: {e}")
            return []

    async def like_tweet(self, tweet_id: int) -> bool:
        """Like a tweet"""
        try:
            endpoint = f"{self.config.backend_url}/tweets/{tweet_id}/likes"
            async with self.session.post(endpoint) as response:
                response.raise_for_status()
                return True
        except Exception as e:
            self.logger.error(f"Failed to like tweet {tweet_id}: {e}")
            return False

    async def retweet(self, tweet_id: int) -> bool:
        """Retweet a tweet"""
        try:
            endpoint = f"{self.config.backend_url}/tweets/{tweet_id}/retweets"
            async with self.session.post(endpoint) as response:
                response.raise_for_status()
                return True
        except Exception as e:
            self.logger.error(f"Failed to retweet {tweet_id}: {e}")
            return False

    async def reply_to_tweet(self, tweet_id: int, reply: Tweet) -> Optional[Tweet]:
        """Reply to a tweet"""
        try:
            endpoint = f"{self.config.backend_url}/tweets/{tweet_id}/reply"
            async with self.session.post(endpoint, json=reply.to_dict()) as response:
                response.raise_for_status()
                data = await response.json()

                tweet_data = data.get('tweet', {})
                return Tweet(
                    id=tweet_data.get('id'),
                    agent_id=tweet_data.get('agent_id', reply.agent_id),
                    content=tweet_data.get('content', reply.content),
                    thread_id=tweet_data.get('thread_id'),
                    likes=tweet_data.get('likes', 0),
                    retweets=tweet_data.get('retweets', 0)
                )
        except Exception as e:
            self.logger.error(f"Failed to reply to tweet {tweet_id}: {e}")
            return None

    async def get_tweet_stats(self, tweet_id: int) -> Optional[TweetStats]:
        """Get statistics for a specific tweet"""
        try:
            endpoint = f"{self.config.backend_url}/tweets/{tweet_id}/stats"
            async with self.session.get(endpoint) as response:
                response.raise_for_status()
                data = await response.json()

                return TweetStats(
                    likes=data.get('likes', 0),
                    retweets=data.get('retweets', 0),
                    replies=data.get('replies', 0)
                )
        except Exception as e:
            self.logger.error(f"Failed to get tweet stats for {tweet_id}: {e}")
            return None

    # Profile operations
    async def create_profile(self, profile: Profile) -> Optional[Profile]:
        """Create a new profile"""
        try:
            endpoint = f"{self.config.backend_url}/profiles"
            async with self.session.post(endpoint, json=profile.to_dict()) as response:
                response.raise_for_status()
                data = await response.json()

                profile_data = data.get('profile', {})
                return Profile(
                    id=profile_data.get('id'),
                    username=profile_data.get('username', profile.username),
                    avatar=profile_data.get('avatar', profile.avatar),
                    metadata=profile_data.get('metadata', profile.metadata)
                )
        except Exception as e:
            self.logger.error(f"Failed to create profile: {e}")
            return None

    async def get_profiles(self) -> List[Profile]:
        """Get all profiles"""
        try:
            endpoint = f"{self.config.backend_url}/profiles"
            async with self.session.get(endpoint) as response:
                response.raise_for_status()
                profiles_data = await response.json()

                profiles = []
                for profile_data in profiles_data:
                    profiles.append(Profile(
                        id=profile_data.get('id'),
                        username=profile_data.get('username'),
                        avatar=profile_data.get('avatar'),
                        metadata=profile_data.get('metadata')
                    ))
                return profiles
        except Exception as e:
            self.logger.error(f"Failed to get profiles: {e}")
            return []

    # Statistics operations
    async def get_agent_stats(self) -> List[AgentStats]:
        """Get statistics for all agents"""
        try:
            endpoint = f"{self.config.backend_url}/stats"
            async with self.session.get(endpoint) as response:
                response.raise_for_status()
                stats_data = await response.json()

                stats = []
                for stat_data in stats_data:
                    stats.append(AgentStats(
                        agent_id=stat_data.get('agent_id'),
                        total_likes=stat_data.get('total_likes', 0),
                        total_retweets=stat_data.get('total_retweets', 0)
                    ))
                return stats
        except Exception as e:
            self.logger.error(f"Failed to get agent stats: {e}")
            return []

    async def get_tweets_with_stats(self) -> List[Tweet]:
        """Get all tweets ordered by engagement"""
        try:
            endpoint = f"{self.config.backend_url}/tweets/stats"
            async with self.session.get(endpoint) as response:
                response.raise_for_status()
                tweets_data = await response.json()

                tweets = []
                for tweet_data in tweets_data:
                    tweets.append(Tweet(
                        id=tweet_data.get('id'),
                        agent_id=tweet_data.get('agent_id'),
                        content=tweet_data.get('content'),
                        thread_id=tweet_data.get('thread_id'),
                        likes=tweet_data.get('likes', 0),
                        retweets=tweet_data.get('retweets', 0)
                    ))
                return tweets
        except Exception as e:
            self.logger.error(f"Failed to get tweets with stats: {e}")
            return []

    # NEW: Add this method to the class
    async def save_wallet_snapshot(self, address: str, balance: float, block_number: int) -> bool:
        """Saves a wallet snapshot to the backend using aiohttp."""
        payload = {
            "wallet_address": address,
            "balance": balance,
            "block_number": block_number
        }
        try:
            endpoint = f"{self.base_url}/wallet_snapshots"
            async with self.session.post(endpoint, json=payload) as response:
                response.raise_for_status()
                self.logger.info(f"✅ Wallet snapshot saved for {address}")
                return True
        except aiohttp.ClientError as e:
            self.logger.error(f"❌ Failed to save wallet snapshot for {address}: {e}")
            return False