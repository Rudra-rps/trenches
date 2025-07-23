"""Data models for backend entities."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    TWEET = "tweet"
    LIKE = "like"
    RETWEET = "retweet"
    REPLY = "reply"
    FOLLOW = "follow"


@dataclass
class Tweet:
    """Tweet entity matching backend model"""
    id: Optional[int] = None
    agent_id: str = ""
    content: str = ""
    thread_id: Optional[int] = None
    likes: int = 0
    retweets: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls"""
        data = {
            "agent_id": self.agent_id,
            "content": self.content
        }
        if self.thread_id:
            data["thread_id"] = self.thread_id
        return data


@dataclass
class Profile:
    """Profile entity matching backend model"""
    id: Optional[int] = None
    username: str = ""
    avatar: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls"""
        return {
            "username": self.username,
            "avatar": self.avatar,
            "metadata": self.metadata or {}
        }


@dataclass
class TweetStats:
    """Tweet statistics from backend"""
    likes: int = 0
    retweets: int = 0
    replies: int = 0


@dataclass
class AgentStats:
    """Agent statistics from backend"""
    agent_id: str = ""
    total_likes: int = 0
    total_retweets: int = 0


@dataclass
class SimulationContext:
    """Context information for simulation"""
    trending_topics: List[str] = None
    activity_level: str = "medium"
    sentiment: str = "neutral"
    time_context: str = "unknown"
    recent_tweets: List[Tweet] = None

    def __post_init__(self):
        if self.trending_topics is None:
            self.trending_topics = []
        if self.recent_tweets is None:
            self.recent_tweets = []
