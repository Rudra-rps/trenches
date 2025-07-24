package main

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/redis/go-redis/v9"
)

var ctx = context.Background()

// RedisClient holds the global Redis connection
var RedisClient *redis.Client

// InitRedis initializes the Redis connection
func InitRedis() {
	RedisClient = redis.NewClient(&redis.Options{
		Addr:     "redis:6379", // Service name in Docker Compose
		Password: "",           // No password for dev
		DB:       0,
	})

	// Test connection
	_, err := RedisClient.Ping(ctx).Result()
	if err != nil {
		log.Fatalf("❌ Redis connection failed: %v", err)
	}
	log.Println("✅ Connected to Redis")
}

// CacheTweet stores a tweet in Redis with a short expiry
func CacheTweet(tweet Tweet) {
	data, err := json.Marshal(tweet)
	if err != nil {
		log.Printf("❌ Failed to marshal tweet: %v", err)
		return
	}

	key := "tweet:" + string(rune(tweet.ID))
	err = RedisClient.Set(ctx, key, data, 5*time.Minute).Err()
	if err != nil {
		log.Printf("❌ Failed to cache tweet: %v", err)
		return
	}

	log.Printf("🗃️ Cached tweet: %d", tweet.ID)
}

// GetCachedTweet tries to get a tweet from cache first
func GetCachedTweet(id string) (*Tweet, error) {
	val, err := RedisClient.Get(ctx, "tweet:"+id).Result()
	if err == redis.Nil {
		// Cache miss
		return nil, nil
	} else if err != nil {
		return nil, err
	}

	var tweet Tweet
	if err := json.Unmarshal([]byte(val), &tweet); err != nil {
		return nil, err
	}

	log.Printf("⚡ Cache hit: %d", tweet.ID)
	return &tweet, nil
}
