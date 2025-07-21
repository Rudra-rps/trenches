package main

import (
	"database/sql/driver"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/jmoiron/sqlx"
	_ "github.com/lib/pq"
)

type JSONB map[string]any

func (j *JSONB) Scan(src interface{}) error {
	if src == nil {
		*j = nil
		return nil
	}
	switch data := src.(type) {

	case []byte:
		return json.Unmarshal(data, j)
	case string:
		return json.Unmarshal([]byte(data), j)

	default:
		return nil
	}
}

func (j JSONB) Value() (driver.Value, error) {
	if j == nil {
		return nil, nil
	}
	return json.Marshal(j)
}

type Tweet struct {
	ID       int    `db:"id" json:"id"`
	AgentID  string `db:"agent_id" json:"agent_id"`
	Content  string `db:"content" json:"content"`
	ThreadID *int   `db:"thread_id" json:"thread_id,omitempty"`
	Likes    int    `db:"likes" json:"likes"`
	Retweets int    `db:"retweets" json:"retweets"`
}

type Profile struct {
	ID       int    `db:"id" json:"id"`
	Username string `db:"username" json:"username"`
	Avatar   string `db:"avatar" json:"avatar"`
	Metadata JSONB  `db:"metadata" json:"metadata"`
}

var db *sqlx.DB

func logEvent(event any) {
	wrapped := map[string]any{
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"event":     event,
	}

	log.Println("Logging event:", wrapped)

	file, err := os.OpenFile("events.jsonl", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Println("Failed to open events log:", err)
		return
	}
	defer file.Close()

	bytes, _ := json.Marshal(wrapped)
	file.Write(bytes)
	file.Write([]byte("\n"))
}

func main() {
	var err error
	db, err = sqlx.Connect("postgres", "host=postgres port=5432 user=trenches password=secret dbname=trenches sslmode=disable")
	if err != nil {
		log.Fatalln("DB connection error:", err)
	}

	schema := `
	CREATE TABLE IF NOT EXISTS tweets (
		id SERIAL PRIMARY KEY,
		agent_id TEXT NOT NULL,
		content TEXT NOT NULL,
		thread_id INT,
		likes INT DEFAULT 0,
		retweets INT DEFAULT 0
	);

	CREATE TABLE IF NOT EXISTS profiles (
		id SERIAL PRIMARY KEY,
		username TEXT NOT NULL,
		avatar TEXT,
		metadata JSONB
	);
	`
	db.MustExec(schema)

	r := gin.Default()

	// Ping
	r.GET("/ping", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "pong"})
	})

	// Create Tweet
	r.POST("/tweets", func(c *gin.Context) {
		var tweet Tweet
		if err := c.ShouldBindJSON(&tweet); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		err := db.QueryRowx(
			`INSERT INTO tweets (agent_id, content, thread_id)
			 VALUES ($1, $2, $3) RETURNING id`,
			tweet.AgentID, tweet.Content, tweet.ThreadID,
		).Scan(&tweet.ID)

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		logEvent(tweet)
		c.JSON(http.StatusCreated, gin.H{"status": "tweet posted", "tweet": tweet})
	})

	// Get Tweets
	r.GET("/tweets", func(c *gin.Context) {
		var tweets []Tweet
		err := db.Select(&tweets, "SELECT * FROM tweets ORDER BY id DESC")
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, tweets)
	})

	// Like
	r.POST("/tweets/:id/likes", func(c *gin.Context) {
		id := c.Param("id")
		_, err := db.Exec(`UPDATE tweets SET likes = COALESCE(likes, 0) + 1 WHERE id=$1`, id)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		logEvent(map[string]any{"action": "like", "tweet_id": id})
		c.JSON(http.StatusOK, gin.H{"status": "tweet liked"})
	})

	// 👍 Get likes count for a tweet
	r.GET("/tweets/:id/likes", func(c *gin.Context) {
		id := c.Param("id")
		var likes int
		err := db.Get(&likes, "SELECT likes FROM tweets WHERE id=$1", id)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "Tweet not found"})
			return
		}
		c.JSON(http.StatusOK, gin.H{"likes": likes})
	})

	// Retweet
	r.POST("/tweets/:id/retweets", func(c *gin.Context) {
		id := c.Param("id")
		_, err := db.Exec(`UPDATE tweets SET retweets = COALESCE(retweets, 0) + 1 WHERE id=$1`, id)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		logEvent(map[string]any{"action": "retweet", "tweet_id": id})
		c.JSON(http.StatusOK, gin.H{"status": "tweet retweeted"})
	})

	// 🔁 Get retweets count for a tweet
	r.GET("/tweets/:id/retweets", func(c *gin.Context) {
		id := c.Param("id")
		var retweets int
		err := db.Get(&retweets, "SELECT retweets FROM tweets WHERE id=$1", id)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "Tweet not found"})
			return
		}
		c.JSON(http.StatusOK, gin.H{"retweets": retweets})
	})

	// Reply
	r.POST("/tweets/:id/reply", func(c *gin.Context) {
		parentID := c.Param("id")
		var tweet Tweet
		if err := c.ShouldBindJSON(&tweet); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		err := db.QueryRowx(
			`INSERT INTO tweets (agent_id, content, thread_id)
			 VALUES ($1, $2, $3) RETURNING id`,
			tweet.AgentID, tweet.Content, parentID,
		).Scan(&tweet.ID)

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		logEvent(tweet)
		c.JSON(http.StatusCreated, gin.H{"status": "reply posted", "tweet": tweet})
	})

	// --- ✅ PROFILES CRUD ---

	// Create
	r.POST("/profiles", func(c *gin.Context) {
		var p Profile
		if err := c.ShouldBindJSON(&p); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		metadataBytes, _ := json.Marshal(p.Metadata)

		err := db.QueryRowx(
			`INSERT INTO profiles (username, avatar, metadata) VALUES ($1, $2, $3) RETURNING id`,
			p.Username, p.Avatar, metadataBytes,
		).Scan(&p.ID)

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		logEvent(p)
		c.JSON(http.StatusCreated, gin.H{"status": "profile created", "profile": p})
	})

	// Get all
	r.GET("/profiles", func(c *gin.Context) {
		var profiles []Profile
		err := db.Select(&profiles, "SELECT * FROM profiles")
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, profiles)
	})

	// Get one
	r.GET("/profiles/:id", func(c *gin.Context) {
		id := c.Param("id")
		var p Profile
		err := db.Get(&p, "SELECT * FROM profiles WHERE id=$1", id)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "Profile not found"})
			return
		}
		c.JSON(http.StatusOK, p)
	})
	// GET /experiments/:id/export
	r.GET("/experiments/:id/export", func(c *gin.Context) {

		filePath := "events.jsonl"

		// Check if file exists
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			c.JSON(http.StatusNotFound, gin.H{"error": "No events log found"})
			return
		}

		// Set headers for download
		c.Header("Content-Disposition", "attachment; filename=events.jsonl")
		c.Header("Content-Type", "application/json")

		c.File(filePath)
	})

	// Update
	r.PUT("/profiles/:id", func(c *gin.Context) {
		id := c.Param("id")
		var p Profile
		if err := c.ShouldBindJSON(&p); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		metadataBytes, _ := json.Marshal(p.Metadata)

		_, err := db.Exec(
			`UPDATE profiles SET username=$1, avatar=$2, metadata=$3 WHERE id=$4`,
			p.Username, p.Avatar, metadataBytes, id,
		)

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		logEvent(map[string]any{"action": "profile_updated", "profile_id": id})
		c.JSON(http.StatusOK, gin.H{"status": "profile updated"})
	})

	// Delete
	r.DELETE("/profiles/:id", func(c *gin.Context) {
		id := c.Param("id")
		_, err := db.Exec(`DELETE FROM profiles WHERE id=$1`, id)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		logEvent(map[string]any{"action": "profile_deleted", "profile_id": id})
		c.JSON(http.StatusOK, gin.H{"status": "profile deleted"})
	})

	// 🧾 Get stats for a specific tweet
	r.GET("/tweets/:id/stats", func(c *gin.Context) {
		id := c.Param("id")

		var stats struct {
			Likes    int `db:"likes" json:"likes"`
			Retweets int `db:"retweets" json:"retweets"`
			Replies  int `json:"replies"`
		}

		// Get likes and retweets
		err := db.Get(&stats, "SELECT likes, retweets FROM tweets WHERE id=$1", id)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "Tweet not found"})
			return
		}

		// Count replies to this tweet
		err = db.Get(&stats.Replies, "SELECT COUNT(*) FROM tweets WHERE thread_id=$1", id)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, stats)
	})

	// 🏆 Get all tweets with stats, ordered by likes + retweets
	r.GET("/tweets/stats", func(c *gin.Context) {
		var tweets []Tweet
		err := db.Select(&tweets, "SELECT * FROM tweets ORDER BY (likes + retweets) DESC")
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, tweets)
	})

	// 📁 Export all logged events (for analysis)
	r.GET("/events/export", func(c *gin.Context) {
		filePath := "events.jsonl"

		// Check if the file exists
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			c.JSON(http.StatusNotFound, gin.H{"error": "No events log found"})
			return
		}

		c.Header("Content-Disposition", "attachment; filename=events.jsonl")
		c.Header("Content-Type", "application/json")
		c.File(filePath)
	})

	// 📰 Get timeline (latest tweets, limited)
	r.GET("/timeline", func(c *gin.Context) {
		limit := 20 // default
		if l := c.Query("limit"); l != "" {
			if parsed, err := strconv.Atoi(l); err == nil {
				limit = parsed
			}
		}

		var tweets []Tweet
		err := db.Select(&tweets, "SELECT * FROM tweets ORDER BY id DESC LIMIT $1", limit)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, tweets)
	})

	// GET /stats — return total likes & retweets grouped by agent
	r.GET("/stats", func(c *gin.Context) {
		var stats []struct {
			AgentID       string `db:"agent_id" json:"agent_id"`
			TotalLikes    int    `db:"total_likes" json:"total_likes"`
			TotalRetweets int    `db:"total_retweets" json:"total_retweets"`
		}

		query := `
		SELECT 
			agent_id, 
			COALESCE(SUM(likes), 0) AS total_likes,
			COALESCE(SUM(retweets), 0) AS total_retweets
		FROM tweets
		GROUP BY agent_id
	`

		err := db.Select(&stats, query)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, stats)
	})

	// Start server
	r.Run(":8080")
}
