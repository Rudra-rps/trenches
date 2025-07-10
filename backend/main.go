package main

import (
	"database/sql/driver"
	"encoding/json"
	"log"
	"net/http"
	"os"

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
	log.Println("Logging event:", event)
	file, err := os.OpenFile("events.jsonl", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Println("Failed to open events log:", err)
		return
	}
	defer file.Close()

	bytes, _ := json.Marshal(event)
	file.Write(bytes)
	file.Write([]byte("\n"))
}

func main() {
	var err error
	db, err = sqlx.Connect("postgres", "host=127.0.0.1 port=5433 user=trenches password=secret dbname=trenches sslmode=disable")
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

	// Start server
	r.Run(":8080")
}
