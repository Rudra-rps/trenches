package main

import (
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/jmoiron/sqlx"
	_ "github.com/lib/pq"
)

type Tweet struct {
	ID       int    `db:"id" json:"id"`
	AgentID  string `db:"agent_id" json:"agent_id"`
	Content  string `db:"content" json:"content"`
	ThreadID *int   `db:"thread_id" json:"thread_id,omitempty"`
	Likes    int    `db:"likes" json:"likes"`
	Retweets int    `db:"retweets" json:"retweets"`
}

type Profile struct {
	ID       int            `db:"id" json:"id"`
	Username string         `db:"username" json:"username"`
	Avatar   string         `db:"avatar" json:"avatar"`
	Metadata map[string]any `db:"metadata" json:"metadata"`
}

var db *sqlx.DB

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

	r.GET("/ping", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "pong"})
	})

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

		c.JSON(http.StatusCreated, gin.H{"status": "tweet posted", "tweet": tweet})
	})

	r.GET("/tweets", func(c *gin.Context) {
		var tweets []Tweet
		err := db.Select(&tweets, "SELECT * FROM tweets ORDER BY id DESC")
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, tweets)
	})

	// 👍 Like a Tweet
	r.POST("/tweets/:id/likes", func(c *gin.Context) {
		id := c.Param("id")
		_, err := db.Exec(`UPDATE tweets SET likes = COALESCE(likes, 0) + 1 WHERE id=$1`, id)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"status": "tweet liked"})
	})

	// 🔁 Retweet a Tweet
	r.POST("/tweets/:id/retweets", func(c *gin.Context) {
		id := c.Param("id")
		_, err := db.Exec(`UPDATE tweets SET retweets = COALESCE(retweets, 0) + 1 WHERE id=$1`, id)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"status": "tweet retweeted"})
	})

	// 💬 Reply to a Tweet
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

		c.JSON(http.StatusCreated, gin.H{"status": "reply posted", "tweet": tweet})
	})

	// ✅ Start server last!
	r.Run(":8080")
}
