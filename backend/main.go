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
}

var db *sqlx.DB

func main() {
	var err error
	db, err = sqlx.Connect("postgres", "host=localhost port=5432 user=trenches password=secret dbname=trenches sslmode=disable")
	if err != nil {
		log.Fatalln("DB connection error:", err)
	}

	// Create table if not exists
	schema := `CREATE TABLE IF NOT EXISTS tweets (
		id SERIAL PRIMARY KEY,
		agent_id TEXT NOT NULL,
		content TEXT NOT NULL,
		thread_id INT
	);`
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
		_, err := db.NamedExec(`INSERT INTO tweets (agent_id, content, thread_id) VALUES (:agent_id, :content, :thread_id)`, &tweet)
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

	r.Run(":8080")
}
