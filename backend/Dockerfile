# Use official Golang base image
FROM golang:1.24

# Set working directory inside container
WORKDIR /app

# Copy Go module files & download dependencies first (for build cache)
COPY go.mod ./
COPY go.sum ./
RUN go mod download

# Copy the rest of your code
COPY . .

# Build the Go app (produces 'app' binary)
RUN go build -o app 

# Expose the port your app listens on
EXPOSE 8080

# Run the binary
CMD ["./app"]
