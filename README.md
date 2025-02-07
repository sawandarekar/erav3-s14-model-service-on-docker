# Text Generation Application

A web application for text generation with customizable max length parameter.

## Backend Application

The backend is built using Flask and provides a REST API for text generation.

### Features
- REST API endpoint for text processing
- Input validation
- Configurable max length
- Health check endpoint
- Detailed error handling

### Docker Setup

#### Build the Backend Container
Navigate to project root directory
```bash
cd /path/to/project

# Build the backend Docker image
docker build -t text-gen-backend -f Docker.backend .
```

#### Run the Backend Container
```bash
docker run -d -p 8000:8000 --name text-gen-backend
```

## Frontend Application

The frontend is built using Gradio and provides a user-friendly interface for text generation.

### Features
- Text input area
- Adjustable max length slider (1-100)
- Real-time text generation
- Clean and modern UI

### Docker Setup

#### Build the Frontend Container
```bash
# Build the frontend Docker image
docker build -t text-gen-frontend -f Docker.frontend .
```

#### Run the Frontend Container
```bash
docker run -d -p 7860:7860 --name text-gen-frontend
```

## Running the Complete Application

### 1. Start the Backend First
```bash
# Build and run backend
docker build -t text-gen-backend -f Docker.backend .
docker run -d -p 8000:8000 --name text-gen-backend text-gen-backend
```

### 2. Then Start the Frontend
```bash
# Build and run frontend
docker build -t text-gen-frontend -f Docker.frontend .
docker run -d -p 7860:7860 --name text-gen-frontend text-gen-frontend
```

### Accessing the Application
- Frontend UI: http://localhost:7860
- Backend API: http://localhost:8000
- Health Check: http://localhost:8000/health

### Useful Docker Commands
```bash
# View running containers
docker ps

# Stop containers
docker stop text-gen-frontend text-gen-backend

# Remove containers
docker rm text-gen-frontend text-gen-backend

# View logs
docker logs text-gen-frontend
docker logs text-gen-backend

# Rebuild and restart both containers
docker-compose up --build -d  # If using docker-compose

### Using Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# Build and start all services
docker-compose up --build -d

# View logs of all services
docker-compose logs

# View logs of specific service
docker-compose logs text-gen-frontend
docker-compose logs text-gen-backend

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Development

### Requirements
Backend:
- Python 3.9+
- Flask 3.0.2
- Flask-CORS 4.0.0
- Requests 2.31.0

Frontend:
- Python 3.9+
- Gradio 4.44.1
- Requests 2.31.0

### Local Setup (Without Docker)
```bash
# Backend
cd backend
pip install -r requirements.txt
python server.py

# Frontend (in another terminal)
cd frontend
pip install -r requirements.txt
python app.py
```

### API Endpoints
- POST `/process`
  - Request body: `{"text": "your text", "max_length": 20}`
  - Response: `{"status": "success", "data": {"generated_text": "..."}}`
- GET `/health`
  - Response: `{"status": "healthy", "service": "text-generation-api"}`

### Notes
- Make sure the backend service is running before starting the frontend
- The frontend expects the backend to be available at `http://localhost:8000`
- Default ports: Backend (8000), Frontend (7860)
