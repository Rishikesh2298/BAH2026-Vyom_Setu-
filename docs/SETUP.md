# Setup & Development Guide

## India Climate Digital Twin - Bharatiya Antariksha Hackathon 2026

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Setup](#local-setup)
3. [Docker Setup](#docker-setup)
4. [Development Workflow](#development-workflow)
5. [Testing](#testing)
6. [Deployment](#deployment)

---

## Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows (with WSL2)
- **Python**: 3.9+
- **Node.js**: 16+
- **Docker** (optional): 20.10+
- **Git**: Latest version

### Installation

#### Windows with WSL2

```bash
# Enable WSL2 and Ubuntu
wsl --install -d Ubuntu-22.04

# Inside WSL2 terminal
sudo apt update && sudo apt upgrade -y
sudo apt install python3.11 python3.11-venv python3-pip nodejs npm git
```

#### macOS

```bash
# Using Homebrew
brew install python@3.11 node git
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip nodejs npm git gdal-bin libgdal-dev
```

---

## Local Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd BAH
```

### 2. Backend Setup

#### Create Virtual Environment

```bash
python3.11 -m venv venv

# Activate
# On Linux/macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

#### Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### Environment Configuration

```bash
# Create .env file
cat > .env << EOF
DATABASE_URL=sqlite:///climate_twin.db
REDIS_URL=redis://localhost:6379
API_KEY=dev_key_change_in_production
LOG_LEVEL=INFO
DEBUG=True
EOF
```

#### Run Backend Server

```bash
# Development with hot reload
python backend/app.py

# Or with uvicorn directly
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at: `http://localhost:8000`
API docs: `http://localhost:8000/docs`

### 3. Frontend Setup

#### Install Dependencies

```bash
cd frontend
npm install
```

#### Environment Configuration

```bash
cat > .env << EOF
REACT_APP_API_URL=http://localhost:8000/api
REACT_APP_API_KEY=dev_key_change_in_production
EOF
```

#### Start Development Server

```bash
npm start
```

Dashboard will be available at: `http://localhost:3000`

### 4. Database Setup (Optional)

#### With SQLite (Default for Development)

```bash
# Automatically created on first run
```

#### With PostgreSQL (Production)

```bash
# Install PostgreSQL
# macOS: brew install postgresql
# Ubuntu: sudo apt install postgresql postgresql-contrib

# Start service
# macOS: brew services start postgresql
# Ubuntu: sudo service postgresql start

# Create database
createdb climate_twin

# Create user
psql -d climate_twin -c "CREATE USER climate_user WITH PASSWORD 'your_password';"

# Update .env
echo "DATABASE_URL=postgresql://climate_user:your_password@localhost:5432/climate_twin" >> .env
```

### 5. Redis Cache Setup (Optional)

```bash
# Install Redis
# macOS: brew install redis
# Ubuntu: sudo apt install redis-server

# Start Redis
# macOS: brew services start redis
# Ubuntu: sudo service redis-server start

# Verify
redis-cli ping  # Should return PONG
```

---

## Docker Setup

### Quick Start (All Services)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services will be available at:

- **Backend API**: http://localhost:8000
- **Frontend Dashboard**: http://localhost:3000
- **Nginx Proxy**: http://localhost
- **Jupyter**: http://localhost:8888
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

### Individual Service Setup

```bash
# Build images
docker-compose build

# Start specific service
docker-compose up -d backend
docker-compose up -d frontend

# View service logs
docker-compose logs backend -f

# Execute command in service
docker-compose exec backend bash
```

---

## Development Workflow

### 1. Code Structure

```
backend/
├── app.py                 # Main FastAPI application
├── routes/                # API endpoint implementations
├── models/                # Data models and schemas
└── utils/                 # Utilities and helpers

data_processing/
├── satellite_ingester.py  # Satellite data ingestion
├── imd_ingester.py        # IMD ground network data
└── preprocessor/          # Data preprocessing

ml_pipeline/
├── data_assimilation.py   # EnKF, 3D-Var, Hybrid
├── climate_models.py      # LSTM, CNN, Transformer, GNN
└── training/              # Model training scripts

frontend/
├── src/
│   ├── components/        # Reusable UI components
│   ├── pages/             # Page components
│   ├── services/          # API service calls
│   └── App.js             # Main app component
└── public/                # Static assets
```

### 2. Adding New Features

#### Backend (FastAPI)

```python
# backend/routes/new_feature.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/new_feature", tags=["new_feature"])

class NewFeatureInput(BaseModel):
    param1: str
    param2: int

@router.post("/")
async def create_feature(data: NewFeatureInput):
    """Create new feature"""
    try:
        result = process_feature(data)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

Register in `backend/app.py`:

```python
from routes import new_feature
app.include_router(new_feature.router)
```

#### Frontend (React)

```jsx
// frontend/src/pages/NewFeature.jsx
import React, { useState, useEffect } from "react";
import { getNewFeature } from "../services/api";

function NewFeature() {
  const [data, setData] = useState(null);

  useEffect(() => {
    getNewFeature().then(setData);
  }, []);

  return <div>{/* UI */}</div>;
}

export default NewFeature;
```

### 3. Configuration Management

```bash
# Development
export ENV=development
export LOG_LEVEL=DEBUG

# Testing
export ENV=testing
export LOG_LEVEL=INFO

# Production
export ENV=production
export LOG_LEVEL=WARNING
```

---

## Testing

### Unit Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_backend.py

# Run with coverage
pytest --cov=backend --cov-report=html

# Run only test_data_assimilation
pytest -k "assimilation"
```

### Integration Tests

```bash
# Start test database
docker-compose up -d postgres redis

# Run integration tests
pytest tests/integration/ -v
```

### Frontend Tests

```bash
cd frontend

# Run tests
npm test

# Coverage
npm test -- --coverage
```

---

## Development Tips

### Hot Reload

Backend automatically reloads on file changes (development mode).
Frontend automatically refreshes on file changes.

### Debugging

#### Python Debugging

```python
# backend/app.py
import pdb

@app.get("/debug")
async def debug_endpoint():
    pdb.set_trace()  # Execution will pause here
    return {"message": "Debug"}
```

Run with interactive terminal:

```bash
python -m pdb backend/app.py
```

#### Browser DevTools

Open Firefox/Chrome DevTools (F12) for frontend debugging.

### Logging

```python
import logging
logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

---

## Deployment

### Production Checklist

- [ ] All tests passing
- [ ] Environment variables set
- [ ] Database migrations run
- [ ] Static files collected
- [ ] Error logging configured
- [ ] Secrets properly managed
- [ ] SSL/HTTPS enabled
- [ ] Rate limiting enabled
- [ ] CORS configured correctly
- [ ] API documentation updated

### Docker Deployment

```bash
# Build production image
docker build -t climate-twin:latest -f backend/Dockerfile .

# Push to registry
docker tag climate-twin:latest your-registry/climate-twin:latest
docker push your-registry/climate-twin:latest

# Deploy with compose
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

See [k8s/README.md](../k8s/README.md) for Kubernetes deployment guide.

---

## Troubleshooting

### Backend Issues

**Port already in use**:

```bash
# Find process on port 8000
lsof -i :8000
# Kill process
kill -9 <PID>
```

**Database connection error**:

```bash
# Check PostgreSQL is running
psql -U climate_user -d climate_twin -c "SELECT 1"
```

### Frontend Issues

**npm install fails**:

```bash
# Clear cache and retry
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Docker Issues

**Container fails to start**:

```bash
# Check logs
docker-compose logs backend

# Rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up
```

---

## Additional Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [PostgreSQL Tutorial](https://www.postgresql.org/docs/)
- [Docker Compose Docs](https://docs.docker.com/compose/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

## Support

For issues or questions:

1. Check existing [GitHub Issues](https://github.com/BAH2026/climate-twin/issues)
2. Create new issue with detailed description
3. Contact mentors: mentors@bharatiyaantaraksaha.org
