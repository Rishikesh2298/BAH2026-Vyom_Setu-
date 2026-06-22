"""
Main FastAPI application for AI-Powered Digital Twin of India's Climate
Bharatiya Antariksha Hackathon 2026
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="India Climate Digital Twin API",
    description="AI-Powered Digital Twin of India's Climate System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "India Climate Digital Twin"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AI-Powered Digital Twin of India's Climate",
        "version": "1.0.0",
        "hackathon": "Bharatiya Antariksha 2026",
        "description": "Digital twin framework for climate system simulation and forecasting",
        "endpoints": {
            "climate": "/api/climate",
            "monitoring": "/api/monitoring",
            "data": "/api/data",
            "config": "/api/config"
        }
    }

# Placeholder routes (to be implemented)

@app.get("/api/climate/current")
async def get_current_climate():
    """Get current climate state"""
    return {
        "message": "Current climate state endpoint",
        "status": "pending implementation"
    }

@app.get("/api/climate/forecast")
async def get_forecast():
    """Get weather forecast"""
    return {
        "message": "Weather forecast endpoint",
        "status": "pending implementation"
    }

@app.get("/api/monitoring/monsoon")
async def get_monsoon_data():
    """Get monsoon tracking data"""
    return {
        "message": "Monsoon monitoring endpoint",
        "status": "pending implementation"
    }

@app.get("/api/monitoring/extremes")
async def get_extreme_events():
    """Get extreme weather alerts"""
    return {
        "message": "Extreme weather monitoring endpoint",
        "status": "pending implementation"
    }

@app.get("/api/data/sources")
async def get_data_sources():
    """Get available data sources"""
    return {
        "data_sources": [
            {"name": "INSAT", "type": "satellite", "frequency": "hourly"},
            {"name": "Oceansat", "type": "satellite", "frequency": "daily"},
            {"name": "IMD", "type": "ground_network", "frequency": "daily"},
            {"name": "MERRA-2", "type": "reanalysis", "frequency": "monthly"},
            {"name": "ERA5", "type": "reanalysis", "frequency": "daily"}
        ]
    }

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
