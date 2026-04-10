from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenManus", version="1.0.0")

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global agent state (will be initialized lazily)
_agent = None

@app.get("/health")
async def health():
    """Health check endpoint - responds immediately"""
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "agent_ready": _agent is not None}
    )

@app.get("/")
async def root():
    """Root endpoint - serve the dashboard"""
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/dashboard")
async def dashboard():
    """Serve the dashboard"""
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/api/status")
async def status():
    """API status endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "service": "OpenManus",
            "status": "online",
            "agent_initialized": _agent is not None,
            "environment": {
                "has_gemini_key": bool(os.getenv("Gemini API Key")),
                "has_wasabi_config": bool(os.getenv("WASABI_BUCKET"))
            }
        }
    )

@app.post("/api/create-video")
async def create_video(prompt: str):
    """Placeholder for video creation endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "pending",
            "message": "Video creation endpoint - coming soon",
            "prompt": prompt
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
