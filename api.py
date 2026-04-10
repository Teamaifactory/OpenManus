from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import os
import logging
import base64
import mimetypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenManus", version="1.0.0")

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global agent state (will be initialized lazily)
_agent = None


async def get_agent():
    """Get or create the Manus agent instance."""
    global _agent
    if _agent is None:
        try:
            from app.agent.manus import Manus
            _agent = await Manus.create()
            logger.info("Manus agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Manus agent: {e}")
            raise
    return _agent


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

@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    files: Optional[List[UploadFile]] = File(default=None),
):
    """
    General-purpose interaction endpoint for OpenManus.

    Accepts a user message and optional file attachments, runs the Manus agent,
    and returns the response in a chat-friendly format.
    """
    try:
        agent = await get_agent()
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": "OpenManus agent is not available. Please check the server configuration.",
                "error": str(e),
            },
        )

    # Build the prompt, appending file context when files are provided
    prompt_parts = [message.strip()]

    if files:
        file_descriptions = []
        for upload in files:
            if not upload.filename:
                continue
            try:
                raw = await upload.read()
                mime = upload.content_type or mimetypes.guess_type(upload.filename)[0] or "application/octet-stream"
                size_kb = len(raw) / 1024

                # For images, embed as base64 so the agent can reference them
                if mime.startswith("image/"):
                    b64 = base64.b64encode(raw).decode("utf-8")
                    file_descriptions.append(
                        f"[Image file: {upload.filename} ({size_kb:.1f} KB), base64: data:{mime};base64,{b64[:80]}... (truncated for prompt)]"
                    )
                else:
                    # For text-based files, try to decode and include content
                    if mime in ("text/plain", "text/csv", "text/markdown") or upload.filename.endswith((".txt", ".md", ".csv")):
                        try:
                            text_content = raw.decode("utf-8", errors="replace")
                            preview = text_content[:2000]
                            file_descriptions.append(
                                f"[Text file: {upload.filename} ({size_kb:.1f} KB)]\n{preview}"
                                + (" ...(truncated)" if len(text_content) > 2000 else "")
                            )
                        except Exception:
                            file_descriptions.append(f"[File: {upload.filename} ({size_kb:.1f} KB, {mime})]")
                    else:
                        file_descriptions.append(f"[File: {upload.filename} ({size_kb:.1f} KB, {mime})]")
            except Exception as fe:
                logger.warning(f"Could not read uploaded file {upload.filename}: {fe}")
                file_descriptions.append(f"[File: {upload.filename} (could not read)]")

        if file_descriptions:
            prompt_parts.append("\n\nAttached files:\n" + "\n\n".join(file_descriptions))

    full_prompt = "\n".join(prompt_parts)

    try:
        logger.info(f"Running Manus agent with prompt: {full_prompt[:200]}...")

        # Reset agent state for a fresh run on each request
        from app.schema import AgentState
        agent.state = AgentState.IDLE
        agent.current_step = 0

        result = await agent.run(full_prompt)

        logger.info("Manus agent completed successfully")
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": message,
                "response": result,
                "files_received": [f.filename for f in (files or []) if f.filename],
            },
        )
    except Exception as e:
        logger.error(f"Agent execution error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": message,
                "response": f"An error occurred while processing your request: {str(e)}",
                "files_received": [f.filename for f in (files or []) if f.filename],
            },
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

