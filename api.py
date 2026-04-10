from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import os
import logging
import base64
import mimetypes
import tomllib
from pathlib import Path

from openai import AsyncOpenAI

from app.agent import CapableAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenManus", version="1.0.0")

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ---------------------------------------------------------------------------
# SimpleLLMAgent — lightweight LLM caller that reads config/config.toml
# directly, avoiding all optional heavy dependencies (boto3, browser-use, …).
# ---------------------------------------------------------------------------

class SimpleLLMAgent:
    """
    Minimal agent that calls an LLM via the OpenAI-compatible SDK.

    It reads the [llm] (and optional [llm.<name>]) sections from
    config/config.toml so it honours whatever model/key entrypoint.sh wrote.
    """

    _DEFAULT_SYSTEM_PROMPT = (
        "You are OpenManus, a helpful AI assistant. "
        "Answer the user's questions clearly and concisely."
    )

    def __init__(self, config_name: str = "default"):
        cfg = self._load_llm_config(config_name)
        self.model: str = cfg["model"]
        self.base_url: str = cfg["base_url"]
        self.api_key: str = cfg["api_key"]
        self.max_tokens: int = int(cfg.get("max_tokens", 4096))
        self.temperature: float = float(cfg.get("temperature", 0.7))
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _load_llm_config(config_name: str) -> dict:
        """
        Load LLM settings from config/config.toml.

        The TOML structure produced by entrypoint.sh is:
            [llm]           → base / default settings
            [llm.vision]    → vision override
            [llm.filter]    → Claude filter override
            [llm.engagement]→ engagement override

        For config_name == "default" we use the top-level [llm] table.
        For any other name we merge [llm] with [llm.<name>].
        """
        project_root = Path(__file__).resolve().parent
        config_path = project_root / "config" / "config.toml"
        if not config_path.exists():
            # Fall back to the example file so the service can at least start
            config_path = project_root / "config" / "config.example.toml"
        if not config_path.exists():
            raise FileNotFoundError(
                "No config/config.toml found. "
                "Make sure entrypoint.sh has run before starting the server."
            )

        with config_path.open("rb") as fh:
            raw = tomllib.load(fh)

        base = {k: v for k, v in raw.get("llm", {}).items() if not isinstance(v, dict)}

        if config_name == "default":
            return base

        override = raw.get("llm", {}).get(config_name, {})
        return {**base, **override}

    async def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """Send *message* to the configured LLM and return the text reply."""
        messages = [
            {
                "role": "system",
                "content": system_prompt or self._DEFAULT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": message},
        ]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""


# Module-level singleton — created once on first request
_capable_agent: Optional[CapableAgent] = None


def get_capable_agent() -> CapableAgent:
    """Return (and lazily create) the module-level CapableAgent."""
    global _capable_agent
    if _capable_agent is None:
        _capable_agent = CapableAgent(config_name="default")
        logger.info(
            "CapableAgent initialised (model=%s, base_url=%s)",
            _capable_agent.model,
            _capable_agent.base_url,
        )
    return _capable_agent


@app.get("/health")
async def health():
    """Health check endpoint - responds immediately"""
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "agent_ready": _capable_agent is not None}
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
            "agent_initialized": _capable_agent is not None,
            "environment": {
                "has_gemini_key": bool(os.getenv("GEMINI_API_KEY")),
                "has_claude_key": bool(os.getenv("CLAUDE_API_KEY")),
            }
        }
    )

@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    files: Optional[List[UploadFile]] = File(default=None),
):
    """
    Chat endpoint for OpenManus.

    Accepts a user message and optional file attachments, calls the configured
    LLM directly via SimpleLLMAgent, and returns the response.
    """
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

                # For images, embed as base64 so the LLM can reference them
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
        agent = get_capable_agent()
        logger.info("Running CapableAgent with prompt: %s...", full_prompt[:200])

        result = await agent.chat(full_prompt)

        logger.info("CapableAgent completed successfully")
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
        logger.error("Agent execution error: %s", e, exc_info=True)
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

