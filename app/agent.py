"""
CapableAgent — an LLM agent with three built-in tools:

  • WebBrowser   — Playwright-powered headless browser for web research
  • FileExecutor — Local filesystem read/write for generated content
  • APICaller    — Generic HTTP client for external APIs (e.g. SaladCloud)

The agent exposes these tools to the LLM via OpenAI function-calling so the
model can decide when and how to invoke them.  Each tool returns a structured
JSON response that is fed back into the conversation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling schema)
# ---------------------------------------------------------------------------

_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "WebBrowser",
            "description": (
                "Browse a URL with a headless Chromium browser (Playwright) and "
                "return the visible page text.  Use this for web research, trend "
                "discovery, or reading any public web page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The fully-qualified URL to visit (must start with http:// or https://).",
                    },
                    "extract_goal": {
                        "type": "string",
                        "description": (
                            "Optional plain-English description of what information "
                            "to extract from the page.  If omitted the full visible "
                            "text is returned (up to 8 000 characters)."
                        ),
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "FileExecutor",
            "description": (
                "Read or write a file on the local filesystem.  Use this to persist "
                "generated content (scripts, metadata, prompts) between pipeline stages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write"],
                        "description": "'read' to retrieve file contents; 'write' to save content to a file.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute path to the target file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (required when operation is 'write').",
                    },
                },
                "required": ["operation", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "APICaller",
            "description": (
                "Make an HTTP request to an external API (e.g. SaladCloud GPU workers "
                "for media generation).  Returns the response status code and body."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                        "description": "HTTP method.",
                    },
                    "url": {
                        "type": "string",
                        "description": "Full URL of the API endpoint.",
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers as key-value pairs.",
                    },
                    "body": {
                        "type": "object",
                        "description": "Optional JSON request body (for POST/PUT/PATCH).",
                    },
                },
                "required": ["method", "url"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------


async def _run_web_browser(url: str, extract_goal: Optional[str] = None) -> Dict[str, Any]:
    """Fetch *url* with Playwright and return visible text."""
    try:
        from playwright.async_api import async_playwright  # type: ignore

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                # Extract visible text via innerText on the body element
                text: str = await page.inner_text("body")
            finally:
                await browser.close()

        # Trim to a reasonable size so we don't blow the context window
        max_chars = 8_000
        trimmed = text[:max_chars]
        if len(text) > max_chars:
            trimmed += f"\n\n[… content truncated at {max_chars} characters]"

        result: Dict[str, Any] = {"url": url, "content": trimmed}
        if extract_goal:
            result["extract_goal"] = extract_goal
        return {"status": "success", "data": result}

    except Exception as exc:
        logger.warning("WebBrowser error for %s: %s", url, exc)
        return {"status": "error", "error": str(exc), "url": url}


async def _run_file_executor(
    operation: str, path: str, content: Optional[str] = None
) -> Dict[str, Any]:
    """Read or write a local file."""
    try:
        file_path = Path(path)

        if operation == "read":
            if not file_path.exists():
                return {"status": "error", "error": f"File not found: {path}"}
            text = file_path.read_text(encoding="utf-8")
            return {"status": "success", "path": str(file_path), "content": text}

        elif operation == "write":
            if content is None:
                return {"status": "error", "error": "'content' is required for write operation"}
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return {
                "status": "success",
                "path": str(file_path),
                "bytes_written": len(content.encode("utf-8")),
            }

        else:
            return {"status": "error", "error": f"Unknown operation: {operation}"}

    except Exception as exc:
        logger.warning("FileExecutor error (%s %s): %s", operation, path, exc)
        return {"status": "error", "error": str(exc)}


async def _run_api_caller(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Make an HTTP request and return status + body."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=method.upper(),
                url=url,
                headers=headers or {},
                json=body,
            )
        try:
            resp_body = response.json()
        except Exception:
            resp_body = response.text

        return {
            "status": "success",
            "http_status": response.status_code,
            "body": resp_body,
        }

    except Exception as exc:
        logger.warning("APICaller error (%s %s): %s", method, url, exc)
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

_TOOL_HANDLERS = {
    "WebBrowser": _run_web_browser,
    "FileExecutor": _run_file_executor,
    "APICaller": _run_api_caller,
}


async def _dispatch_tool(name: str, arguments: str) -> str:
    """Parse *arguments* JSON, call the matching handler, return JSON string."""
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return json.dumps({"status": "error", "error": f"Unknown tool: {name}"})

    try:
        kwargs = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return json.dumps({"status": "error", "error": f"Invalid JSON arguments: {exc}"})

    result = await handler(**kwargs)
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CapableAgent
# ---------------------------------------------------------------------------


class CapableAgent:
    """
    LLM agent that can browse the web, read/write files, and call external APIs.

    It extends the same config-loading pattern as SimpleLLMAgent so it works
    transparently as a drop-in replacement in api.py.
    """

    _DEFAULT_SYSTEM_PROMPT = (
        "You are OpenManus, a capable AI assistant with access to three tools:\n"
        "  • WebBrowser   — browse any public URL with a headless Chromium browser\n"
        "  • FileExecutor — read and write local files\n"
        "  • APICaller    — make HTTP requests to external APIs (e.g. SaladCloud)\n\n"
        "Use these tools whenever they would help you answer the user's request. "
        "Think step-by-step, call tools as needed, and provide a clear final answer."
    )

    def __init__(self, config_name: str = "default"):
        cfg = self._load_llm_config(config_name)
        self.model: str = cfg["model"]
        self.base_url: str = cfg["base_url"]
        self.api_key: str = cfg["api_key"]
        self.max_tokens: int = int(cfg.get("max_tokens", 4096))
        self.temperature: float = float(cfg.get("temperature", 0.7))
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    # ------------------------------------------------------------------
    # Config loading (identical to SimpleLLMAgent)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_llm_config(config_name: str) -> dict:
        import tomllib

        project_root = Path(__file__).resolve().parent
        config_path = project_root / "config" / "config.toml"
        if not config_path.exists():
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

    # ------------------------------------------------------------------
    # Agentic chat loop with tool calling
    # ------------------------------------------------------------------

    async def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        Run an agentic loop: send *message* to the LLM, execute any tool calls
        the model requests, feed results back, and return the final text reply.
        """
        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": system_prompt or self._DEFAULT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": message},
        ]

        # Allow up to 10 tool-call rounds before forcing a final answer
        max_rounds = 10
        for round_num in range(max_rounds):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            choice = response.choices[0]
            assistant_msg = choice.message

            # Append the assistant turn (may include tool_calls)
            messages.append(assistant_msg.model_dump(exclude_unset=False))  # type: ignore[arg-type]

            # If no tool calls were requested, we have the final answer
            if not assistant_msg.tool_calls:
                return assistant_msg.content or ""

            # Execute each requested tool call
            for tool_call in assistant_msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                logger.info(
                    "CapableAgent calling tool %s (round %d/%d)",
                    tool_name,
                    round_num + 1,
                    max_rounds,
                )

                tool_result = await _dispatch_tool(tool_name, tool_args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

        # If we exhausted all rounds, ask for a final answer without tools
        logger.warning("CapableAgent reached max tool-call rounds; requesting final answer.")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""
