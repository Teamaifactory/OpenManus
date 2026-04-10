#!/bin/sh
set -e

CONFIG_FILE="config/config.toml"

# Generate config/config.toml from environment variables at container startup.
# Railway stores the Gemini API key under the variable name "Gemini API Key"
# (with spaces). POSIX shells cannot reference variables with spaces via normal
# ${VAR} syntax, so we use `printenv` to retrieve the value safely.
GEMINI_API_KEY="$(printenv 'Gemini API Key' || true)"

if [ -z "$GEMINI_API_KEY" ]; then
  echo "WARNING: 'Gemini API Key' environment variable is not set. The agent will fail to authenticate with the Gemini API."
fi

mkdir -p "$(dirname "$CONFIG_FILE")"

cat > "$CONFIG_FILE" <<EOF
# Auto-generated at container startup from environment variables.
# Do not edit manually — changes will be overwritten on restart.

# Global LLM configuration
[llm]
model       = "gemini-2.0-flash"
base_url    = "https://generativelanguage.googleapis.com/v1beta/openai/"
api_key     = "$GEMINI_API_KEY"
temperature = 0.0
max_tokens  = 8096

# Vision model configuration for image/video analysis
[llm.vision]
model       = "gemini-2.0-flash-exp"
base_url    = "https://generativelanguage.googleapis.com/v1beta/openai/"
api_key     = "$GEMINI_API_KEY"
temperature = 0.0
max_tokens  = 8192

# MCP (Model Context Protocol) configuration
[mcp]
server_reference = "app.mcp.server"

# Run-flow configuration
[runflow]
use_data_analysis_agent = false
EOF

echo "config/config.toml written successfully."

exec python -m uvicorn api:app --host 0.0.0.0 --port 8000
