# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py

# Test the system (runs benchmarks)
python client_example.py

# Check server health
curl http://localhost:8000/health

# View performance metrics
curl http://localhost:8000/metrics

# Reset metrics
curl -X POST http://localhost:8000/metrics/reset
```

### Testing a single endpoint
```bash
# Test chat completions endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

## Architecture

This is a dual-model LLM system optimized for ultra-low latency voice interactions:

1. **Dual Model Design**: Uses a small model (Qwen3-8B) for instant responses (<200ms) and a large model (DeepSeek-V3) for comprehensive answers
2. **Single-file Architecture**: All server logic is in `main.py` with the `BlastOffLLM` class orchestrating both models
3. **Streaming Pipeline**: Small model streams immediately, large model seamlessly continues from where small model stops
4. **OpenAI Compatibility**: Implements OpenAI API format at `/v1/chat/completions` endpoint

## Key Implementation Details

- **Prefix Continuation**: The large model receives the small model's response as context to ensure seamless continuation
- **Performance Tracking**: Built-in statistics tracking for latency analysis (P50, P95, averages)
- **Environment Configuration**: Uses `.env` file for model endpoints and API keys (see `.env.example`)
- **Error Handling**: Graceful fallback to direct mode if small model fails
- **Voice Optimization**: System prompts emphasize conversational, voice-friendly responses

## Configuration

Create a `.env` file from `.env.example` with:
- `SMALL_MODEL_URL`: Endpoint for the fast response model
- `LARGE_MODEL_URL`: Endpoint for the comprehensive model
- API keys for both models
- `LARGE_MODEL_SUPPORTS_PREFIX`: Set to "false" for models without prefix support (e.g., OpenAI GPT)
- Server settings (host, port, log level)

## Model Compatibility

The system supports both models with and without prefix capabilities:
- **Models with prefix support** (DeepSeek, Qwen): Use native prefix parameter for seamless continuation
- **Models without prefix support** (OpenAI GPT): Use message history approach where small model response is added as assistant message