import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import time
from openai import AsyncOpenAI
import uvicorn
import logging
from dataclasses import dataclass, field
import statistics
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BlastOff LLM",
              description="Fast LLM Response with Small Model Prefix")


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    stream: bool = True  # Always streaming for voice assistant
    stop: Optional[Union[str, List[str]]] = None
    # Custom parameter to disable quick response for comparison
    disable_quick_response: Optional[bool] = False


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LatencyStats:
    first_response_times: List[float] = field(default_factory=list)
    total_response_times: List[float] = field(default_factory=list)
    quick_response_times: List[float] = field(default_factory=list)
    large_model_times: List[float] = field(default_factory=list)
    request_count: int = 0

    def add_request(self, first_response_time: float, total_time: float,
                    quick_time: float = None, large_time: float = None):
        self.first_response_times.append(first_response_time)
        self.total_response_times.append(total_time)
        if quick_time is not None:
            self.quick_response_times.append(quick_time)
        if large_time is not None:
            self.large_model_times.append(large_time)
        self.request_count += 1

    def get_stats(self):
        def safe_stats(data):
            if not data:
                return {"avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0}
            return {
                "avg": statistics.mean(data),
                "min": min(data),
                "max": max(data),
                "p50": statistics.median(data),
                "p95": sorted(data)[int(len(data) * 0.95)] if len(data) > 1 else data[0]
            }

        return {
            "total_requests": self.request_count,
            "first_response_latency": safe_stats(self.first_response_times),
            "total_response_latency": safe_stats(self.total_response_times),
            "quick_response_latency": safe_stats(self.quick_response_times),
            "large_model_latency": safe_stats(self.large_model_times)
        }


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class BlastOffLLM:
    def __init__(self):
        # Statistics tracking
        self.stats = LatencyStats()
        # For comparison without quick response
        self.direct_mode_stats = LatencyStats()
        # Configuration for different model types
        self.small_model_config = {
            "api_key": os.getenv("SMALL_MODEL_API_KEY", "your-small-model-api-key"),
            "base_url": os.getenv("SMALL_MODEL_BASE_URL", "https://api.siliconflow.cn/v1"),
            "model": os.getenv("SMALL_MODEL_NAME", "Qwen/Qwen2-1.5B-Instruct")
        }

        self.large_model_config = {
            "api_key": os.getenv("LARGE_MODEL_API_KEY", "your-large-model-api-key"),
            "base_url": os.getenv("LARGE_MODEL_BASE_URL", "https://api.siliconflow.cn/v1"),
            "model": os.getenv("LARGE_MODEL_NAME", "deepseek-ai/DeepSeek-V2.5"),
            "supports_prefix": os.getenv("LARGE_MODEL_SUPPORTS_PREFIX", "true").lower() == "true"
        }

        # Initialize clients
        self.small_client = AsyncOpenAI(
            api_key=self.small_model_config["api_key"],
            base_url=self.small_model_config["base_url"]
        )

        self.large_client = AsyncOpenAI(
            api_key=self.large_model_config["api_key"],
            base_url=self.large_model_config["base_url"]
        )

        # Predefined quick responses for AI voice assistant scenarios
        self.quick_responses = {
            "greeting": ["你好！", "嗨！", "您好，", "我在这里，"],
            "question": ["让我想想，", "这个问题，", "关于这个，", "我来解答，"],
            "request": ["好的，", "明白了，", "我来帮您，", "让我来，"],
            "thinking": ["嗯，", "我觉得，", "让我分析，", "根据情况，"]
        }

    def _categorize_request(self, messages: List[Message]) -> str:
        """Categorize the request for AI voice assistant context"""
        last_message = messages[-1].content.lower()

        # Greeting patterns
        if any(word in last_message for word in ["你好", "hello", "hi", "嗨"]):
            return "greeting"
        # Question patterns
        elif any(word in last_message for word in ["什么", "如何", "怎么", "为什么", "why", "how", "what", "?", "？"]):
            return "question"
        # Request patterns
        elif any(word in last_message for word in ["请", "帮我", "能不能", "可以", "help", "please"]):
            return "request"
        else:
            return "thinking"

    async def _get_quick_response(self, messages: List[Message]) -> tuple[str, float]:
        """Generate quick response using small model or predefined responses"""
        start_time = time.time()
        try:
            # Try to get a contextual quick response from small model for voice assistant
            quick_prompt = [
                {"role": "system", "content": "你是一个AI语音助手。请用1-3个字的简短语气词回应用户，比如：'你好！'、'好的，'、'嗯，'、'让我想想，'，要自然像真人对话。只输出语气词，不要完整回答。"},
                {"role": "user", "content": messages[-1].content}
            ]

            response = await self.small_client.chat.completions.create(
                model=self.small_model_config["model"],
                messages=quick_prompt,
                max_tokens=10,
                temperature=0.3,
                extra_body={"enable_thinking": False}
            )

            quick_text = response.choices[0].message.content.strip()

            # Fallback to predefined responses if small model response is too long or inappropriate
            if len(quick_text) > 6 or not self._is_appropriate_quick_response(quick_text):
                category = self._categorize_request(messages)
                import random
                quick_text = random.choice(self.quick_responses[category])

            elapsed_time = time.time() - start_time
            return quick_text, elapsed_time

        except Exception as e:
            logger.warning(f"Small model failed, using fallback: {e}")
            # Fallback to predefined quick responses
            category = self._categorize_request(messages)
            import random
            elapsed_time = time.time() - start_time
            return random.choice(self.quick_responses[category]), elapsed_time

    def _is_appropriate_quick_response(self, text: str) -> bool:
        """Check if the quick response is appropriate for voice assistant"""
        # Should be short and conversational
        if len(text) > 6:
            return False
        # Should not contain complex sentences
        if any(char in text for char in ["。", "！", "？", "，"] if text.count(char) > 1):
            return False
        return True

    async def _get_full_response(self, messages: List[Message], prefix: str = None, **kwargs) -> tuple[str, float]:
        """Generate full response for AI voice assistant with optional prefix"""
        start_time = time.time()
        try:
            # Enhance messages with voice assistant context
            enhanced_messages = [
                {"role": "system", "content": "你是一个友好的AI语音助手，用自然对话的方式回应用户。回答要简洁明了，适合语音交互。"}
            ] + [msg.dict() for msg in messages]

            extra_body = {
                "enable_thinking": False
            }
            
            # Handle prefix based on model capability
            if prefix:
                if self.large_model_config.get("supports_prefix", True):
                    # Use native prefix feature if supported
                    extra_body["prefix"] = prefix
                else:
                    # Fallback: Add prefix as assistant message for continuation
                    enhanced_messages.append({"role": "assistant", "content": prefix})
                    # Add a hint in system message for smooth continuation
                    enhanced_messages[0]["content"] += f"\n继续前面的回答，前面已经说了：'{prefix}'，请自然地继续补充完整。"

            response = await self.large_client.chat.completions.create(
                model=self.large_model_config["model"],
                messages=enhanced_messages,
                extra_body=extra_body if extra_body else None,
                max_tokens=kwargs.get("max_tokens", 150),  # Shorter for voice
                temperature=kwargs.get("temperature", 0.7),
                stop=kwargs.get("stop")
            )

            elapsed_time = time.time() - start_time
            return response.choices[0].message.content, elapsed_time

        except Exception as e:
            logger.error(f"Large model failed: {e}")
            raise HTTPException(
                status_code=500, detail="Large model unavailable")

    async def _stream_response(self, messages: List[Message], disable_quick: bool = False, **kwargs) -> AsyncGenerator[str, None]:
        """Stream response for AI voice assistant with optional quick prefix"""
        request_start_time = time.time()
        first_chunk_sent = False
        quick_response = ""
        quick_time = 0

        if not disable_quick:
            # Step 1: Get quick voice assistant response
            quick_response, quick_time = await self._get_quick_response(messages)

            # Stream the quick response first
            chunk_data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": kwargs.get("model", "voice-assistant"),
                "choices": [{
                    "index": 0,
                    "delta": {"content": quick_response},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
            first_chunk_sent = True

        # Step 2: Get full response with voice assistant context
        large_model_start = time.time()
        try:
            # Enhance messages for voice assistant context
            enhanced_messages = [
                {"role": "system", "content": "你是一个友好的AI语音助手，用自然对话的方式回应用户。回答要简洁明了，适合语音交互。"}
            ] + [msg.dict() for msg in messages]

            extra_body = {}
            
            # Handle prefix based on model capability
            if quick_response and not disable_quick:
                if self.large_model_config.get("supports_prefix", True):
                    # Use native prefix feature if supported
                    extra_body["prefix"] = quick_response
                else:
                    # Fallback: Add prefix as assistant message for continuation
                    enhanced_messages.append({"role": "assistant", "content": quick_response})
                    # Add a hint in system message for smooth continuation
                    enhanced_messages[0]["content"] += f"\n继续前面的回答，前面已经说了：'{quick_response}'，请自然地继续补充完整。"

            response = await self.large_client.chat.completions.create(
                model=self.large_model_config["model"],
                messages=enhanced_messages,
                extra_body=extra_body if extra_body else None,
                max_tokens=kwargs.get("max_tokens", 150),  # Shorter for voice
                temperature=kwargs.get("temperature", 0.7),
                stop=kwargs.get("stop"),
                stream=True
            )

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    if not first_chunk_sent:
                        # Record first response time for direct mode
                        first_response_time = time.time() - request_start_time

                    chunk_data = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": kwargs.get("model", "voice-assistant"),
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk.choices[0].delta.content},
                            "finish_reason": chunk.choices[0].finish_reason
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    first_chunk_sent = True

                if chunk.choices[0].finish_reason:
                    # Record statistics
                    total_time = time.time() - request_start_time
                    large_model_time = time.time() - large_model_start

                    if disable_quick:
                        first_response_time = time.time() - request_start_time
                        self.direct_mode_stats.add_request(
                            first_response_time, total_time, large_time=large_model_time)
                    else:
                        first_response_time = quick_time
                        self.stats.add_request(
                            first_response_time, total_time, quick_time, large_model_time)
                    break

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            # Send error chunk
            error_chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": kwargs.get("model", "voice-assistant"),
                "choices": [{
                    "index": 0,
                    "delta": {"content": " [抱歉，出现了问题]"},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

        # End of stream
        yield "data: [DONE]\n\n"

    async def chat_completion(self, request: ChatCompletionRequest) -> StreamingResponse:
        """Voice assistant chat completion endpoint (streaming only)"""
        disable_quick = getattr(request, 'disable_quick_response', False)

        return StreamingResponse(
            self._stream_response(
                request.messages,
                disable_quick=disable_quick,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop
            ),
            media_type="text/plain"
        )


# Initialize the BlastOff LLM instance
blastoff_llm = BlastOffLLM()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    try:
        return await blastoff_llm.chat_completion(request)
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "AI Voice Assistant - Fast Response API", "mode": "streaming_only"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/metrics")
async def get_metrics():
    """Get latency statistics and performance metrics"""
    return {
        "quick_response_mode": blastoff_llm.stats.get_stats(),
        "direct_mode": blastoff_llm.direct_mode_stats.get_stats(),
        "comparison": {
            "quick_mode_requests": blastoff_llm.stats.request_count,
            "direct_mode_requests": blastoff_llm.direct_mode_stats.request_count,
            "avg_first_response_improvement": (
                blastoff_llm.direct_mode_stats.get_stats()["first_response_latency"]["avg"] -
                blastoff_llm.stats.get_stats()["first_response_latency"]["avg"]
                if blastoff_llm.stats.request_count > 0 and blastoff_llm.direct_mode_stats.request_count > 0
                else 0
            )
        }
    }


@app.post("/metrics/reset")
async def reset_metrics():
    """Reset all statistics"""
    blastoff_llm.stats = LatencyStats()
    blastoff_llm.direct_mode_stats = LatencyStats()
    return {"message": "Metrics reset successfully"}

if __name__ == "__main__":
    # Get server configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    uvicorn.run(app, host=host, port=port, log_level=log_level)
