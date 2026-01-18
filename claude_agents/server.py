"""
FastAPI 服务 - 多 Agent 异步系统的 HTTP 接口

提供 OpenAI 兼容的 API 端点：
- POST /v1/chat/completions - 聊天补全（流式）
- GET /health - 健康检查
- GET /metrics - 性能指标
"""

import os
import time
import json
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .orchestrator import MultiAgentOrchestrator, ResponsePhase


# 加载环境变量
load_dotenv()


# ========== 数据模型 ==========

class Message(BaseModel):
    """聊天消息"""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """聊天补全请求"""
    model: str = "claude-multi-agent"
    messages: List[Message]
    stream: bool = True
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096

    # 自定义参数
    disable_fast_response: bool = False  # 禁用快速响应
    show_thinking: bool = False  # 显示思考过程
    thinking_model: str = "sonnet"  # 思考模型选择


class UsageInfo(BaseModel):
    """使用统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    """补全选项"""
    index: int = 0
    message: Optional[Message] = None
    delta: Optional[Dict[str, str]] = None
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """聊天补全响应"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[UsageInfo] = None


# ========== 性能统计 ==========

@dataclass
class PerformanceStats:
    """性能统计"""
    fast_response_times: List[float] = field(default_factory=list)
    full_response_times: List[float] = field(default_factory=list)
    total_requests: int = 0

    def add_request(self, fast_ms: float, total_ms: float):
        """添加请求统计"""
        self.fast_response_times.append(fast_ms)
        self.full_response_times.append(total_ms)
        self.total_requests += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取统计数据"""
        def calc_percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(sorted_data) else f
            return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

        return {
            "total_requests": self.total_requests,
            "fast_response": {
                "avg_ms": sum(self.fast_response_times) / len(self.fast_response_times) if self.fast_response_times else 0,
                "p50_ms": calc_percentile(self.fast_response_times, 50),
                "p95_ms": calc_percentile(self.fast_response_times, 95),
                "min_ms": min(self.fast_response_times) if self.fast_response_times else 0,
                "max_ms": max(self.fast_response_times) if self.fast_response_times else 0,
            },
            "full_response": {
                "avg_ms": sum(self.full_response_times) / len(self.full_response_times) if self.full_response_times else 0,
                "p50_ms": calc_percentile(self.full_response_times, 50),
                "p95_ms": calc_percentile(self.full_response_times, 95),
                "min_ms": min(self.full_response_times) if self.full_response_times else 0,
                "max_ms": max(self.full_response_times) if self.full_response_times else 0,
            }
        }

    def reset(self):
        """重置统计"""
        self.fast_response_times.clear()
        self.full_response_times.clear()
        self.total_requests = 0


# 全局统计
stats = PerformanceStats()


# ========== FastAPI 应用 ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("=" * 50)
    print("Claude Multi-Agent Server Starting...")
    print("=" * 50)
    print(f"API Key: {'configured' if os.getenv('ANTHROPIC_API_KEY') else 'NOT SET'}")
    print("Endpoints:")
    print("  - POST /v1/chat/completions")
    print("  - GET  /health")
    print("  - GET  /metrics")
    print("  - POST /metrics/reset")
    print("=" * 50)
    yield
    print("Server shutting down...")


app = FastAPI(
    title="Claude Multi-Agent Server",
    description="基于 Claude Agent SDK 的多 Agent 异步低延迟系统",
    version="0.1.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== API 端点 ==========

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "claude-multi-agent",
        "timestamp": int(time.time())
    }


@app.get("/metrics")
async def get_metrics():
    """获取性能指标"""
    return stats.get_stats()


@app.post("/metrics/reset")
async def reset_metrics():
    """重置性能指标"""
    stats.reset()
    return {"status": "reset", "message": "Metrics have been reset"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    聊天补全端点 - OpenAI 兼容

    支持流式和非流式响应。
    """
    # 验证 API Key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    # 获取用户消息
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    user_message = request.messages[-1].content

    # 构建对话历史
    conversation_history = None
    if len(request.messages) > 1:
        conversation_history = [
            {"role": m.role, "content": m.content}
            for m in request.messages[:-1]
        ]

    # 创建协调器
    orchestrator = MultiAgentOrchestrator(
        api_key=api_key,
        thinking_model=request.thinking_model,
        show_thinking=request.show_thinking,
    )

    if request.stream:
        # 流式响应
        return StreamingResponse(
            generate_stream(orchestrator, user_message, conversation_history, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        # 非流式响应
        return await generate_complete(orchestrator, user_message, conversation_history, request)


async def generate_stream(
    orchestrator: MultiAgentOrchestrator,
    user_message: str,
    conversation_history: Optional[List[Dict[str, str]]],
    request: ChatCompletionRequest,
):
    """生成流式响应"""
    request_id = f"chatcmpl-{int(time.time())}"
    fast_response_ms = 0.0
    total_ms = 0.0

    try:
        async for event in orchestrator.process(user_message, conversation_history):
            if event.phase == ResponsePhase.FAST:
                fast_response_ms = event.latency_ms

            if event.phase == ResponsePhase.DONE:
                # 记录统计
                total_ms = event.latency_ms
                stats.add_request(fast_response_ms, total_ms)

                # 发送完成 chunk
                done_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(done_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            elif event.content:
                # 发送内容 chunk
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": event.content},
                        "finish_reason": None
                    }]
                }

                # 标记响应阶段
                if event.phase == ResponsePhase.FAST:
                    chunk["phase"] = "fast"
                elif event.phase == ResponsePhase.THINKING:
                    chunk["phase"] = "thinking"
                elif event.phase == ResponsePhase.FULL:
                    chunk["phase"] = "full"

                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    except Exception as e:
        error_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "error": str(e)
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


async def generate_complete(
    orchestrator: MultiAgentOrchestrator,
    user_message: str,
    conversation_history: Optional[List[Dict[str, str]]],
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """生成完整响应（非流式）"""
    full_content = ""
    fast_response_ms = 0.0
    total_ms = 0.0

    async for event in orchestrator.process(user_message, conversation_history):
        if event.phase == ResponsePhase.FAST:
            fast_response_ms = event.latency_ms

        if event.content and not event.is_thinking:
            full_content += event.content

        if event.phase == ResponsePhase.DONE:
            total_ms = event.latency_ms
            stats.add_request(fast_response_ms, total_ms)

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=full_content),
                finish_reason="stop"
            )
        ],
        usage=UsageInfo()
    )


# ========== 直接模式端点（用于对比测试）==========

@app.post("/v1/chat/completions/direct")
async def chat_completions_direct(request: ChatCompletionRequest):
    """
    直接模式 - 不使用快速响应

    用于与多 Agent 模式进行性能对比。
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    user_message = request.messages[-1].content
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:
        async def stream_direct():
            request_id = f"chatcmpl-{int(time.time())}"
            try:
                async with client.messages.stream(
                    model="claude-sonnet-4-20250514",
                    max_tokens=request.max_tokens or 4096,
                    messages=messages,
                ) as stream:
                    async for text in stream.text_stream:
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "claude-sonnet-direct",
                            "choices": [{
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            stream_direct(),
            media_type="text/event-stream"
        )
    else:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=request.max_tokens or 4096,
            messages=messages,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model="claude-sonnet-direct",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=response.content[0].text),
                    finish_reason="stop"
                )
            ]
        )


# ========== 启动入口 ==========

def main():
    """启动服务"""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    uvicorn.run(
        "claude_agents.server:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=True,
    )


if __name__ == "__main__":
    main()
