"""
FastAPI 服务 - 多 Agent 异步系统的 HTTP 接口

提供 OpenAI 兼容的 API 端点：
- POST /v1/chat/completions - 聊天补全（流式）
- GET /health - 健康检查
- GET /metrics - 性能指标
"""

import os
import re
import time
import json
import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator, Literal
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .orchestrator import MultiAgentOrchestrator, ResponsePhase


# 加载环境变量
load_dotenv()


# ========== 流式模式 ==========

class StreamMode(str, Enum):
    """流式输出模式"""
    TOKEN = "token"       # 每个 token 立即输出（最低延迟）
    SENTENCE = "sentence" # 完整句子输出（适合 TTS）
    PHRASE = "phrase"     # 短语/从句输出（平衡模式）
    WORD = "word"         # 按词输出（中文按字符块）


class SentenceBuffer:
    """
    句子缓冲器 - 用于检测句子/短语边界

    支持中英文混合文本的句子分割。
    """

    # 句子结束标点（完整句子）
    SENTENCE_ENDINGS = set('。！？.!?\n')

    # 短语分隔标点（从句/短语）
    PHRASE_ENDINGS = set('，、；：,;:')

    # 所有分隔符
    ALL_DELIMITERS = SENTENCE_ENDINGS | PHRASE_ENDINGS

    def __init__(self, mode: StreamMode = StreamMode.SENTENCE):
        self.mode = mode
        self.buffer = ""

    def _get_delimiters(self) -> set:
        """根据模式获取分隔符"""
        if self.mode == StreamMode.SENTENCE:
            return self.SENTENCE_ENDINGS
        elif self.mode == StreamMode.PHRASE:
            return self.ALL_DELIMITERS
        else:
            return set()

    def add(self, text: str) -> List[str]:
        """
        添加文本到缓冲区，返回完整的句子/短语列表

        Args:
            text: 新增的文本片段

        Returns:
            完整句子/短语的列表（可能为空）
        """
        if self.mode == StreamMode.TOKEN:
            # Token 模式：直接返回
            return [text] if text else []

        if self.mode == StreamMode.WORD:
            # Word 模式：按空格或中文字符分割
            return self._split_words(text)

        # Sentence/Phrase 模式：检测边界
        self.buffer += text
        return self._extract_complete_units()

    def _extract_complete_units(self) -> List[str]:
        """提取完整的句子/短语"""
        delimiters = self._get_delimiters()
        results = []

        while True:
            # 找到最近的分隔符位置
            min_pos = -1
            for delim in delimiters:
                pos = self.buffer.find(delim)
                if pos != -1 and (min_pos == -1 or pos < min_pos):
                    min_pos = pos

            if min_pos == -1:
                # 没有找到分隔符，保留在缓冲区
                break

            # 提取完整单元（包含分隔符）
            unit = self.buffer[:min_pos + 1].strip()
            if unit:
                results.append(unit)

            # 更新缓冲区
            self.buffer = self.buffer[min_pos + 1:]

        return results

    def _split_words(self, text: str) -> List[str]:
        """按词分割（支持中英文）"""
        self.buffer += text
        results = []

        # 简单的词分割：空格分割英文，连续中文字符作为一个块
        # 使用正则匹配：英文单词 | 中文字符序列 | 标点
        pattern = r'[a-zA-Z]+|[\u4e00-\u9fff]+|[^\s\w]'

        while True:
            match = re.search(pattern, self.buffer)
            if not match:
                break

            # 检查是否有完整的词（后面跟着空格或标点或缓冲区结束）
            end_pos = match.end()
            if end_pos < len(self.buffer):
                next_char = self.buffer[end_pos]
                if next_char.isalnum() or '\u4e00' <= next_char <= '\u9fff':
                    # 词还没结束，等待更多输入
                    break

            word = match.group().strip()
            if word:
                results.append(word + ' ')  # 添加空格分隔

            self.buffer = self.buffer[end_pos:].lstrip()

        return results

    def flush(self) -> str:
        """刷新缓冲区，返回剩余内容"""
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining


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

    # 流式输出模式
    # - token: 每个 token 立即输出（默认，最低延迟）
    # - sentence: 完整句子输出（适合 TTS，按 。！？.!? 分割）
    # - phrase: 短语输出（按 ，、；,; 等分割）
    # - word: 按词输出
    stream_mode: StreamMode = StreamMode.TOKEN


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
    print("=" * 60)
    print("Claude Multi-Agent Server Starting...")
    print("=" * 60)
    print(f"API Key: {'configured' if os.getenv('ANTHROPIC_API_KEY') else 'NOT SET'}")
    print("\nEndpoints:")
    print("  - POST /v1/chat/completions          (OpenAI compatible)")
    print("  - POST /v1/chat/completions/sentences (Sentence mode)")
    print("  - POST /v1/chat/completions/tts      (TTS optimized)")
    print("  - POST /v1/chat/completions/direct   (Direct mode)")
    print("  - GET  /health")
    print("  - GET  /metrics")
    print("  - POST /metrics/reset")
    print("\nStream modes: token | sentence | phrase | word")
    print("=" * 60)
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
    """
    生成流式响应

    支持多种流式模式：
    - token: 每个 token 立即输出
    - sentence: 完整句子输出（适合 TTS）
    - phrase: 短语输出
    - word: 按词输出
    """
    request_id = f"chatcmpl-{int(time.time())}"
    fast_response_ms = 0.0
    total_ms = 0.0

    # 创建句子缓冲器
    buffer = SentenceBuffer(mode=request.stream_mode)

    def make_chunk(content: str, phase: Optional[str] = None, finish: bool = False) -> str:
        """创建 SSE chunk"""
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {} if finish else {"content": content},
                "finish_reason": "stop" if finish else None
            }]
        }
        if phase:
            chunk["phase"] = phase
        # 添加 stream_mode 信息
        chunk["stream_mode"] = request.stream_mode.value
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    try:
        async for event in orchestrator.process(user_message, conversation_history):
            if event.phase == ResponsePhase.FAST:
                fast_response_ms = event.latency_ms

            if event.phase == ResponsePhase.DONE:
                # 刷新缓冲区中剩余的内容
                remaining = buffer.flush()
                if remaining:
                    yield make_chunk(remaining, "full")

                # 记录统计
                total_ms = event.latency_ms
                stats.add_request(fast_response_ms, total_ms)

                # 发送完成 chunk
                yield make_chunk("", finish=True)
                yield "data: [DONE]\n\n"

            elif event.content:
                # 确定当前阶段
                phase = None
                if event.phase == ResponsePhase.FAST:
                    phase = "fast"
                    # Fast 响应总是立即输出（不经过缓冲）
                    yield make_chunk(event.content, phase)
                elif event.phase == ResponsePhase.THINKING:
                    phase = "thinking"
                    # Thinking 内容也立即输出
                    yield make_chunk(event.content, phase)
                elif event.phase == ResponsePhase.FULL:
                    phase = "full"
                    # Full 响应根据 stream_mode 决定输出方式
                    units = buffer.add(event.content)
                    for unit in units:
                        yield make_chunk(unit, phase)

    except Exception as e:
        error_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "error": str(e)
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


async def generate_stream_sentence(
    orchestrator: MultiAgentOrchestrator,
    user_message: str,
    conversation_history: Optional[List[Dict[str, str]]],
    request: ChatCompletionRequest,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    句子模式流式生成器 - 返回结构化数据而非 SSE 字符串

    适合直接集成到其他系统（如 TTS pipeline）
    """
    buffer = SentenceBuffer(mode=request.stream_mode)

    async for event in orchestrator.process(user_message, conversation_history):
        if event.phase == ResponsePhase.DONE:
            remaining = buffer.flush()
            if remaining:
                yield {
                    "type": "sentence",
                    "content": remaining,
                    "phase": "full",
                    "is_final": False
                }
            yield {
                "type": "done",
                "content": "",
                "is_final": True
            }

        elif event.content:
            if event.phase == ResponsePhase.FAST:
                # Fast 响应立即输出
                yield {
                    "type": "fast_response",
                    "content": event.content,
                    "phase": "fast",
                    "is_final": False
                }
            elif event.phase == ResponsePhase.FULL:
                # 通过缓冲器处理
                units = buffer.add(event.content)
                for unit in units:
                    yield {
                        "type": "sentence",
                        "content": unit,
                        "phase": "full",
                        "is_final": False
                    }


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


# ========== TTS 友好的句子流端点 ==========

@app.post("/v1/chat/completions/sentences")
async def chat_completions_sentences(request: ChatCompletionRequest):
    """
    句子模式端点 - 专为 TTS 设计

    返回格式（JSON Lines）：
    {"type": "fast_response", "content": "好的，", "index": 0}
    {"type": "sentence", "content": "让我来解释一下。", "index": 1}
    {"type": "sentence", "content": "机器学习是人工智能的一个分支。", "index": 2}
    {"type": "done", "total_sentences": 3}

    特点：
    - 每个句子作为独立的 JSON 对象
    - 包含句子索引，方便 TTS 排队
    - Fast response 单独标记，可优先处理
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    user_message = request.messages[-1].content
    conversation_history = None
    if len(request.messages) > 1:
        conversation_history = [
            {"role": m.role, "content": m.content}
            for m in request.messages[:-1]
        ]

    orchestrator = MultiAgentOrchestrator(
        api_key=api_key,
        thinking_model=request.thinking_model,
        show_thinking=request.show_thinking,
    )

    # 强制使用句子模式
    request.stream_mode = StreamMode.SENTENCE

    async def generate_sentences():
        sentence_index = 0

        async for item in generate_stream_sentence(
            orchestrator, user_message, conversation_history, request
        ):
            if item["type"] == "done":
                yield json.dumps({
                    "type": "done",
                    "total_sentences": sentence_index
                }, ensure_ascii=False) + "\n"
            else:
                yield json.dumps({
                    "type": item["type"],
                    "content": item["content"],
                    "index": sentence_index,
                    "phase": item.get("phase", "full")
                }, ensure_ascii=False) + "\n"
                sentence_index += 1

    return StreamingResponse(
        generate_sentences(),
        media_type="application/x-ndjson",  # JSON Lines 格式
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/v1/chat/completions/tts")
async def chat_completions_tts(request: ChatCompletionRequest):
    """
    TTS Pipeline 端点 - 返回适合直接送入 TTS 的数据

    返回格式（JSON Lines）：
    {"text": "好的，", "priority": "high", "index": 0}
    {"text": "让我来解释一下。", "priority": "normal", "index": 1}
    {"text": "DONE", "priority": "end", "index": -1}

    特点：
    - Fast response 标记为 high priority（可以立即开始 TTS）
    - 普通句子标记为 normal priority
    - 简化的数据结构，减少解析开销
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    user_message = request.messages[-1].content
    conversation_history = None
    if len(request.messages) > 1:
        conversation_history = [
            {"role": m.role, "content": m.content}
            for m in request.messages[:-1]
        ]

    orchestrator = MultiAgentOrchestrator(
        api_key=api_key,
        thinking_model=request.thinking_model,
        show_thinking=False,  # TTS 不需要思考过程
    )

    request.stream_mode = StreamMode.SENTENCE

    async def generate_tts_stream():
        index = 0

        async for item in generate_stream_sentence(
            orchestrator, user_message, conversation_history, request
        ):
            if item["type"] == "done":
                yield json.dumps({
                    "text": "DONE",
                    "priority": "end",
                    "index": -1
                }, ensure_ascii=False) + "\n"
            elif item["type"] == "fast_response":
                yield json.dumps({
                    "text": item["content"],
                    "priority": "high",
                    "index": index
                }, ensure_ascii=False) + "\n"
                index += 1
            elif item["type"] == "sentence":
                yield json.dumps({
                    "text": item["content"],
                    "priority": "normal",
                    "index": index
                }, ensure_ascii=False) + "\n"
                index += 1

    return StreamingResponse(
        generate_tts_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
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
