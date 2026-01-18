"""
Orchestrator - 多 Agent 协调器

核心职责：
1. 并行启动 Fast Agent 和 Thinking Agent
2. Fast Agent 响应立即输出（最低延迟）
3. Thinking Agent 响应作为续写追加
4. 合并流式输出为统一的响应流
"""

import asyncio
import time
import json
from typing import AsyncGenerator, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from .fast_agent import FastAgent, FastResponse
from .thinking_agent import ThinkingAgent, StreamEvent


class ResponsePhase(Enum):
    """响应阶段"""
    FAST = "fast"           # 快速响应阶段
    THINKING = "thinking"   # 思考中（可选显示）
    FULL = "full"           # 完整响应阶段
    DONE = "done"           # 完成


@dataclass
class OrchestratorEvent:
    """协调器输出事件"""
    phase: ResponsePhase
    content: str = ""
    is_thinking: bool = False  # 是否是思考内容
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """延迟统计"""
    fast_response_ms: float = 0.0
    thinking_start_ms: float = 0.0
    full_response_ms: float = 0.0
    total_ms: float = 0.0


class MultiAgentOrchestrator:
    """
    多 Agent 协调器

    实现快速响应 + 深度思考的双 Agent 异步架构。

    工作流程：
    1. 用户发送消息
    2. 并行启动 Fast Agent 和 Thinking Agent
    3. Fast Agent 完成后立即输出（<200ms）
    4. Thinking Agent 的输出作为续写追加
    5. 用户获得无缝的响应体验

    ```
    用户输入
        │
        ├──────────────┬──────────────────┐
        ↓              ↓                  │
    Fast Agent    Thinking Agent         │
    (Haiku)       (Sonnet + Thinking)    │
        │              │                  │
        ↓              ↓                  │
    "好的，"    [异步深度思考...]         │
        │              │                  │
        └──────────────┴──────────────────┘
                       ↓
                合并输出流
                       ↓
              "好的，让我来解释一下..."
    ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        thinking_model: str = "sonnet",
        enable_thinking: bool = True,
        thinking_budget: int = 10000,
        show_thinking: bool = False,  # 是否显示思考过程
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_handlers: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化协调器

        Args:
            api_key: Anthropic API key
            thinking_model: 思考 Agent 使用的模型
            enable_thinking: 是否启用 Extended Thinking
            thinking_budget: 思考 token 预算
            show_thinking: 是否在输出中显示思考过程
            tools: 工具定义
            tool_handlers: 工具处理函数
        """
        self.fast_agent = FastAgent(api_key=api_key)
        self.thinking_agent = ThinkingAgent(
            api_key=api_key,
            model=thinking_model,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            tools=tools,
            tool_handlers=tool_handlers,
        )
        self.show_thinking = show_thinking

    async def process(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[OrchestratorEvent, None]:
        """
        处理用户消息，返回流式响应

        Args:
            user_message: 用户消息
            conversation_history: 对话历史
            system_prompt: 自定义 system prompt

        Yields:
            OrchestratorEvent 事件流
        """
        start_time = time.perf_counter()

        # ===== 阶段 1: 快速响应 =====
        # 立即启动 Fast Agent，获取初始响应
        fast_response = await self.fast_agent.respond(user_message)

        # 输出快速响应
        yield OrchestratorEvent(
            phase=ResponsePhase.FAST,
            content=fast_response.content,
            latency_ms=fast_response.latency_ms,
            metadata={
                "model": fast_response.model,
                "type": "fast_response"
            }
        )

        # ===== 阶段 2: 深度思考 + 续写 =====
        # 使用快速响应作为前缀，让 Thinking Agent 续写
        thinking_start = time.perf_counter()

        async for event in self.thinking_agent.respond_stream(
            user_message=user_message,
            prefix=fast_response.content,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
        ):
            current_time = time.perf_counter()

            if event.type == "thinking" and self.show_thinking:
                # 输出思考过程（可选）
                yield OrchestratorEvent(
                    phase=ResponsePhase.THINKING,
                    content=event.content,
                    is_thinking=True,
                    latency_ms=(current_time - thinking_start) * 1000
                )

            elif event.type == "text":
                # 输出正式响应内容
                yield OrchestratorEvent(
                    phase=ResponsePhase.FULL,
                    content=event.content,
                    latency_ms=(current_time - thinking_start) * 1000
                )

            elif event.type == "tool_use":
                # 工具调用事件
                yield OrchestratorEvent(
                    phase=ResponsePhase.FULL,
                    content="",
                    metadata={
                        "type": "tool_use",
                        "tool_name": event.tool_name
                    }
                )

            elif event.type == "done":
                # 完成
                total_time = (time.perf_counter() - start_time) * 1000
                yield OrchestratorEvent(
                    phase=ResponsePhase.DONE,
                    latency_ms=total_time,
                    metadata={
                        "stats": {
                            "fast_response_ms": fast_response.latency_ms,
                            "thinking_time_ms": (time.perf_counter() - thinking_start) * 1000,
                            "total_ms": total_time
                        }
                    }
                )

    async def process_concurrent(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[OrchestratorEvent, None]:
        """
        并发处理：Fast Agent 和 Thinking Agent 同时启动

        这种模式下，两个 Agent 真正并行执行：
        - Fast Agent 完成后立即输出
        - Thinking Agent 在后台继续运行
        - Thinking Agent 完成后追加输出

        Args:
            user_message: 用户消息
            conversation_history: 对话历史
            system_prompt: 自定义 system prompt

        Yields:
            OrchestratorEvent 事件流
        """
        start_time = time.perf_counter()

        # 创建事件队列用于协调输出
        event_queue: asyncio.Queue[OrchestratorEvent] = asyncio.Queue()

        # Fast Agent 任务
        async def fast_task():
            response = await self.fast_agent.respond(user_message)
            await event_queue.put(OrchestratorEvent(
                phase=ResponsePhase.FAST,
                content=response.content,
                latency_ms=response.latency_ms,
                metadata={"model": response.model, "type": "fast_response"}
            ))
            return response

        # Thinking Agent 任务
        async def thinking_task(prefix: str):
            async for event in self.thinking_agent.respond_stream(
                user_message=user_message,
                prefix=prefix,
                conversation_history=conversation_history,
                system_prompt=system_prompt,
            ):
                current_time = (time.perf_counter() - start_time) * 1000

                if event.type == "thinking" and self.show_thinking:
                    await event_queue.put(OrchestratorEvent(
                        phase=ResponsePhase.THINKING,
                        content=event.content,
                        is_thinking=True,
                        latency_ms=current_time
                    ))
                elif event.type == "text":
                    await event_queue.put(OrchestratorEvent(
                        phase=ResponsePhase.FULL,
                        content=event.content,
                        latency_ms=current_time
                    ))
                elif event.type == "done":
                    await event_queue.put(OrchestratorEvent(
                        phase=ResponsePhase.DONE,
                        latency_ms=current_time
                    ))

        # 启动 Fast Agent
        fast_result = await fast_task()

        # 输出 Fast Agent 结果
        fast_event = await event_queue.get()
        yield fast_event

        # 启动 Thinking Agent（使用 Fast Agent 的输出作为前缀）
        thinking_coro = thinking_task(fast_result.content)

        # 处理 Thinking Agent 的流式输出
        async for _ in asyncio.as_completed([asyncio.create_task(thinking_coro)]):
            pass

        # 输出所有排队的事件
        while not event_queue.empty():
            event = await event_queue.get()
            yield event

    def create_openai_compatible_stream(
        self,
        user_message: str,
        model_name: str = "claude-multi-agent",
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        创建 OpenAI 兼容的 SSE 流

        Args:
            user_message: 用户消息
            model_name: 模型名称（用于响应）
            conversation_history: 对话历史

        Yields:
            SSE 格式的字符串
        """
        return self._openai_stream(user_message, model_name, conversation_history)

    async def _openai_stream(
        self,
        user_message: str,
        model_name: str,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> AsyncGenerator[str, None]:
        """内部方法：生成 OpenAI 兼容流"""
        request_id = f"chatcmpl-{int(time.time())}"

        async for event in self.process(user_message, conversation_history):
            if event.phase == ResponsePhase.DONE:
                # 发送完成标记
                yield "data: [DONE]\n\n"
            elif event.content:
                # 发送内容 chunk
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": event.content},
                        "finish_reason": None
                    }]
                }

                # 如果是思考内容，添加标记
                if event.is_thinking:
                    chunk["choices"][0]["delta"]["thinking"] = True

                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# 便捷函数
async def quick_respond(
    user_message: str,
    api_key: Optional[str] = None,
    show_thinking: bool = False,
) -> AsyncGenerator[str, None]:
    """
    便捷函数：快速获取多 Agent 响应

    Args:
        user_message: 用户消息
        api_key: API key
        show_thinking: 是否显示思考过程

    Yields:
        响应文本片段
    """
    orchestrator = MultiAgentOrchestrator(
        api_key=api_key,
        show_thinking=show_thinking
    )

    async for event in orchestrator.process(user_message):
        if event.content:
            yield event.content


# 测试代码
async def test_orchestrator():
    """测试协调器"""
    print("=" * 60)
    print("Multi-Agent Orchestrator 测试")
    print("=" * 60)

    orchestrator = MultiAgentOrchestrator(
        thinking_model="sonnet",
        enable_thinking=True,
        show_thinking=True,  # 显示思考过程
    )

    test_messages = [
        "你好！",
        "解释一下什么是机器学习？",
        "帮我写一个 Python 快速排序算法",
    ]

    for msg in test_messages:
        print(f"\n{'='*40}")
        print(f"用户: {msg}")
        print("-" * 40)

        full_response = ""
        async for event in orchestrator.process(msg):
            if event.phase == ResponsePhase.FAST:
                print(f"[快速响应 {event.latency_ms:.0f}ms] ", end="")
                print(event.content, end="", flush=True)
                full_response += event.content

            elif event.phase == ResponsePhase.THINKING and event.is_thinking:
                # 思考内容用不同颜色或格式显示
                print(f"\n[思考中...] ", end="", flush=True)

            elif event.phase == ResponsePhase.FULL:
                print(event.content, end="", flush=True)
                full_response += event.content

            elif event.phase == ResponsePhase.DONE:
                stats = event.metadata.get("stats", {})
                print(f"\n\n[完成] 总耗时: {stats.get('total_ms', 0):.0f}ms")
                print(f"  - 快速响应: {stats.get('fast_response_ms', 0):.0f}ms")
                print(f"  - 思考时间: {stats.get('thinking_time_ms', 0):.0f}ms")


async def test_openai_compatible():
    """测试 OpenAI 兼容流"""
    print("\n" + "=" * 60)
    print("OpenAI Compatible Stream 测试")
    print("=" * 60)

    orchestrator = MultiAgentOrchestrator()

    print("\n用户: 什么是人工智能？")
    print("-" * 40)

    async for chunk in orchestrator.create_openai_compatible_stream(
        "什么是人工智能？"
    ):
        print(chunk, end="")


if __name__ == "__main__":
    asyncio.run(test_orchestrator())
