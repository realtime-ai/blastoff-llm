"""
Thinking Agent - 使用 Claude Sonnet/Opus + Extended Thinking 实现深度思考

目标：提供高质量、深思熟虑的完整回答
策略：
1. 使用 Sonnet 或 Opus 模型
2. 启用 Extended Thinking 进行深度推理
3. 支持工具调用（搜索、计算等）
4. 异步执行，不阻塞快速响应
"""

import asyncio
import time
from typing import AsyncGenerator, Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
import anthropic


@dataclass
class ThinkingResponse:
    """思考 Agent 的响应结果"""
    content: str
    thinking: Optional[str] = None  # Extended thinking 内容
    latency_ms: float = 0.0
    model: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StreamEvent:
    """流式事件"""
    type: str  # "thinking", "text", "tool_use", "done"
    content: str = ""
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None


class ThinkingAgent:
    """
    深度思考 Agent

    使用 Claude Sonnet/Opus + Extended Thinking 提供高质量回答。
    支持工具调用，异步执行。
    """

    # 可用模型
    MODELS = {
        "sonnet": "claude-sonnet-4-20250514",
        "opus": "claude-opus-4-20250514",
        "sonnet-3.5": "claude-3-5-sonnet-20241022",  # fallback
    }

    # 默认 system prompt
    DEFAULT_SYSTEM_PROMPT = """你是一个专业的 AI 助手。请：
1. 仔细分析用户的问题
2. 提供准确、全面的回答
3. 如果需要，使用可用的工具获取信息
4. 回答应该清晰、结构化

注意：用户可能已经收到了一个简短的初步响应（如"好的"、"让我想想"），
你的任务是提供完整的、有深度的回答来延续那个初步响应。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "sonnet",
        enable_thinking: bool = True,
        thinking_budget: int = 10000,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_handlers: Optional[Dict[str, Callable]] = None,
    ):
        """
        初始化 Thinking Agent

        Args:
            api_key: Anthropic API key
            model: 模型选择 ("sonnet", "opus", "sonnet-3.5")
            enable_thinking: 是否启用 Extended Thinking
            thinking_budget: 思考 token 预算
            tools: 工具定义列表
            tool_handlers: 工具处理函数映射
        """
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = self.MODELS.get(model, self.MODELS["sonnet"])
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.tools = tools or []
        self.tool_handlers = tool_handlers or {}

    def _build_messages(
        self,
        user_message: str,
        prefix: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        构建消息列表

        Args:
            user_message: 当前用户消息
            prefix: 快速响应的前缀（用于续写）
            conversation_history: 对话历史

        Returns:
            消息列表
        """
        messages = []

        # 添加对话历史
        if conversation_history:
            messages.extend(conversation_history)

        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})

        # 如果有前缀，添加为 assistant 消息（用于续写）
        if prefix:
            messages.append({
                "role": "assistant",
                "content": prefix
            })

        return messages

    async def _handle_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any]
    ) -> str:
        """
        处理工具调用

        Args:
            tool_name: 工具名称
            tool_input: 工具输入

        Returns:
            工具执行结果
        """
        if tool_name in self.tool_handlers:
            handler = self.tool_handlers[tool_name]
            if asyncio.iscoroutinefunction(handler):
                return await handler(tool_input)
            else:
                return handler(tool_input)
        else:
            return f"Tool '{tool_name}' not implemented"

    async def respond(
        self,
        user_message: str,
        prefix: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> ThinkingResponse:
        """
        生成完整响应

        Args:
            user_message: 用户消息
            prefix: 快速响应前缀（用于续写）
            conversation_history: 对话历史
            system_prompt: 自定义 system prompt

        Returns:
            ThinkingResponse
        """
        start_time = time.perf_counter()

        messages = self._build_messages(user_message, prefix, conversation_history)
        system = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # 如果有前缀，调整 system prompt
        if prefix:
            system += f"\n\n注意：你已经说了'{prefix}'，请继续你的回答，不要重复这部分内容。"

        try:
            # 构建请求参数
            request_params = {
                "model": self.model,
                "max_tokens": 4096,
                "system": system,
                "messages": messages,
            }

            # 添加工具（如果有）
            if self.tools:
                request_params["tools"] = self.tools

            # 添加 Extended Thinking（如果启用）
            if self.enable_thinking:
                request_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget
                }

            response = await self.client.messages.create(**request_params)

            # 解析响应
            content = ""
            thinking = ""
            tool_calls = []

            for block in response.content:
                if block.type == "thinking":
                    thinking = block.thinking
                elif block.type == "text":
                    content = block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "name": block.name,
                        "input": block.input,
                        "id": block.id
                    })

            latency_ms = (time.perf_counter() - start_time) * 1000

            return ThinkingResponse(
                content=content,
                thinking=thinking,
                latency_ms=latency_ms,
                model=self.model,
                tool_calls=tool_calls
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ThinkingResponse(
                content=f"抱歉，处理请求时出错: {str(e)}",
                latency_ms=latency_ms,
                model=self.model
            )

    async def respond_stream(
        self,
        user_message: str,
        prefix: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        流式生成响应

        Args:
            user_message: 用户消息
            prefix: 快速响应前缀
            conversation_history: 对话历史
            system_prompt: 自定义 system prompt

        Yields:
            StreamEvent 事件
        """
        messages = self._build_messages(user_message, prefix, conversation_history)
        system = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        if prefix:
            system += f"\n\n注意：你已经说了'{prefix}'，请继续你的回答，不要重复这部分内容。"

        try:
            request_params = {
                "model": self.model,
                "max_tokens": 4096,
                "system": system,
                "messages": messages,
            }

            if self.tools:
                request_params["tools"] = self.tools

            if self.enable_thinking:
                request_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget
                }

            async with self.client.messages.stream(**request_params) as stream:
                current_block_type = None

                async for event in stream:
                    if event.type == "content_block_start":
                        current_block_type = event.content_block.type
                        if current_block_type == "tool_use":
                            yield StreamEvent(
                                type="tool_use",
                                tool_name=event.content_block.name
                            )

                    elif event.type == "content_block_delta":
                        if hasattr(event.delta, "thinking"):
                            yield StreamEvent(
                                type="thinking",
                                content=event.delta.thinking
                            )
                        elif hasattr(event.delta, "text"):
                            yield StreamEvent(
                                type="text",
                                content=event.delta.text
                            )
                        elif hasattr(event.delta, "partial_json"):
                            yield StreamEvent(
                                type="tool_input",
                                content=event.delta.partial_json
                            )

                    elif event.type == "content_block_stop":
                        pass

                    elif event.type == "message_stop":
                        yield StreamEvent(type="done")

        except Exception as e:
            yield StreamEvent(
                type="error",
                content=f"Error: {str(e)}"
            )
            yield StreamEvent(type="done")


# 内置工具示例
def create_calculator_tool() -> tuple[Dict[str, Any], Callable]:
    """创建计算器工具"""
    tool_def = {
        "name": "calculator",
        "description": "执行数学计算",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式"
                }
            },
            "required": ["expression"]
        }
    }

    def handler(input_data: Dict[str, Any]) -> str:
        try:
            # 安全的数学计算
            expression = input_data["expression"]
            # 只允许数字和基本运算符
            allowed = set("0123456789+-*/()._ ")
            if not all(c in allowed for c in expression):
                return "Invalid expression"
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {e}"

    return tool_def, handler


def create_search_tool() -> tuple[Dict[str, Any], Callable]:
    """创建搜索工具（模拟）"""
    tool_def = {
        "name": "web_search",
        "description": "搜索互联网获取信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询"
                }
            },
            "required": ["query"]
        }
    }

    async def handler(input_data: Dict[str, Any]) -> str:
        query = input_data["query"]
        # 这里可以集成实际的搜索 API
        return f"Search results for '{query}': [模拟搜索结果]"

    return tool_def, handler


# 测试代码
async def test_thinking_agent():
    """测试 Thinking Agent"""
    # 创建工具
    calc_tool, calc_handler = create_calculator_tool()

    agent = ThinkingAgent(
        model="sonnet",
        enable_thinking=True,
        thinking_budget=5000,
        tools=[calc_tool],
        tool_handlers={"calculator": calc_handler}
    )

    test_messages = [
        "什么是量子计算？",
        "帮我计算 (15 + 27) * 3",
    ]

    print("=" * 50)
    print("Thinking Agent 测试")
    print("=" * 50)

    for msg in test_messages:
        print(f"\n用户: {msg}")
        print("响应（流式）:")

        async for event in agent.respond_stream(msg, prefix="好的，"):
            if event.type == "thinking":
                print(f"[思考] {event.content}", end="", flush=True)
            elif event.type == "text":
                print(event.content, end="", flush=True)
            elif event.type == "done":
                print("\n")


if __name__ == "__main__":
    asyncio.run(test_thinking_agent())
