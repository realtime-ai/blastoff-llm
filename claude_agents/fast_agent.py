"""
Fast Agent - 使用 Claude Haiku 实现快速响应

目标：<200ms 的首次响应延迟
策略：
1. 使用最快的 Haiku 模型
2. 限制 max_tokens 到最小
3. 不使用工具调用
4. 专注于生成简短的应答词
"""

import asyncio
import time
from typing import AsyncGenerator, Optional
from dataclasses import dataclass
import anthropic


@dataclass
class FastResponse:
    """快速响应结果"""
    content: str
    latency_ms: float
    model: str


class FastAgent:
    """
    快速响应 Agent

    使用 Claude Haiku 模型实现超低延迟的初始响应。
    适用于语音助手场景，需要在 200ms 内给出反馈。
    """

    # Haiku 模型 ID
    MODEL = "claude-3-5-haiku-20241022"

    # 快速响应的 system prompt
    SYSTEM_PROMPT = """你是一个语音助手的快速响应模块。你的任务是：
1. 在用户说话后立即给出简短的应答词（1-5个字）
2. 表示你已收到并理解用户的请求
3. 暗示你正在处理中

应答词示例：
- 问候：你好！/ 嗨！/ 早上好！
- 问题：好的 / 让我想想 / 嗯
- 请求：没问题 / 好的 / 收到
- 复杂问题：让我查一下 / 稍等

只输出应答词，不要输出其他内容。"""

    # 预定义的快速响应（作为 fallback）
    FALLBACK_RESPONSES = {
        "greeting": ["你好！", "嗨！", "哈喽！"],
        "question": ["让我想想", "好问题", "嗯"],
        "request": ["好的", "没问题", "收到"],
        "thinking": ["让我查一下", "稍等", "思考中"],
        "default": ["好的", "嗯", "收到"]
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 Fast Agent

        Args:
            api_key: Anthropic API key，如果为 None 则从环境变量读取
        """
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    def _categorize_input(self, user_input: str) -> str:
        """
        分类用户输入

        Args:
            user_input: 用户输入文本

        Returns:
            输入类别: greeting, question, request, thinking, default
        """
        user_input_lower = user_input.lower()

        # 问候检测
        greetings = ["你好", "hello", "hi", "嗨", "早上好", "晚上好", "下午好"]
        if any(g in user_input_lower for g in greetings):
            return "greeting"

        # 问题检测
        if "?" in user_input or "？" in user_input or any(
            q in user_input_lower for q in ["什么", "怎么", "为什么", "哪里", "谁", "how", "what", "why"]
        ):
            return "question"

        # 请求检测
        requests = ["帮我", "请", "能不能", "可以", "help", "please"]
        if any(r in user_input_lower for r in requests):
            return "request"

        # 复杂/思考类
        if len(user_input) > 50:
            return "thinking"

        return "default"

    def _get_fallback_response(self, category: str) -> str:
        """获取 fallback 响应"""
        import random
        responses = self.FALLBACK_RESPONSES.get(category, self.FALLBACK_RESPONSES["default"])
        return random.choice(responses)

    async def respond(self, user_message: str) -> FastResponse:
        """
        生成快速响应

        Args:
            user_message: 用户消息

        Returns:
            FastResponse 包含响应内容和延迟
        """
        start_time = time.perf_counter()
        category = self._categorize_input(user_message)

        try:
            response = await self.client.messages.create(
                model=self.MODEL,
                max_tokens=20,  # 限制输出长度
                temperature=0.3,  # 低温度，更确定性
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            content = response.content[0].text.strip()

            # 验证响应长度，太长则使用 fallback
            if len(content) > 20:
                content = self._get_fallback_response(category)

        except Exception as e:
            print(f"[FastAgent] Error: {e}, using fallback")
            content = self._get_fallback_response(category)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return FastResponse(
            content=content,
            latency_ms=latency_ms,
            model=self.MODEL
        )

    async def respond_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        流式生成快速响应

        Args:
            user_message: 用户消息

        Yields:
            响应内容的每个 token
        """
        category = self._categorize_input(user_message)

        try:
            async with self.client.messages.stream(
                model=self.MODEL,
                max_tokens=20,
                temperature=0.3,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            ) as stream:
                content_length = 0
                async for text in stream.text_stream:
                    content_length += len(text)
                    if content_length <= 20:  # 限制总长度
                        yield text
                    else:
                        break

        except Exception as e:
            print(f"[FastAgent] Stream error: {e}, using fallback")
            yield self._get_fallback_response(category)


# 测试代码
async def test_fast_agent():
    """测试 Fast Agent"""
    agent = FastAgent()

    test_messages = [
        "你好！",
        "今天天气怎么样？",
        "帮我写一段 Python 代码",
        "解释一下量子计算的基本原理，以及它和经典计算的区别",
    ]

    print("=" * 50)
    print("Fast Agent 测试")
    print("=" * 50)

    for msg in test_messages:
        print(f"\n用户: {msg}")
        response = await agent.respond(msg)
        print(f"响应: {response.content}")
        print(f"延迟: {response.latency_ms:.2f}ms")


if __name__ == "__main__":
    asyncio.run(test_fast_agent())
