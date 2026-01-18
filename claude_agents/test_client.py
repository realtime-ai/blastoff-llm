"""
测试客户端 - 用于测试多 Agent 系统的性能

功能：
1. 测试多 Agent 模式 vs 直接模式
2. 测量首次响应延迟
3. 生成性能报告
"""

import asyncio
import time
import json
import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import httpx


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    mode: str  # "multi-agent" or "direct"
    first_chunk_ms: float = 0.0
    total_ms: float = 0.0
    first_chunk_content: str = ""
    full_response: str = ""
    error: Optional[str] = None


@dataclass
class TestReport:
    """测试报告"""
    results: List[TestResult] = field(default_factory=list)

    def add_result(self, result: TestResult):
        self.results.append(result)

    def print_report(self):
        """打印测试报告"""
        print("\n" + "=" * 70)
        print("性能测试报告")
        print("=" * 70)

        # 按测试名称分组
        tests = {}
        for r in self.results:
            if r.test_name not in tests:
                tests[r.test_name] = {}
            tests[r.test_name][r.mode] = r

        print(f"\n{'测试场景':<20} {'模式':<15} {'首次响应':<12} {'总耗时':<12} {'改进':<10}")
        print("-" * 70)

        for test_name, modes in tests.items():
            multi_agent = modes.get("multi-agent")
            direct = modes.get("direct")

            if multi_agent:
                improvement = ""
                if direct and direct.first_chunk_ms > 0:
                    imp_pct = ((direct.first_chunk_ms - multi_agent.first_chunk_ms)
                              / direct.first_chunk_ms * 100)
                    improvement = f"{imp_pct:.0f}% faster"

                print(f"{test_name:<20} {'multi-agent':<15} "
                      f"{multi_agent.first_chunk_ms:>8.0f}ms   "
                      f"{multi_agent.total_ms:>8.0f}ms   "
                      f"{improvement:<10}")

            if direct:
                print(f"{'':<20} {'direct':<15} "
                      f"{direct.first_chunk_ms:>8.0f}ms   "
                      f"{direct.total_ms:>8.0f}ms")

        # 计算平均值
        multi_agent_results = [r for r in self.results if r.mode == "multi-agent"]
        direct_results = [r for r in self.results if r.mode == "direct"]

        if multi_agent_results:
            avg_first = sum(r.first_chunk_ms for r in multi_agent_results) / len(multi_agent_results)
            avg_total = sum(r.total_ms for r in multi_agent_results) / len(multi_agent_results)
            print(f"\n{'[Multi-Agent 平均]':<20} {'':<15} {avg_first:>8.0f}ms   {avg_total:>8.0f}ms")

        if direct_results:
            avg_first = sum(r.first_chunk_ms for r in direct_results) / len(direct_results)
            avg_total = sum(r.total_ms for r in direct_results) / len(direct_results)
            print(f"{'[Direct 平均]':<20} {'':<15} {avg_first:>8.0f}ms   {avg_total:>8.0f}ms")

        print("=" * 70)


class MultiAgentTester:
    """多 Agent 系统测试器"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.report = TestReport()

    async def test_streaming(
        self,
        messages: List[Dict[str, str]],
        test_name: str,
        mode: str = "multi-agent",
        endpoint: str = "/v1/chat/completions",
    ) -> TestResult:
        """
        测试流式响应

        Args:
            messages: 消息列表
            test_name: 测试名称
            mode: 模式名称
            endpoint: API 端点

        Returns:
            TestResult
        """
        result = TestResult(test_name=test_name, mode=mode)

        try:
            start_time = time.perf_counter()
            first_chunk_time = None
            full_response = ""

            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}{endpoint}",
                    json={
                        "model": "claude-multi-agent",
                        "messages": messages,
                        "stream": True,
                    },
                    headers={"Content-Type": "application/json"},
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data)
                                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")

                                if content:
                                    if first_chunk_time is None:
                                        first_chunk_time = time.perf_counter()
                                        result.first_chunk_ms = (first_chunk_time - start_time) * 1000
                                        result.first_chunk_content = content

                                    full_response += content
                                    # 实时打印
                                    print(content, end="", flush=True)

                            except json.JSONDecodeError:
                                pass

            result.total_ms = (time.perf_counter() - start_time) * 1000
            result.full_response = full_response
            print()  # 换行

        except Exception as e:
            result.error = str(e)
            print(f"\n错误: {e}")

        return result

    async def run_comparison_test(
        self,
        user_message: str,
        test_name: str,
    ):
        """
        运行对比测试

        Args:
            user_message: 用户消息
            test_name: 测试名称
        """
        messages = [{"role": "user", "content": user_message}]

        print(f"\n{'='*60}")
        print(f"测试: {test_name}")
        print(f"输入: {user_message}")
        print(f"{'='*60}")

        # 测试 Multi-Agent 模式
        print(f"\n[Multi-Agent 模式]")
        print("-" * 40)
        result_multi = await self.test_streaming(
            messages, test_name, "multi-agent", "/v1/chat/completions"
        )
        self.report.add_result(result_multi)
        print(f"首次响应: {result_multi.first_chunk_ms:.0f}ms | "
              f"总耗时: {result_multi.total_ms:.0f}ms")

        # 等待一下避免限流
        await asyncio.sleep(1)

        # 测试直接模式
        print(f"\n[直接模式]")
        print("-" * 40)
        result_direct = await self.test_streaming(
            messages, test_name, "direct", "/v1/chat/completions/direct"
        )
        self.report.add_result(result_direct)
        print(f"首次响应: {result_direct.first_chunk_ms:.0f}ms | "
              f"总耗时: {result_direct.total_ms:.0f}ms")

        # 计算改进
        if result_direct.first_chunk_ms > 0:
            improvement = ((result_direct.first_chunk_ms - result_multi.first_chunk_ms)
                          / result_direct.first_chunk_ms * 100)
            print(f"\n首次响应改进: {improvement:.0f}%")

    async def run_all_tests(self):
        """运行所有测试"""
        test_cases = [
            ("你好！", "问候"),
            ("今天天气怎么样？", "简单问题"),
            ("帮我解释一下什么是机器学习？", "知识查询"),
            ("请帮我写一个 Python 快速排序算法", "代码请求"),
            ("分析一下人工智能对就业市场的影响", "复杂分析"),
        ]

        print("\n" + "=" * 70)
        print("Claude Multi-Agent 系统性能测试")
        print("=" * 70)

        for user_message, test_name in test_cases:
            await self.run_comparison_test(user_message, test_name)
            await asyncio.sleep(2)  # 避免限流

        # 打印报告
        self.report.print_report()


async def test_single_request(
    message: str = "你好！",
    base_url: str = "http://localhost:8001"
):
    """单次请求测试"""
    tester = MultiAgentTester(base_url)

    print(f"\n测试消息: {message}")
    print("-" * 40)

    result = await tester.test_streaming(
        [{"role": "user", "content": message}],
        "单次测试",
        "multi-agent"
    )

    print(f"\n首次响应: {result.first_chunk_ms:.0f}ms")
    print(f"总耗时: {result.total_ms:.0f}ms")
    print(f"首次内容: {result.first_chunk_content}")


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent 系统测试客户端")
    parser.add_argument("--url", default="http://localhost:8001", help="服务器 URL")
    parser.add_argument("--message", "-m", help="测试消息（单次测试）")
    parser.add_argument("--full", "-f", action="store_true", help="运行完整测试套件")
    args = parser.parse_args()

    if args.message:
        await test_single_request(args.message, args.url)
    elif args.full:
        tester = MultiAgentTester(args.url)
        await tester.run_all_tests()
    else:
        # 默认运行单次测试
        await test_single_request("你好！解释一下什么是量子计算？", args.url)


if __name__ == "__main__":
    asyncio.run(main())
