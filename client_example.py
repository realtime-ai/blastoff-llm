#!/usr/bin/env python3
"""
AI Voice Assistant Client Example
Tests the fast response streaming API with voice assistant scenarios
"""

from openai import OpenAI
import time
import requests
import statistics
from typing import List

class VoiceAssistantTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.client = OpenAI(
            api_key="test-key",
            base_url=f"{base_url}/v1"
        )
    
    def reset_metrics(self):
        """Reset server-side metrics"""
        try:
            response = requests.post(f"{self.base_url}/metrics/reset")
            if response.status_code == 200:
                print("✓ Server metrics reset")
            else:
                print("⚠ Failed to reset server metrics")
        except Exception as e:
            print(f"⚠ Could not reset metrics: {e}")
    
    def get_metrics(self):
        """Get server-side metrics"""
        try:
            response = requests.get(f"{self.base_url}/metrics")
            if response.status_code == 200:
                return response.json()
            else:
                print("⚠ Failed to get server metrics")
                return None
        except Exception as e:
            print(f"⚠ Could not get metrics: {e}")
            return None

    def test_voice_response(self, messages: List[dict], mode_name: str, disable_quick: bool = False) -> float:
        """Test voice assistant response and return first response time"""
        start_time = time.time()
        first_chunk_time = None
        
        try:
            extra_body = {}
            if disable_quick:
                extra_body["disable_quick_response"] = True
                
            stream = self.client.chat.completions.create(
                model="voice-assistant",
                messages=messages,
                stream=True,
                max_tokens=150,
                extra_body=extra_body if extra_body else None
            )
            
            print(f"\n🎙️ {mode_name} Response:")
            response_text = ""
            for chunk in stream:
                if first_chunk_time is None and chunk.choices[0].delta.content:
                    first_chunk_time = time.time()
                    first_response_time = first_chunk_time - start_time
                
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    response_text += content
                    
            print()  # New line after response
            
            if first_chunk_time:
                print(f"⏱️ First response time: {first_response_time:.3f}s")
                return first_response_time
            else:
                print("⚠ No response received")
                return 0
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return 0

    def run_voice_assistant_tests(self):
        """Run voice assistant specific tests"""
        print("=" * 60)
        print("🎤 AI VOICE ASSISTANT LATENCY TEST")
        print("=" * 60)
        
        self.reset_metrics()
        
        # Voice assistant test scenarios
        test_scenarios = [
            {
                "name": "Greeting",
                "messages": [{"role": "user", "content": "你好！"}],
                "description": "Basic greeting interaction"
            },
            {
                "name": "Simple Question",
                "messages": [{"role": "user", "content": "今天天气怎么样？"}],
                "description": "Weather inquiry"
            },
            {
                "name": "Request Help",
                "messages": [{"role": "user", "content": "请帮我设个明天上午9点的闹钟"}],
                "description": "Task request"
            },
            {
                "name": "Information Query",
                "messages": [{"role": "user", "content": "什么是人工智能？"}],
                "description": "Knowledge question"
            },
            {
                "name": "Conversation Context",
                "messages": [
                    {"role": "user", "content": "我想学习做菜"},
                    {"role": "assistant", "content": "好的，我可以帮你学习做菜！"},
                    {"role": "user", "content": "推荐一个简单的菜谱"}
                ],
                "description": "Multi-turn conversation"
            }
        ]
        
        quick_times = []
        direct_times = []
        
        for i, scenario in enumerate(test_scenarios):
            print(f"\n📝 Test {i+1}: {scenario['name']}")
            print(f"   Description: {scenario['description']}")
            print(f"   User: {scenario['messages'][-1]['content']}")
            
            # Test with quick response (default mode)
            quick_time = self.test_voice_response(
                scenario['messages'], 
                "Quick Response Mode", 
                disable_quick=False
            )
            if quick_time > 0:
                quick_times.append(quick_time)
            
            print()
            time.sleep(0.5)  # Brief pause between tests
            
            # Test without quick response (direct mode)
            direct_time = self.test_voice_response(
                scenario['messages'], 
                "Direct Mode", 
                disable_quick=True
            )
            if direct_time > 0:
                direct_times.append(direct_time)
            
            print("-" * 50)
        
        # Calculate and display results
        if quick_times and direct_times:
            quick_avg = statistics.mean(quick_times)
            direct_avg = statistics.mean(direct_times)
            improvement = direct_avg - quick_avg
            improvement_pct = (improvement / direct_avg) * 100
            
            print(f"\n📊 VOICE ASSISTANT PERFORMANCE RESULTS:")
            print(f"   Quick Response Mode:")
            print(f"     Average first response: {quick_avg:.3f}s")
            print(f"     Best response: {min(quick_times):.3f}s")
            print(f"     Worst response: {max(quick_times):.3f}s")
            print(f"   Direct Mode:")
            print(f"     Average first response: {direct_avg:.3f}s")
            print(f"     Best response: {min(direct_times):.3f}s")
            print(f"     Worst response: {max(direct_times):.3f}s")
            print(f"   🚀 Improvement: {improvement:.3f}s ({improvement_pct:.1f}% faster)")
            
            # Voice assistant specific analysis
            print(f"\n🎯 VOICE ASSISTANT ANALYSIS:")
            if quick_avg < 0.2:
                print("   ✅ Excellent: Sub-200ms response feels natural")
            elif quick_avg < 0.5:
                print("   ✅ Good: Response time acceptable for voice interaction")
            else:
                print("   ⚠️ Needs improvement: May feel slow for voice interaction")
        
        # Get server metrics
        print(f"\n📈 Server Metrics:")
        metrics = self.get_metrics()
        if metrics:
            quick_stats = metrics.get('quick_response_mode', {})
            direct_stats = metrics.get('direct_mode', {})
            
            if quick_stats.get('total_requests', 0) > 0:
                print(f"   Quick mode requests: {quick_stats['total_requests']}")
                print(f"   Quick mode avg latency: {quick_stats['first_response_latency']['avg']:.3f}s")
            
            if direct_stats.get('total_requests', 0) > 0:
                print(f"   Direct mode requests: {direct_stats['total_requests']}")
                print(f"   Direct mode avg latency: {direct_stats['first_response_latency']['avg']:.3f}s")

def test_basic_interaction():
    """Simple test for basic voice assistant interaction"""
    print("=== Basic Voice Assistant Test ===")
    
    client = OpenAI(
        api_key="test-key",
        base_url="http://localhost:8000/v1"
    )
    
    print("User: 你好，你是谁？")
    
    start_time = time.time()
    stream = client.chat.completions.create(
        model="voice-assistant",
        messages=[{"role": "user", "content": "你好，你是谁？"}],
        stream=True,
        max_tokens=100
    )
    
    print("Assistant: ", end="", flush=True)
    first_chunk = True
    for chunk in stream:
        if chunk.choices[0].delta.content:
            if first_chunk:
                first_response_time = time.time() - start_time
                print(f"[First response: {first_response_time:.3f}s] ", end="")
                first_chunk = False
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print(f"\nTotal time: {time.time() - start_time:.3f}s\n")

if __name__ == "__main__":
    print("🎤 AI Voice Assistant Client Test")
    print("Make sure the server is running on http://localhost:8000")
    print()
    
    try:
        # Basic test first
        test_basic_interaction()
        
        # Comprehensive latency testing
        tester = VoiceAssistantTester()
        tester.run_voice_assistant_tests()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the AI Voice Assistant server is running and accessible.")