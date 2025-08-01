



### 小模型

model = Qwen/Qwen3-8B

extra_body = {
    enable_thinking: false
}



### 前缀续写
​
1. 使用场景
前缀续写中，用户提供希望输出的前缀信息，来让模型基于用户提供的前缀信息来补全其余的内容。 基于上述能力，模型能有更好的指令遵循能力，满足用户一些特定场景的指定格式的问题。
​
2. 使用方式
在请求中添加

extra_body={"prefix":"希望的前缀内容"}
​
3. 支持模型列表
目前大语言类模型支持上述参数。
注意：支持的模型情况可能会发生变化，请查阅本文档了解最新支持的模型列表。
​
4. 使用示例
下面是基于 OpenAI 库使用前缀续写的例子：


client = OpenAI(
    api_key="您的 APIKEY", # 从https://cloud.siliconflow.cn/account/ak获取
    base_url="https://api.siliconflow.cn/v1"
)
 
messages = [
    {"role": "user", "content": "Please write quick sort code"},
]

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V2.5",
    messages=messages,
    extra_body={"prefix":"```python\n"}
)

print(response.choices[0].message.content)