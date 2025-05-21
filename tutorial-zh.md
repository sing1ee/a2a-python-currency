
# 使用 A2A Python SDK 实现 CurrencyAgent
谷歌官方的[a2a-python](https://github.com/google/a2a-python) SDK最近频繁的更新，我们的教程也需要跟着更新，这篇文章，我们通过 a2a-python sdk的 0.2.3 版本，实现一个简单的CurrencyAgent。

## 源码
项目的源码在[]，欢迎 star 。


## 准备
- uv 0.7.2，用来进行项目管理
- Python 3.13+，一定要这个版本以上，a2a-python 的要求
- openai/openrouter 的 apiKey，baseURL，我使用的是 [OpenRouter](https://openrouter.ai/)，有更多的模型可以选择。

详细过程
## 创建项目：
```bash
uv init a2a-python-currency
cd a2a-python-currency
```
## 创建虚拟环境
```bash
uv venv
source .venv/bin/activate
```
## 添加依赖
```bash
uv add a2a-python
uv add dotenv
```
## 配置环境变量
```bash
echo OPENROUTER_API_KEY=your_api_key >> .env
echo OPENROUTER_BASE_URL=your_base_url >> .env

# example 
OPENROUTER_API_KEY=你的OpenRouter API密钥
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```
## 创建 Agent
完整的代码如下：
```python
import logging
import json
from typing import Any, Dict, List, Optional
import httpx
from os import getenv
from dotenv import load_dotenv
from collections.abc import AsyncIterable

load_dotenv()

logger = logging.getLogger(__name__)

class CurrencyAgent:
    """Currency Conversion Agent using OpenAI API."""

    SYSTEM_PROMPT = """You are a specialized assistant for currency conversions.
Your sole purpose is to use the 'get_exchange_rate' tool to answer questions about currency exchange rates.
If the user asks about anything other than currency conversion or exchange rates,
politely state that you cannot help with that topic and can only assist with currency-related queries.
Do not attempt to answer unrelated questions or use tools for other purposes.

You have access to the following tool:
- get_exchange_rate: Get current exchange rate between two currencies

When using the tool, respond in the following JSON format:
{
    "status": "completed" | "input_required" | "error",
    "message": "your response message"
}

If you need to use the tool, respond with:
{
    "status": "tool_use",
    "tool": "get_exchange_rate",
    "parameters": {
        "currency_from": "USD",
        "currency_to": "EUR",
        "currency_date": "latest"
    }
}
Note: Return the response in the JSON format, only json is allowed.
"""

    def __init__(self):
        self.api_key = getenv("OPENROUTER_API_KEY")
        self.api_base = getenv("OPENROUTER_BASE_URL")
        self.model = "anthropic/claude-3.7-sonnet"
        self.conversation_history: List[Dict[str, str]] = []

    async def get_exchange_rate(
        self,
        currency_from: str = 'USD',
        currency_to: str = 'EUR',
        currency_date: str = 'latest',
    ) -> Dict[str, Any]:
        """Get current exchange rate between currencies."""
        try:
            response = httpx.get(
                f'https://api.frankfurter.app/{currency_date}',
                params={'from': currency_from, 'to': currency_to},
            )
            response.raise_for_status()
            data = response.json()
            if 'rates' not in data:
                logger.error(f'rates not found in response: {data}')
                return {'error': 'Invalid API response format.'}
            logger.info(f'API response: {data}')
            return data
        except httpx.HTTPError as e:
            logger.error(f'API request failed: {e}')
            return {'error': f'API request failed: {e}'}
        except ValueError:
            logger.error('Invalid JSON response from API')
            return {'error': 'Invalid JSON response from API.'}

    async def _call_openai(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call OpenAI API through OpenRouter."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()

    async def stream(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
        """Stream the response for a given query."""
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": query})

        # Prepare messages for API call
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + self.conversation_history

        # Get response from OpenAI
        response = await self._call_openai(messages)
        assistant_message = response["choices"][0]["message"]["content"]
        print(assistant_message)
        try:
            # Try to parse the response as JSON
            parsed_response = json.loads(assistant_message)
            
            # If it's a tool use request
            if parsed_response.get("status") == "tool_use":
                tool_name = parsed_response["tool"]
                parameters = parsed_response["parameters"]
                
                # Yield tool usage status
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Looking up the exchange rates..."
                }
                
                if tool_name == "get_exchange_rate":
                    # Yield processing status
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": "Processing the exchange rates..."
                    }
                    
                    tool_result = await self.get_exchange_rate(**parameters)
                    
                    # Add tool result to conversation history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": json.dumps({"tool_result": tool_result})
                    })
                    
                    # Get final response after tool use
                    final_response = await self._call_openai(messages)
                    final_message = final_response["choices"][0]["message"]["content"]
                    parsed_response = json.loads(final_message)

            # Add assistant response to conversation history
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Yield final response
            if parsed_response["status"] == "completed":
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": parsed_response["message"]
                }
            elif parsed_response["status"] in ["input_required", "error"]:
                yield {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": parsed_response["message"]
                }
            else:
                yield {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": "We are unable to process your request at the moment. Please try again."
                }

        except json.JSONDecodeError:
            # If response is not valid JSON, return error
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "Invalid response format from the model."
            } 
```

其主要功能和实现逻辑：

### 1. 核心功能
- 专门处理货币转换和汇率查询相关的请求
- 使用 Frankfurter API 获取实时汇率数据
- 通过 OpenRouter 调用 Claude 3.7 Sonnet 模型进行对话处理

### 2. 系统架构
Agent 主要由以下几个部分组成：

#### 2.1 系统提示词（System Prompt）
- 定义了 Agent 的专门用途：仅处理货币转换相关查询
- 规定了响应格式：必须使用 JSON 格式
- 定义了工具使用方式：通过 `get_exchange_rate` 工具获取汇率信息

#### 2.2 主要方法
1. **初始化方法 `__init__`**
   - 配置 API 密钥和基础 URL
   - 初始化对话历史记录

2. **汇率查询方法 `get_exchange_rate`**
   - 参数：源货币、目标货币、日期（默认为最新）
   - 调用 Frankfurter API 获取汇率数据
   - 返回 JSON 格式的汇率信息

3. **流式处理方法 `stream`**
   - 提供流式响应功能
   - 实时返回处理状态和结果
   - 支持工具调用的中间状态反馈

### 3. 工作流程
1. **接收用户查询**
   - 将用户消息添加到对话历史

2. **模型处理**
   - 将系统提示词和对话历史发送给模型
   - 模型分析是否需要使用工具

3. **工具调用（如需要）**
   - 如果模型决定使用工具，会返回工具调用请求
   - 执行汇率查询
   - 将查询结果添加到对话历史

4. **生成最终响应**
   - 基于工具调用结果生成最终回答
   - 返回格式化的 JSON 响应

### 4. 响应格式
Agent 的响应始终采用 JSON 格式，包含以下状态：
- `completed`：任务完成
- `input_required`：需要用户输入
- `error`：发生错误
- `tool_use`：需要使用工具

### 5. 错误处理
- 包含完整的错误处理机制
- 处理 API 调用失败
- 处理 JSON 解析错误
- 处理无效响应格式


## 测试 Agent
编写测试的代码：
```python
import asyncio
import logging
from currency_agent import CurrencyAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    agent = CurrencyAgent()
    
    # Test cases
    test_queries = [
        "What is the current exchange rate from USD to EUR?",
        "Can you tell me the exchange rate between GBP and JPY?",
        "What's the weather like today?",  # This should be rejected as it's not currency-related
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: {query}")
        async for response in agent.stream(query, "test_session"):
            logger.info(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main()) 
```
如果一切正常，尤其别忘了 environment 配置正确，你应该可以看到类似下面的输出：
```bash
uv run python test_currency_agent.py
INFO:__main__:
Testing query: What is the current exchange rate from USD to EUR?
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
INFO:__main__:Response: {'is_task_complete': False, 'require_user_input': False, 'content': 'Looking up the exchange rates...'}
INFO:__main__:Response: {'is_task_complete': False, 'require_user_input': False, 'content': 'Processing the exchange rates...'}
INFO:httpx:HTTP Request: GET https://api.frankfurter.app/latest?from=USD&to=EUR "HTTP/1.1 200 OK"
INFO:currency_agent:API response: {'amount': 1.0, 'base': 'USD', 'date': '2025-05-20', 'rates': {'EUR': 0.8896}}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
INFO:currency_agent:Final message: {'role': 'assistant', 'content': '{\n    "status": "completed",\n    "message": "The current exchange rate from USD to EUR is 0.8896. This means that 1 US Dollar equals 0.8896 Euros as of May 20, 2025."\n}', 'refusal': None, 'reasoning': None}
INFO:__main__:Response: {'is_task_complete': True, 'require_user_input': False, 'content': 'The current exchange rate from USD to EUR is 0.8896. This means that 1 US Dollar equals 0.8896 Euros as of May 20, 2025.'}
INFO:__main__:
Testing query: Can you tell me the exchange rate between GBP and JPY?
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
INFO:__main__:Response: {'is_task_complete': False, 'require_user_input': False, 'content': 'Looking up the exchange rates...'}
INFO:__main__:Response: {'is_task_complete': False, 'require_user_input': False, 'content': 'Processing the exchange rates...'}
INFO:httpx:HTTP Request: GET https://api.frankfurter.app/latest?from=GBP&to=JPY "HTTP/1.1 200 OK"
INFO:currency_agent:API response: {'amount': 1.0, 'base': 'GBP', 'date': '2025-05-20', 'rates': {'JPY': 193.15}}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
INFO:currency_agent:Final message: {'role': 'assistant', 'content': '{\n    "status": "completed",\n    "message": "The current exchange rate from British Pounds (GBP) to Japanese Yen (JPY) is 193.15 JPY for 1 GBP as of May 20, 2025."\n}', 'refusal': None, 'reasoning': None}
INFO:__main__:Response: {'is_task_complete': True, 'require_user_input': False, 'content': 'The current exchange rate from British Pounds (GBP) to Japanese Yen (JPY) is 193.15 JPY for 1 GBP as of May 20, 2025.'}
INFO:__main__:
Testing query: What's the weather like today?
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
INFO:__main__:Response: {'is_task_complete': True, 'require_user_input': False, 'content': "I'm sorry, but I can only assist with currency conversion and exchange rate queries. I cannot provide information about the weather or other unrelated topics. If you have any questions about currency exchange rates or conversions, I'd be happy to help with those."}
```

## 
