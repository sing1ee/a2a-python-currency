
# 使用 A2A Python SDK 实现 CurrencyAgent
谷歌官方的[a2a-python](https://github.com/google/a2a-python) SDK最近频繁的更新，我们的教程也需要跟着更新，这篇文章，我们通过 a2a-python sdk的 `0.2.3` 版本，实现一个简单的CurrencyAgent。

## 源码
项目的源码在[a2a-python-currency](https://github.com/sing1ee/a2a-python-currency)，欢迎 star 。


## 准备
- [uv](https://docs.astral.sh/uv/#installation) 0.7.2，用来进行项目管理
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
uv add a2a-sdk uvicorn dotenv click
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

## 实现 AgentExecutor
```python
from currency_agent import CurrencyAgent  # type: ignore[import-untyped]

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact


class CurrencyAgentExecutor(AgentExecutor):
    """Currency AgentExecutor Example."""

    def __init__(self):
        self.agent = CurrencyAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input()
        task = context.current_task

        if not context.message:
            raise Exception('No message provided')

        if not task:
            task = new_task(context.message)
            event_queue.enqueue_event(task)
        # invoke the underlying agent, using streaming results
        async for event in self.agent.stream(query, task.contextId):
            if event['is_task_complete']:
                event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        append=False,
                        contextId=task.contextId,
                        taskId=task.id,
                        lastChunk=True,
                        artifact=new_text_artifact(
                            name='current_result',
                            description='Result of request to agent.',
                            text=event['content'],
                        ),
                    )
                )
                event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(state=TaskState.completed),
                        final=True,
                        contextId=task.contextId,
                        taskId=task.id,
                    )
                )
            elif event['require_user_input']:
                event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(
                            state=TaskState.input_required,
                            message=new_agent_text_message(
                                event['content'],
                                task.contextId,
                                task.id,
                            ),
                        ),
                        final=True,
                        contextId=task.contextId,
                        taskId=task.id,
                    )
                )
            else:
                event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(
                            state=TaskState.working,
                            message=new_agent_text_message(
                                event['content'],
                                task.contextId,
                                task.id,
                            ),
                        ),
                        final=False,
                        contextId=task.contextId,
                        taskId=task.id,
                    )
                )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception('cancel not supported')

```

我来分析这段代码的逻辑：
这是一个名为 `CurrencyAgentExecutor` 的代理执行器类，主要用于处理货币相关的代理操作。让我详细分析其结构和功能：

A2A 代理处理请求和生成响应/事件的核心逻辑由AgentExecutor。A2A Python SDK 提供了一个抽象基类 *a2a.server.agent_execution.AgentExecutor* ，你需要实现它。

AgentExecutor 类定义了两个主要方法：
- `async def execute(self, context: RequestContext, event_queue: EventQueue)`：处理需要响应或事件流的传入请求。它处理用户的输入（通过 context 获取）并使用 `event_queue` 发送 Message、Task、TaskStatusUpdateEvent 或 TaskArtifactUpdateEvent 对象。
- `async def cancel(self, context: RequestContext, event_queue: EventQueue)`：处理取消正在进行的任务的请求。

RequestContext 提供有关传入请求的信息，如用户的消息和任何现有的任务详情。EventQueue 由执行器用来向客户端发送事件。

## 实现 AgentServer

代码
```python
import os
import sys

import click
import httpx

from currency_agent import CurrencyAgent  # type: ignore[import-untyped]
from agent_executor import CurrencyAgentExecutor  # type: ignore[import-untyped]
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill


load_dotenv()


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10000)
def main(host: str, port: int):

    client = httpx.AsyncClient()
    request_handler = DefaultRequestHandler(
        agent_executor=CurrencyAgentExecutor(),
        task_store=InMemoryTaskStore(),
        push_notifier=InMemoryPushNotifier(client),
    )

    server = A2AStarletteApplication(
        agent_card=get_agent_card(host, port), http_handler=request_handler
    )
    import uvicorn

    uvicorn.run(server.build(), host=host, port=port)


def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the Currency Agent."""
    capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
    skill = AgentSkill(
        id='convert_currency',
        name='Currency Exchange Rates Tool',
        description='Helps with exchange values between various currencies',
        tags=['currency conversion', 'currency exchange'],
        examples=['What is exchange rate between USD and GBP?'],
    )
    return AgentCard(
        name='Currency Agent',
        description='Helps with exchange rates for currencies',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=CurrencyAgent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes=CurrencyAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )


if __name__ == '__main__':
    main()


```

### AgentSkill
AgentSkill描述了代理可以执行的特定能力或功能。它是一个构建块，告诉客户端代理适合执行哪些类型的任务。
AgentSkill 的关键属性（在 a2a.types 中定义）：
- id：技能的唯一标识符。
- name：人类可读的名称。
- description：对技能功能的更详细解释。
- tags：用于分类和发现的关键词。
- examples：示例提示或用例。
- inputModes / outputModes：支持的输入和输出 MIME 类型（例如，"text/plain"，"application/json"）。


这个技能非常简单：处理汇率转换，输入和输出都是 `text`, 在 AgentCard 中定义。
### AgentCard
AgentCard 是 A2A 服务器提供的 JSON 文档，通常位于 `.well-known/agent.json` 端点。它就像是代理的数字名片。
AgentCard 的关键属性（在 a2a.types 中定义）：
- name、description、version：基本身份信息。
- url：可以访问 A2A 服务的端点。
- capabilities：指定支持的 A2A 功能，如 streaming 或 pushNotifications。
- defaultInputModes / defaultOutputModes：代理的默认 MIME 类型。
- skills：代理提供的 AgentSkill 对象列表。

### AgentServer

- DefaultRequestHandler：
SDK 提供了 DefaultRequestHandler。这个处理器接收 AgentExecutor 实现（这里是 CurrencyAgentExecutor）和一个 TaskStore（这里是 InMemoryTaskStore）。
它将传入的 A2A RPC 调用路由到执行器上的适当方法（如 execute 或 cancel）。
TaskStore 被 DefaultRequestHandler 用来管理任务的生命周期，特别是对于有状态交互、流式处理和重新订阅。
即使AgentExecutor很简单，处理器也需要一个任务存储。

- A2AStarletteApplication：
A2AStarletteApplication 类使用 agent_card 和 request_handler（在其构造函数中称为 http_handler）进行实例化。
agent_card 非常关键，因为服务器将默认在 `/.well-known/agent.json` 端点上公开它。
request_handler 负责通过与您的 AgentExecutor 交互来处理所有传入的 A2A 方法调用。

- uvicorn.run(server_app_builder.build(), ...)：
A2AStarletteApplication 有一个 build() 方法，用于构建实际的 [Starlette](https://www.starlette.io/) 应用程序。
然后使用 `uvicorn.run()` 运行此应用程序，使您的代理可通过 HTTP 访问。
host='0.0.0.0' 使服务器在您机器上的所有网络接口上可访问。
port=9999 指定要监听的端口。这与 AgentCard 中的 url 相匹配。

## 运行

### 运行 Server
```bash
uv run python main.py
```
输出：
```bash
INFO:     Started server process [70842]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:10000 (Press CTRL+C to quit)
```

### 运行 Client
client 代码如下
```python
from a2a.client import A2AClient
from typing import Any
from uuid import uuid4
from a2a.types import (
    SendMessageResponse,
    GetTaskResponse,
    SendMessageSuccessResponse,
    Task,
    TaskState,
    SendMessageRequest,
    MessageSendParams,
    GetTaskRequest,
    TaskQueryParams,
    SendStreamingMessageRequest,
)
import httpx
import traceback

AGENT_URL = 'http://localhost:10000'


def create_send_message_payload(
    text: str, task_id: str | None = None, context_id: str | None = None
) -> dict[str, Any]:
    """Helper function to create the payload for sending a task."""
    payload: dict[str, Any] = {
        'message': {
            'role': 'user',
            'parts': [{'kind': 'text', 'text': text}],
            'messageId': uuid4().hex,
        },
    }

    if task_id:
        payload['message']['taskId'] = task_id

    if context_id:
        payload['message']['contextId'] = context_id
    return payload


def print_json_response(response: Any, description: str) -> None:
    """Helper function to print the JSON representation of a response."""
    print(f'--- {description} ---')
    if hasattr(response, 'root'):
        print(f'{response.root.model_dump_json(exclude_none=True)}\n')
    else:
        print(f'{response.model_dump(mode="json", exclude_none=True)}\n')


async def run_single_turn_test(client: A2AClient) -> None:
    """Runs a single-turn non-streaming test."""

    send_payload = create_send_message_payload(
        text='how much is 100 USD in CAD?'
    )
    request = SendMessageRequest(params=MessageSendParams(**send_payload))

    print('--- Single Turn Request ---')
    # Send Message
    send_response: SendMessageResponse = await client.send_message(request)
    print_json_response(send_response, 'Single Turn Request Response')
    if not isinstance(send_response.root, SendMessageSuccessResponse):
        print('received non-success response. Aborting get task ')
        return

    if not isinstance(send_response.root.result, Task):
        print('received non-task response. Aborting get task ')
        return

    task_id: str = send_response.root.result.id
    print('---Query Task---')
    # query the task
    get_request = GetTaskRequest(params=TaskQueryParams(id=task_id))
    get_response: GetTaskResponse = await client.get_task(get_request)
    print_json_response(get_response, 'Query Task Response')


async def run_streaming_test(client: A2AClient) -> None:
    """Runs a single-turn streaming test."""

    send_payload = create_send_message_payload(
        text='how much is 50 EUR in JPY?'
    )

    request = SendStreamingMessageRequest(
        params=MessageSendParams(**send_payload)
    )

    print('--- Single Turn Streaming Request ---')
    stream_response = client.send_message_streaming(request)
    async for chunk in stream_response:
        print_json_response(chunk, 'Streaming Chunk')


async def run_multi_turn_test(client: A2AClient) -> None:
    """Runs a multi-turn non-streaming test."""
    print('--- Multi-Turn Request ---')
    # --- First Turn ---

    first_turn_payload = create_send_message_payload(
        text='how much is 100 USD?'
    )
    request1 = SendMessageRequest(
        params=MessageSendParams(**first_turn_payload)
    )
    first_turn_response: SendMessageResponse = await client.send_message(
        request1
    )
    print_json_response(first_turn_response, 'Multi-Turn: First Turn Response')

    context_id: str | None = None
    if isinstance(
        first_turn_response.root, SendMessageSuccessResponse
    ) and isinstance(first_turn_response.root.result, Task):
        task: Task = first_turn_response.root.result
        context_id = task.contextId  # Capture context ID

        # --- Second Turn (if input required) ---
        if task.status.state == TaskState.input_required and context_id:
            print('--- Multi-Turn: Second Turn (Input Required) ---')
            second_turn_payload = create_send_message_payload(
                'in GBP', task.id, context_id
            )
            request2 = SendMessageRequest(
                params=MessageSendParams(**second_turn_payload)
            )
            second_turn_response = await client.send_message(request2)
            print_json_response(
                second_turn_response, 'Multi-Turn: Second Turn Response'
            )
        elif not context_id:
            print('Warning: Could not get context ID from first turn response.')
        else:
            print(
                'First turn completed, no further input required for this test case.'
            )


async def main() -> None:
    """Main function to run the tests."""
    print(f'Connecting to agent at {AGENT_URL}...')
    try:
        async with httpx.AsyncClient(timeout=100) as httpx_client:
            client = await A2AClient.get_client_from_agent_card_url(
                httpx_client, AGENT_URL
            )
            print('Connection successful.')
            await run_single_turn_test(client)
            await run_streaming_test(client)
            await run_multi_turn_test(client)

    except Exception as e:
        traceback.print_exc()
        print(f'An error occurred: {e}')
        print('Ensure the agent server is running.')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())

```

运行如下：
```bash
uv run python test_client.py
Connecting to agent at http://localhost:10000...
Connection successful.
--- Single Turn Request ---
--- Single Turn Request Response ---
{"id":"f403b867-1b98-466e-b6ae-e4506e41d02a","jsonrpc":"2.0","result":{"artifacts":[{"artifactId":"50edd29a-57e8-4c68-9eea-10d2127a954f","description":"Result of request to agent.","name":"current_result","parts":[{"kind":"text","text":"Based on the current exchange rate, 100 USD is equivalent to 139.52 CAD. This rate is as of 2025-05-20."}]}],"contextId":"9b70f49f-7e4e-495d-b90b-9aad39b2bdbe","history":[{"contextId":"9b70f49f-7e4e-495d-b90b-9aad39b2bdbe","kind":"message","messageId":"5a66747536d745378ccb914832ad0d30","parts":[{"kind":"text","text":"how much is 100 USD in CAD?"}],"role":"user","taskId":"8e471995-756f-4bc0-afff-b07f86d8c608"},{"contextId":"9b70f49f-7e4e-495d-b90b-9aad39b2bdbe","kind":"message","messageId":"f3f645af-158d-428d-887f-301a518d5150","parts":[{"kind":"text","text":"Looking up the exchange rates..."}],"role":"agent","taskId":"8e471995-756f-4bc0-afff-b07f86d8c608"},{"contextId":"9b70f49f-7e4e-495d-b90b-9aad39b2bdbe","kind":"message","messageId":"74c90f14-8e21-4ffc-9a46-c3cfec984620","parts":[{"kind":"text","text":"Processing the exchange rates..."}],"role":"agent","taskId":"8e471995-756f-4bc0-afff-b07f86d8c608"}],"id":"8e471995-756f-4bc0-afff-b07f86d8c608","kind":"task","status":{"state":"completed"}}}

---Query Task---
--- Query Task Response ---
{"id":"f07d56ae-b998-4da5-bb6f-53f1edd7c315","jsonrpc":"2.0","result":{"artifacts":[{"artifactId":"50edd29a-57e8-4c68-9eea-10d2127a954f","description":"Result of request to agent.","name":"current_result","parts":[{"kind":"text","text":"Based on the current exchange rate, 100 USD is equivalent to 139.52 CAD. This rate is as of 2025-05-20."}]}],"contextId":"9b70f49f-7e4e-495d-b90b-9aad39b2bdbe","history":[{"contextId":"9b70f49f-7e4e-495d-b90b-9aad39b2bdbe","kind":"message","messageId":"5a66747536d745378ccb914832ad0d30","parts":[{"kind":"text","text":"how much is 100 USD in CAD?"}],"role":"user","taskId":"8e471995-756f-4bc0-afff-b07f86d8c608"},{"contextId":"9b70f49f-7e4e-495d-b90b-9aad39b2bdbe","kind":"message","messageId":"f3f645af-158d-428d-887f-301a518d5150","parts":[{"kind":"text","text":"Looking up the exchange rates..."}],"role":"agent","taskId":"8e471995-756f-4bc0-afff-b07f86d8c608"},{"contextId":"9b70f49f-7e4e-495d-b90b-9aad39b2bdbe","kind":"message","messageId":"74c90f14-8e21-4ffc-9a46-c3cfec984620","parts":[{"kind":"text","text":"Processing the exchange rates..."}],"role":"agent","taskId":"8e471995-756f-4bc0-afff-b07f86d8c608"}],"id":"8e471995-756f-4bc0-afff-b07f86d8c608","kind":"task","status":{"state":"completed"}}}

--- Single Turn Streaming Request ---
--- Streaming Chunk ---
{"id":"06695ada-11e5-4bf5-8e5f-0e4731002def","jsonrpc":"2.0","result":{"contextId":"221722b1-5ee4-4a61-adb7-65c513f5c1aa","history":[{"contextId":"221722b1-5ee4-4a61-adb7-65c513f5c1aa","kind":"message","messageId":"dcc004a7af814b18b1a8868b1bd091be","parts":[{"kind":"text","text":"how much is 50 EUR in JPY?"}],"role":"user","taskId":"7c4c23c1-2652-4f68-a233-71dae8e84499"}],"id":"7c4c23c1-2652-4f68-a233-71dae8e84499","kind":"task","status":{"state":"submitted"}}}

--- Streaming Chunk ---
{"id":"06695ada-11e5-4bf5-8e5f-0e4731002def","jsonrpc":"2.0","result":{"contextId":"221722b1-5ee4-4a61-adb7-65c513f5c1aa","final":false,"kind":"status-update","status":{"message":{"contextId":"221722b1-5ee4-4a61-adb7-65c513f5c1aa","kind":"message","messageId":"7d799de0-9073-46c1-a300-4b4a25591145","parts":[{"kind":"text","text":"Looking up the exchange rates..."}],"role":"agent","taskId":"7c4c23c1-2652-4f68-a233-71dae8e84499"},"state":"working"},"taskId":"7c4c23c1-2652-4f68-a233-71dae8e84499"}}

--- Streaming Chunk ---
{"id":"06695ada-11e5-4bf5-8e5f-0e4731002def","jsonrpc":"2.0","result":{"contextId":"221722b1-5ee4-4a61-adb7-65c513f5c1aa","final":false,"kind":"status-update","status":{"message":{"contextId":"221722b1-5ee4-4a61-adb7-65c513f5c1aa","kind":"message","messageId":"b0d03a04-43d4-44fb-b1bd-b6321b960633","parts":[{"kind":"text","text":"Processing the exchange rates..."}],"role":"agent","taskId":"7c4c23c1-2652-4f68-a233-71dae8e84499"},"state":"working"},"taskId":"7c4c23c1-2652-4f68-a233-71dae8e84499"}}

--- Streaming Chunk ---
{"id":"06695ada-11e5-4bf5-8e5f-0e4731002def","jsonrpc":"2.0","result":{"append":false,"artifact":{"artifactId":"252c35db-8cca-4797-be24-6f3b57b4194b","description":"Result of request to agent.","name":"current_result","parts":[{"kind":"text","text":"50 EUR is equivalent to 8,129.50 JPY based on the current exchange rate (1 EUR = 162.59 JPY as of May 20, 2025)."}]},"contextId":"221722b1-5ee4-4a61-adb7-65c513f5c1aa","kind":"artifact-update","lastChunk":true,"taskId":"7c4c23c1-2652-4f68-a233-71dae8e84499"}}

--- Streaming Chunk ---
{"id":"06695ada-11e5-4bf5-8e5f-0e4731002def","jsonrpc":"2.0","result":{"contextId":"221722b1-5ee4-4a61-adb7-65c513f5c1aa","final":true,"kind":"status-update","status":{"state":"completed"},"taskId":"7c4c23c1-2652-4f68-a233-71dae8e84499"}}

--- Multi-Turn Request ---
--- Multi-Turn: First Turn Response ---
{"id":"1fcb346b-4ba3-472d-938b-216c6e387e73","jsonrpc":"2.0","result":{"contextId":"136fcbf7-e696-4624-a772-8068091f72c8","history":[{"contextId":"136fcbf7-e696-4624-a772-8068091f72c8","kind":"message","messageId":"a6e95e1328594ebb9d01b8ec56dd7aaa","parts":[{"kind":"text","text":"how much is 100 USD?"}],"role":"user","taskId":"5011389c-3fba-4c81-b4c4-4c9e5d9d0a6b"}],"id":"5011389c-3fba-4c81-b4c4-4c9e5d9d0a6b","kind":"task","status":{"message":{"contextId":"136fcbf7-e696-4624-a772-8068091f72c8","kind":"message","messageId":"bc93f6be-f6a6-4f5e-88bd-861646e6d6bb","parts":[{"kind":"text","text":"To provide a currency conversion, I need to know which currency you want to convert USD to. Please specify the target currency. For example: 'How much is 100 USD in EUR?'"}],"role":"agent","taskId":"5011389c-3fba-4c81-b4c4-4c9e5d9d0a6b"},"state":"input-required"}}}

--- Multi-Turn: Second Turn (Input Required) ---
--- Multi-Turn: Second Turn Response ---
{"id":"4907a948-7667-4177-a05d-49ddd4b30204","jsonrpc":"2.0","result":{"contextId":"136fcbf7-e696-4624-a772-8068091f72c8","history":[{"contextId":"136fcbf7-e696-4624-a772-8068091f72c8","kind":"message","messageId":"a6e95e1328594ebb9d01b8ec56dd7aaa","parts":[{"kind":"text","text":"how much is 100 USD?"}],"role":"user","taskId":"5011389c-3fba-4c81-b4c4-4c9e5d9d0a6b"},{"contextId":"136fcbf7-e696-4624-a772-8068091f72c8","kind":"message","messageId":"bc93f6be-f6a6-4f5e-88bd-861646e6d6bb","parts":[{"kind":"text","text":"To provide a currency conversion, I need to know which currency you want to convert USD to. Please specify the target currency. For example: 'How much is 100 USD in EUR?'"}],"role":"agent","taskId":"5011389c-3fba-4c81-b4c4-4c9e5d9d0a6b"},{"contextId":"136fcbf7-e696-4624-a772-8068091f72c8","kind":"message","messageId":"f343dd31d96b4f7a87d633b6f2cefaf0","parts":[{"kind":"text","text":"in GBP"}],"role":"user","taskId":"5011389c-3fba-4c81-b4c4-4c9e5d9d0a6b"},{"contextId":"136fcbf7-e696-4624-a772-8068091f72c8","kind":"message","messageId":"cff157b7-ecdd-4057-a2bf-c93168c37fd0","parts":[{"kind":"text","text":"Looking up the exchange rates..."}],"role":"agent","taskId":"5011389c-3fba-4c81-b4c4-4c9e5d9d0a6b"},{"contextId":"136fcbf7-e696-4624-a772-8068091f72c8","kind":"message","messageId":"ea6a5ffb-0106-4ba8-a68e-497d50718594","parts":[{"kind":"text","text":"Processing the exchange rates..."}],"role":"agent","taskId":"5011389c-3fba-4c81-b4c4-4c9e5d9d0a6b"}],"id":"5011389c-3fba-4c81-b4c4-4c9e5d9d0a6b","kind":"task","status":{"message":{"contextId":"136fcbf7-e696-4624-a772-8068091f72c8","kind":"message","messageId":"602a8178-ebd3-4447-ab50-53b487369737","parts":[{"kind":"text","text":"We are unable to process your request at the moment. Please try again."}],"role":"agent","taskId":"5011389c-3fba-4c81-b4c4-4c9e5d9d0a6b"},"state":"input-required"}}}
```

教程结束。