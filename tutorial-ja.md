# A2A Python SDKを使用したCurrencyAgentの実装

Googleの公式SDK [a2a-python](https://github.com/google/a2a-python)は頻繁に更新されており、私たちのチュートリアルも更新する必要があります。この記事では、a2a-python SDKのバージョン`0.2.3`を使用して、シンプルなCurrencyAgentを実装します。

## 目次
- [ソースコード](#ソースコード)
- [準備](#準備)
- [詳細な手順](#詳細な手順)
  - [プロジェクトの作成](#プロジェクトの作成)
  - [仮想環境の作成](#仮想環境の作成)
  - [依存関係の追加](#依存関係の追加)
  - [環境変数の設定](#環境変数の設定)
  - [エージェントの作成](#エージェントの作成)
    - [主要な機能](#1-主要な機能)
    - [システムアーキテクチャ](#2-システムアーキテクチャ)
      - [System Prompt](#21-system-prompt)
      - [主要なメソッド](#22-主要なメソッド)
    - [ワークフロー](#3-ワークフロー)
    - [レスポンス形式](#4-レスポンス形式)
    - [エラー処理](#5-エラー処理)
  - [エージェントのテスト](#エージェントのテスト)
  - [AgentExecutorの実装](#agentexecutorの実装)
  - [AgentServerの実装](#agentserverの実装)
    - [AgentSkill](#agentskill)
    - [AgentCard](#agentcard)
    - [AgentServer](#agentserver-1)
  - [実行](#実行)
    - [サーバーの実行](#サーバーの実行)
    - [クライアントの実行](#クライアントの実行)

## ソースコード
プロジェクトのソースコードは[a2a-python-currency](https://github.com/sing1ee/a2a-python-currency)で入手できます。スターを付けることをお勧めします。

## 準備
- [uv](https://docs.astral.sh/uv/#installation) 0.7.2、プロジェクト管理用
- Python 3.13+、a2a-pythonに必要なバージョン
- openai/openrouterのapiKeyとbaseURL。私は[OpenRouter](https://openrouter.ai/)を使用しています。これはより多くのモデルオプションを提供します。

## 詳細な手順

### プロジェクトの作成
```bash
uv init a2a-python-currency
cd a2a-python-currency
```

### 仮想環境の作成
```bash
uv venv
source .venv/bin/activate
```

### 依存関係の追加
```bash
uv add a2a-sdk uvicorn dotenv click
```

### 環境変数の設定
```bash
echo OPENROUTER_API_KEY=your_api_key >> .env
echo OPENROUTER_BASE_URL=your_base_url >> .env

# 例
OPENROUTER_API_KEY=あなたのOpenRouterのAPIキー
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

### エージェントの作成
完全なコードは以下の通りです：
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
    """OpenAI APIを使用した通貨変換エージェント。"""

    SYSTEM_PROMPT = """あなたは通貨変換に特化したアシスタントです。
あなたの唯一の目的は、為替レートに関する質問に答えるために'tool_use'ツールを使用することです。
ユーザーが通貨変換や為替レート以外の質問をした場合、
丁寧に、そのトピックについてはお手伝いできないこと、通貨に関連する質問のみお手伝いできることを伝えてください。
関連のない質問に答えたり、ツールを他の目的で使用しようとしないでください。

あなたは以下のツールにアクセスできます：
- get_exchange_rate: 2つの通貨間の現在の為替レートを取得する

ツールを使用する際は、以下のJSON形式で応答してください：
{
    "status": "completed" | "input_required" | "error",
    "message": "あなたの応答メッセージ"
}

ツールを使用する必要がある場合は、以下のように応答してください：
{
    "status": "tool_use",
    "tool": "get_exchange_rate",
    "parameters": {
        "currency_from": "USD",
        "currency_to": "EUR",
        "currency_date": "latest"
    }
}
注意：応答はJSON形式でのみ返してください。JSONのみが許可されています。
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
        """通貨間の現在の為替レートを取得します。"""
        try:
            response = httpx.get(
                f'https://api.frankfurter.app/{currency_date}',
                params={'from': currency_from, 'to': currency_to},
            )
            response.raise_for_status()
            data = response.json()
            if 'rates' not in data:
                logger.error(f'rates not found in response: {data}')
                return {'error': 'APIレスポンスの形式が無効です。'}
            logger.info(f'API response: {data}')
            return data
        except httpx.HTTPError as e:
            logger.error(f'API request failed: {e}')
            return {'error': f'APIリクエストが失敗しました：{e}'}
        except ValueError:
            logger.error('Invalid JSON response from API')
            return {'error': 'APIからのJSONレスポンスが無効です。'}

    async def _call_openai(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """OpenRouter経由でOpenAI APIを呼び出します。"""
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
        """指定されたクエリに対するレスポンスをストリーミングします。"""
        # ユーザーメッセージを会話履歴に追加
        self.conversation_history.append({"role": "user", "content": query})

        # API呼び出し用のメッセージを準備
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + self.conversation_history

        # OpenAIからのレスポンスを取得
        response = await self._call_openai(messages)
        assistant_message = response["choices"][0]["message"]["content"]
        print(assistant_message)
        try:
            # レスポンスをJSONとして解析
            parsed_response = json.loads(assistant_message)
            
            # ツール使用のリクエストの場合
            if parsed_response.get("status") == "tool_use":
                tool_name = parsed_response["tool"]
                parameters = parsed_response["parameters"]
                
                # ツール使用のステータスを出力
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "為替レートを検索中..."
                }
                
                if tool_name == "get_exchange_rate":
                    # 処理ステータスを出力
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": "為替レートを処理中..."
                    }
                    
                    tool_result = await self.get_exchange_rate(**parameters)
                    
                    # ツールの結果を会話履歴に追加
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": json.dumps({"tool_result": tool_result})
                    })
                    
                    # ツール使用後の最終レスポンスを取得
                    final_response = await self._call_openai(messages)
                    final_message = final_response["choices"][0]["message"]["content"]
                    parsed_response = json.loads(final_message)

            # アシスタントのレスポンスを会話履歴に追加
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # 最終レスポンスを出力
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
                    "content": "現在、リクエストを処理できません。もう一度お試しください。"
                }

        except json.JSONDecodeError:
            # レスポンスが有効なJSONでない場合、エラーを返す
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "モデルからのレスポンス形式が無効です。"
            } 
```

主要な機能と実装ロジックの分析：

#### 1. 主要な機能
- 通貨変換と為替レートのリクエスト処理に特化
- Frankfurter APIを使用してリアルタイムの為替レートデータを取得
- OpenRouter経由でClaude 3.7 Sonnetモデルを使用して会話を処理

#### 2. システムアーキテクチャ
エージェントは以下の主要コンポーネントで構成されています：

##### 2.1 System Prompt
- エージェントの具体的な目的を定義：通貨変換のリクエストのみを処理
- レスポンス形式を定義：JSON形式を使用する必要がある
- ツールの使用を定義：為替レート情報を取得するために`get_exchange_rate`ツールを使用

##### 2.2 主要なメソッド
1. **初期化メソッド `__init__`**
   - APIキーとベースURLを設定
   - 会話履歴を初期化

2. **為替レート取得メソッド `get_exchange_rate`**
   - パラメータ：ソース通貨、ターゲット通貨、日付（デフォルトは最新）
   - Frankfurter APIを呼び出して為替レートデータを取得
   - 為替レート情報をJSON形式で返す

3. **ストリーミングメソッド `stream`**
   - ストリーミングレスポンス機能を提供
   - 処理ステータスと結果をリアルタイムで返す
   - ツール呼び出しの中間ステータスをサポート

#### 3. ワークフロー
1. **ユーザーリクエストの受信**
   - ユーザーメッセージを会話履歴に追加

2. **モデルによる処理**
   - System Promptと会話履歴をモデルに送信
   - モデルがツールの使用が必要かどうかを分析

3. **ツール呼び出し（必要な場合）**
   - モデルがツールを使用すると判断した場合、ツール呼び出しリクエストを返す
   - 為替レートリクエストを実行
   - リクエスト結果を会話履歴に追加

4. **最終レスポンスの生成**
   - ツール呼び出しの結果に基づいて最終レスポンスを生成
   - JSON形式でレスポンスを返す

#### 4. レスポンス形式
エージェントのレスポンスは常に以下の状態を持つJSON形式を使用：
- `completed`: タスク完了
- `input_required`: ユーザー入力が必要
- `error`: エラーが発生
- `tool_use`: ツールの使用が必要

#### 5. エラー処理
- 完全なエラー処理メカニズムを含む
- API呼び出しの失敗を処理
- JSON解析エラーを処理
- 無効なレスポンス形式を処理

### エージェントのテスト
テストコードは以下の通りです：
```python
import asyncio
import logging
from currency_agent import CurrencyAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    agent = CurrencyAgent()
    
    # テストケース
    test_queries = [
        "What is the current exchange rate from USD to EUR?",
        "Can you tell me the exchange rate between GBP and JPY?",
        "What's the weather like today?",  # 通貨に関連しないため拒否されるべき
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: {query}")
        async for response in agent.stream(query, "test_session"):
            logger.info(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main()) 
```

環境設定が正しく行われている場合、特に環境設定が正しい場合、以下のような出力が表示されるはずです：
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
```

### AgentExecutorの実装
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
    """通貨用のAgentExecutorの例。"""

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
        # ストリーミング結果を使用してベースエージェントを呼び出す
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
                            description='エージェントへのリクエスト結果。',
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

このコードのロジック分析：
これは`CurrencyAgentExecutor`という名前のAgentExecutorクラスで、主に通貨に関連するエージェント操作を処理します。その構造と機能を詳しく分析しましょう：

A2Aリクエストの処理とレスポンス/イベントの生成の中心ロジックは、AgentExecutorを通じて実装されています。A2A Python SDKは、実装する必要がある抽象基本クラス*a2a.server.agent_execution.AgentExecutor*を提供します。

AgentExecutorクラスは2つの主要なメソッドを定義します：
- `async def execute(self, context: RequestContext, event_queue: EventQueue)`: レスポンスやイベントストリームを必要とする受信リクエストを処理します。コンテキストから取得したユーザー入力を処理し、`event_queue`を使用してMessage、Task、TaskStatusUpdateEvent、またはTaskArtifactUpdateEventオブジェクトを送信します。
- `async def cancel(self, context: RequestContext, event_queue: EventQueue)`: 現在のタスクのキャンセルリクエストを処理します。

RequestContextは、ユーザーメッセージや既存のタスク詳細など、受信リクエストに関する情報を提供します。EventQueueは、エージェントがクライアントにイベントを送信するために使用されます。

### AgentServerの実装

コード：
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
    """通貨エージェントのエージェントカードを返します。"""
    capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
    skill = AgentSkill(
        id='convert_currency',
        name='為替レートツール',
        description='異なる通貨間の為替レートを支援します',
        tags=['通貨変換', '為替'],
        examples=['What is exchange rate between USD and GBP?'],
    )
    return AgentCard(
        name='通貨エージェント',
        description='為替レートを支援します',
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

#### AgentSkill
AgentSkillは、エージェントが実行できるスキルや特定の機能を記述します。これは、クライアントにエージェントが適しているタスクの種類を通知するビルディングブロックです。
AgentSkillの主要な属性（a2a.typesで定義）：
- id: スキルの一意の識別子
- name: 人間が読める名前
- description: スキルの機能性のより詳細な説明
- tags: 分類と検出のためのキーワード
- examples: プロンプトやユースケースの例
- inputModes / outputModes: 入力と出力のサポートされるMIMEタイプ（例："text/plain"、"application/json"）

このスキルは非常にシンプルです：通貨変換の処理、入力と出力は`text`、AgentCardで定義されています。

#### AgentCard
AgentCardは、A2Aサーバーによって提供されるJSONドキュメントで、通常は`.well-known/agent.json`エンドポイントに配置されます。これはエージェントのデジタル名刺のようなものです。
AgentCardの主要な属性（a2a.typesで定義）：
- name, description, version: 基本的な識別情報
- url: A2Aサービスにアクセスするためのエンドポイント
- capabilities: streamingやpushNotificationsなどのサポートされるA2A機能を指定
- defaultInputModes / defaultOutputModes: エージェントのデフォルトMIMEタイプ
- skills: エージェントが提供するAgentSkillオブジェクトのリスト

#### AgentServer

- DefaultRequestHandler:
SDKはDefaultRequestHandlerを提供します。このハンドラーは、AgentExecutorの実装（ここではCurrencyAgentExecutor）とTaskStore（ここではInMemoryTaskStore）を受け取ります。
受信したA2A RPC呼び出しを、executeやcancelなどのエージェントの適切なメソッドにルーティングします。
TaskStoreは、DefaultRequestHandlerによってタスクのライフサイクルを管理するために使用され、特に状態のある相互作用、ストリーミング、再サブスクリプションに使用されます。
AgentExecutorがシンプルであっても、ハンドラーにはタスクストアが必要です。

- A2AStarletteApplication:
A2AStarletteApplicationクラスは、agent_cardとrequest_handler（コンストラクタではhttp_handlerと呼ばれる）を使用して作成されます。
agent_cardは非常に重要です。サーバーはデフォルトで`/.well-known/agent.json`エンドポイントでこれを提供します。
request_handlerは、そのAgentExecutorとの相互作用を通じて、受信したすべてのA2Aメソッド呼び出しの処理を担当します。

- uvicorn.run(server_app_builder.build(), ...):
A2AStarletteApplicationには、実際の[Starlette](https://www.starlette.io/)アプリケーションを構築するためのbuild()メソッドがあります。
このアプリケーションは`uvicorn.run()`を使用して実行され、HTTP経由でエージェントにアクセスできるようになります。
host='0.0.0.0'は、サーバーをマシンのすべてのネットワークインターフェースでアクセス可能にします。
port=9999はリッスンするポートを指定します。これはAgentCardのurlに対応します。

### 実行

#### サーバーの実行
```bash
uv run python main.py
```
出力：
```bash
INFO:     Started server process [70842]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:10000 (Press CTRL+C to quit)
```

#### クライアントの実行
クライアントコードは以下の通りです：
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
    """タスク送信用のペイロードを作成するヘルパー関数。"""
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
    """レスポンスのJSON表現を出力するヘルパー関数。"""
    print(f'--- {description} ---')
    if hasattr(response, 'root'):
        print(f'{response.root.model_dump_json(exclude_none=True)}\n')
    else:
        print(f'{response.model_dump(mode="json", exclude_none=True)}\n')


async def run_single_turn_test(client: A2AClient) -> None:
    """ストリーミングなしの単一ターンテストを実行します。"""

    send_payload = create_send_message_payload(
        text='how much is 100 USD in CAD?'
    )
    request = SendMessageRequest(params=MessageSendParams(**send_payload))

    print('--- Single Turn Request ---')
    # メッセージを送信
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
    # タスクをクエリ
    get_request = GetTaskRequest(params=TaskQueryParams(id=task_id))
    get_response: GetTaskResponse = await client.get_task(get_request)
    print_json_response(get_response, 'Query Task Response')


async def run_streaming_test(client: A2AClient) -> None:
    """ストリーミング付きの単一ターンテストを実行します。"""

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
    """ストリーミングなしの複数ターンテストを実行します。"""
    print('--- Multi-Turn Request ---')
    # --- 最初のターン ---

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
        context_id = task.contextId  # コンテキストIDをキャプチャ

        # --- 2番目のターン（入力が必要な場合） ---
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
    """テストを実行するメイン関数。"""
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

実行：
```bash
uv run python test_client.py
```

チュートリアル終了。 