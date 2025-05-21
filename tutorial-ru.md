# Реализация CurrencyAgent с использованием SDK A2A Python

Официальный SDK Google [a2a-python](https://github.com/google/a2a-python) получает частые обновления, и наш учебник также нуждается в обновлении. В этой статье мы реализуем простой CurrencyAgent, используя версию `0.2.3` SDK a2a-python.

## Содержание
- [Исходный код](#исходный-код)
- [Подготовка](#подготовка)
- [Подробный процесс](#подробный-процесс)
  - [Создание проекта](#создание-проекта)
  - [Создание виртуального окружения](#создание-виртуального-окружения)
  - [Добавление зависимостей](#добавление-зависимостей)
  - [Настройка переменных окружения](#настройка-переменных-окружения)
  - [Создание агента](#создание-агента)
    - [Основные функции](#1-основные-функции)
    - [Архитектура системы](#2-архитектура-системы)
      - [System Prompt](#21-system-prompt)
      - [Основные методы](#22-основные-методы)
    - [Рабочий процесс](#3-рабочий-процесс)
    - [Формат ответа](#4-формат-ответа)
    - [Обработка ошибок](#5-обработка-ошибок)
  - [Тестирование агента](#тестирование-агента)
  - [Реализация AgentExecutor](#реализация-agentexecutor)
  - [Реализация AgentServer](#реализация-agentserver)
    - [AgentSkill](#agentskill)
    - [AgentCard](#agentcard)
    - [AgentServer](#agentserver-1)
  - [Запуск](#запуск)
    - [Запуск сервера](#запуск-сервера)
    - [Запуск клиента](#запуск-клиента)

## Исходный код
Исходный код проекта доступен на [a2a-python-currency](https://github.com/sing1ee/a2a-python-currency). Не стесняйтесь поставить звезду.

## Подготовка
- [uv](https://docs.astral.sh/uv/#installation) 0.7.2, для управления проектом
- Python 3.13+, эта версия необходима для a2a-python
- apiKey и baseURL от openai/openrouter. Я использую [OpenRouter](https://openrouter.ai/), который предлагает больше вариантов моделей.

## Подробный процесс

### Создание проекта
```bash
uv init a2a-python-currency
cd a2a-python-currency
```

### Создание виртуального окружения
```bash
uv venv
source .venv/bin/activate
```

### Добавление зависимостей
```bash
uv add a2a-sdk uvicorn dotenv click
```

### Настройка переменных окружения
```bash
echo OPENROUTER_API_KEY=your_api_key >> .env
echo OPENROUTER_BASE_URL=your_base_url >> .env

# пример
OPENROUTER_API_KEY=ваш_ключ_api_OpenRouter
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

### Создание агента
Полный код выглядит следующим образом:
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
    """Агент конвертации валют, использующий API OpenAI."""

    SYSTEM_PROMPT = """Вы - специализированный ассистент для конвертации валют.
Ваша единственная цель - использовать инструмент 'get_exchange_rate' для ответов на вопросы о курсах валют.
Если пользователь задает вопросы не о конвертации валют или курсах валют,
вежливо сообщите, что вы не можете помочь по этой теме и можете помогать только с вопросами, связанными с валютами.
Не пытайтесь отвечать на несвязанные вопросы или использовать инструменты для других целей.

У вас есть доступ к следующему инструменту:
- get_exchange_rate: Получить текущий курс обмена между двумя валютами

При использовании инструмента отвечайте в следующем формате JSON:
{
    "status": "completed" | "input_required" | "error",
    "message": "ваше сообщение ответа"
}

Если вам нужно использовать инструмент, ответьте:
{
    "status": "tool_use",
    "tool": "get_exchange_rate",
    "parameters": {
        "currency_from": "USD",
        "currency_to": "EUR",
        "currency_date": "latest"
    }
}
Примечание: Возвращайте ответ в формате JSON, только json разрешен.
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
        """Получить текущий курс обмена между валютами."""
        try:
            response = httpx.get(
                f'https://api.frankfurter.app/{currency_date}',
                params={'from': currency_from, 'to': currency_to},
            )
            response.raise_for_status()
            data = response.json()
            if 'rates' not in data:
                logger.error(f'rates not found in response: {data}')
                return {'error': 'Неверный формат ответа API.'}
            logger.info(f'API response: {data}')
            return data
        except httpx.HTTPError as e:
            logger.error(f'API request failed: {e}')
            return {'error': f'Ошибка запроса API: {e}'}
        except ValueError:
            logger.error('Invalid JSON response from API')
            return {'error': 'Неверный JSON-ответ от API.'}

    async def _call_openai(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Вызвать API OpenAI через OpenRouter."""
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
        """Потоковая передача ответа для заданного запроса."""
        # Добавить сообщение пользователя в историю разговора
        self.conversation_history.append({"role": "user", "content": query})

        # Подготовить сообщения для вызова API
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + self.conversation_history

        # Получить ответ от OpenAI
        response = await self._call_openai(messages)
        assistant_message = response["choices"][0]["message"]["content"]
        print(assistant_message)
        try:
            # Попытаться разобрать ответ как JSON
            parsed_response = json.loads(assistant_message)
            
            # Если это запрос на использование инструмента
            if parsed_response.get("status") == "tool_use":
                tool_name = parsed_response["tool"]
                parameters = parsed_response["parameters"]
                
                # Выдать статус использования инструмента
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Поиск курсов валют..."
                }
                
                if tool_name == "get_exchange_rate":
                    # Выдать статус обработки
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": "Обработка курсов валют..."
                    }
                    
                    tool_result = await self.get_exchange_rate(**parameters)
                    
                    # Добавить результат инструмента в историю разговора
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": json.dumps({"tool_result": tool_result})
                    })
                    
                    # Получить окончательный ответ после использования инструмента
                    final_response = await self._call_openai(messages)
                    final_message = final_response["choices"][0]["message"]["content"]
                    parsed_response = json.loads(final_message)

            # Добавить ответ ассистента в историю разговора
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Выдать окончательный ответ
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
                    "content": "В настоящее время мы не можем обработать ваш запрос. Пожалуйста, попробуйте снова."
                }

        except json.JSONDecodeError:
            # Если ответ не является допустимым JSON, вернуть ошибку
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "Неверный формат ответа от модели."
            } 
```

Анализ основных функций и логики реализации:

#### 1. Основные функции
- Специализируется на обработке запросов конвертации валют и курсов валют
- Использует API Frankfurter для получения данных о курсах валют в реальном времени
- Обрабатывает разговоры через модель Claude 3.7 Sonnet через OpenRouter

#### 2. Архитектура системы
Агент состоит из нескольких основных компонентов:

##### 2.1 System Prompt
- Определяет конкретную цель агента: обрабатывать только запросы конвертации валют
- Определяет формат ответа: должен использовать формат JSON
- Определяет использование инструментов: использовать инструмент `get_exchange_rate` для получения информации о курсах валют

##### 2.2 Основные методы
1. **Метод инициализации `__init__`**
   - Настраивает API-ключ и базовый URL
   - Инициализирует историю разговоров

2. **Метод запроса курса валют `get_exchange_rate`**
   - Параметры: исходная валюта, целевая валюта, дата (последняя по умолчанию)
   - Вызывает API Frankfurter для получения данных о курсах валют
   - Возвращает информацию о курсе валют в формате JSON

3. **Метод потоковой передачи `stream`**
   - Обеспечивает функциональность потокового ответа
   - Возвращает статус обработки и результаты в реальном времени
   - Поддерживает промежуточный статус для вызовов инструментов

#### 3. Рабочий процесс
1. **Получение запроса пользователя**
   - Добавляет сообщение пользователя в историю разговоров

2. **Обработка моделью**
   - Отправляет System Prompt и историю разговоров модели
   - Модель анализирует, нужно ли использовать инструмент

3. **Вызов инструмента (если необходимо)**
   - Если модель решает использовать инструмент, возвращает запрос на вызов инструмента
   - Выполняет запрос курса валют
   - Добавляет результаты запроса в историю разговоров

4. **Генерация окончательного ответа**
   - Генерирует окончательный ответ на основе результатов вызова инструмента
   - Возвращает ответ в формате JSON

#### 4. Формат ответа
Ответы агента всегда используют формат JSON со следующими состояниями:
- `completed`: задача завершена
- `input_required`: требуется ввод пользователя
- `error`: произошла ошибка
- `tool_use`: необходимо использование инструмента

#### 5. Обработка ошибок
- Включает полный механизм обработки ошибок
- Обрабатывает сбои вызовов API
- Обрабатывает ошибки разбора JSON
- Обрабатывает неверные форматы ответов

### Тестирование агента
Код теста выглядит следующим образом:
```python
import asyncio
import logging
from currency_agent import CurrencyAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    agent = CurrencyAgent()
    
    # Тестовые случаи
    test_queries = [
        "What is the current exchange rate from USD to EUR?",
        "Can you tell me the exchange rate between GBP and JPY?",
        "What's the weather like today?",  # Должен быть отклонен как не связанный с валютами
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: {query}")
        async for response in agent.stream(query, "test_session"):
            logger.info(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main()) 
```

Если все настроено правильно, особенно конфигурация окружения, вы должны увидеть вывод, похожий на:
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

### Реализация AgentExecutor
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
    """Пример AgentExecutor для валют."""

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
        # вызвать базовый агент, используя результаты потоковой передачи
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
                            description='Результат запроса к агенту.',
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

Анализ логики этого кода:
Это класс AgentExecutor под названием `CurrencyAgentExecutor`, который в основном обрабатывает операции агента, связанные с валютами. Давайте проанализируем его структуру и функциональность подробно:

Центральная логика для обработки запросов A2A и генерации ответов/событий реализована через AgentExecutor. SDK A2A Python предоставляет абстрактный базовый класс *a2a.server.agent_execution.AgentExecutor*, который вы должны реализовать.

Класс AgentExecutor определяет два основных метода:
- `async def execute(self, context: RequestContext, event_queue: EventQueue)`: обрабатывает полученные запросы, требующие ответов или потоков событий. Он обрабатывает пользовательский ввод (полученный через контекст) и использует `event_queue` для отправки объектов Message, Task, TaskStatusUpdateEvent или TaskArtifactUpdateEvent.
- `async def cancel(self, context: RequestContext, event_queue: EventQueue)`: обрабатывает запросы на отмену текущих задач.

RequestContext предоставляет информацию о полученном запросе, такую как сообщение пользователя и любые существующие детали задачи. EventQueue используется агентом для отправки событий клиенту.

### Реализация AgentServer

Код:
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
    """Возвращает карточку агента для агента валют."""
    capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
    skill = AgentSkill(
        id='convert_currency',
        name='Инструмент курсов валют',
        description='Помогает с обменными курсами между различными валютами',
        tags=['конвертация валют', 'обмен валют'],
        examples=['What is exchange rate between USD and GBP?'],
    )
    return AgentCard(
        name='Агент валют',
        description='Помогает с курсами валют',
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
AgentSkill описывает навыки или конкретные функции, которые может выполнять агент. Это строительный блок, который информирует клиента о типах задач, для которых подходит агент.
Основные атрибуты AgentSkill (определены в a2a.types):
- id: уникальный идентификатор навыка
- name: человекочитаемое имя
- description: более подробное объяснение функциональности навыка
- tags: ключевые слова для классификации и обнаружения
- examples: примеры промптов или случаев использования
- inputModes / outputModes: поддерживаемые типы MIME для ввода и вывода (например, "text/plain", "application/json")

Этот навык очень прост: обработка конвертации валют, ввод и вывод - `text`, определено в AgentCard.

#### AgentCard
AgentCard - это JSON-документ, предоставляемый сервером A2A, обычно расположенный в эндпоинте `.well-known/agent.json`. Это как цифровая визитная карточка агента.
Основные атрибуты AgentCard (определены в a2a.types):
- name, description, version: основная информация об идентичности
- url: эндпоинт для доступа к сервису A2A
- capabilities: указывает поддерживаемые функции A2A, такие как streaming или pushNotifications
- defaultInputModes / defaultOutputModes: типы MIME по умолчанию для агента
- skills: список объектов AgentSkill, предоставляемых агентом

#### AgentServer

- DefaultRequestHandler:
SDK предоставляет DefaultRequestHandler. Этот обработчик получает реализацию AgentExecutor (здесь CurrencyAgentExecutor) и TaskStore (здесь InMemoryTaskStore).
Он маршрутизирует полученные вызовы RPC A2A к соответствующим методам агента (таким как execute или cancel).
TaskStore используется DefaultRequestHandler для управления жизненным циклом задач, особенно для взаимодействий с состоянием, потоковой передачи и повторной подписки.
Даже если AgentExecutor прост, обработчику нужен хранилище задач.

- A2AStarletteApplication:
Класс A2AStarletteApplication создается с использованием agent_card и request_handler (называется http_handler в конструкторе).
agent_card очень важен, так как сервер будет предоставлять его по умолчанию в эндпоинте `/.well-known/agent.json`.
request_handler отвечает за обработку всех полученных вызовов метода A2A через взаимодействие с его AgentExecutor.

- uvicorn.run(server_app_builder.build(), ...):
A2AStarletteApplication имеет метод build() для построения реального приложения [Starlette](https://www.starlette.io/).
Это приложение затем запускается с использованием `uvicorn.run()`, делая ваш агент доступным через HTTP.
host='0.0.0.0' делает сервер доступным на всех сетевых интерфейсах вашей машины.
port=9999 указывает порт для прослушивания. Это соответствует url в AgentCard.

### Запуск

#### Запуск сервера
```bash
uv run python main.py
```
Вывод:
```bash
INFO:     Started server process [70842]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:10000 (Press CTRL+C to quit)
```

#### Запуск клиента
Код клиента выглядит следующим образом:
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
    """Вспомогательная функция для создания полезной нагрузки для отправки задачи."""
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
    """Вспомогательная функция для вывода JSON-представления ответа."""
    print(f'--- {description} ---')
    if hasattr(response, 'root'):
        print(f'{response.root.model_dump_json(exclude_none=True)}\n')
    else:
        print(f'{response.model_dump(mode="json", exclude_none=True)}\n')


async def run_single_turn_test(client: A2AClient) -> None:
    """Выполняет одношаговый тест без потоковой передачи."""

    send_payload = create_send_message_payload(
        text='how much is 100 USD in CAD?'
    )
    request = SendMessageRequest(params=MessageSendParams(**send_payload))

    print('--- Single Turn Request ---')
    # Отправить сообщение
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
    # запросить задачу
    get_request = GetTaskRequest(params=TaskQueryParams(id=task_id))
    get_response: GetTaskResponse = await client.get_task(get_request)
    print_json_response(get_response, 'Query Task Response')


async def run_streaming_test(client: A2AClient) -> None:
    """Выполняет одношаговый тест с потоковой передачей."""

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
    """Выполняет многошаговый тест без потоковой передачи."""
    print('--- Multi-Turn Request ---')
    # --- Первый шаг ---

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
        context_id = task.contextId  # Захватить ID контекста

        # --- Второй шаг (если требуется ввод) ---
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
    """Основная функция для выполнения тестов."""
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

Запуск:
```bash
uv run python test_client.py
```

Конец учебника. 

[https://a2aprotocol.ai/blog/a2a-sdk-currency-agent-tutorial-ru](https://a2aprotocol.ai/blog/a2a-sdk-currency-agent-tutorial-ru)