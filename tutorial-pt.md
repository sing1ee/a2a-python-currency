# Implementando CurrencyAgent usando o A2A Python SDK

O SDK oficial do Google [a2a-python](https://github.com/google/a2a-python) tem recebido atualizações frequentes, e nosso tutorial também precisa ser atualizado. Neste artigo, implementaremos um CurrencyAgent simples usando a versão `0.2.3` do a2a-python SDK.

## Conteúdo
- [Código Fonte](#código-fonte)
- [Preparação](#preparação)
- [Processo Detalhado](#processo-detalhado)
  - [Criando o Projeto](#criando-o-projeto)
  - [Criando o Ambiente Virtual](#criando-o-ambiente-virtual)
  - [Adicionando Dependências](#adicionando-dependências)
  - [Configurando Variáveis de Ambiente](#configurando-variáveis-de-ambiente)
  - [Criando o Agent](#criando-o-agent)
    - [Funcionalidades Principais](#1-funcionalidades-principais)
    - [Arquitetura do Sistema](#2-arquitetura-do-sistema)
      - [System Prompt](#21-system-prompt)
      - [Métodos Principais](#22-métodos-principais)
    - [Fluxo de Trabalho](#3-fluxo-de-trabalho)
    - [Formato de Resposta](#4-formato-de-resposta)
    - [Tratamento de Erros](#5-tratamento-de-erros)
  - [Testando o Agent](#testando-o-agent)
  - [Implementando AgentExecutor](#implementando-agentexecutor)
  - [Implementando AgentServer](#implementando-agentserver)
    - [AgentSkill](#agentskill)
    - [AgentCard](#agentcard)
    - [AgentServer](#agentserver-1)
  - [Execução](#execução)
    - [Executando o Servidor](#executando-o-servidor)
    - [Executando o Cliente](#executando-o-cliente)

## Código Fonte
O código fonte do projeto está disponível em [a2a-python-currency](https://github.com/sing1ee/a2a-python-currency). Sinta-se à vontade para dar uma estrela.

## Preparação
- [uv](https://docs.astral.sh/uv/#installation) 0.7.2, para gerenciamento do projeto
- Python 3.13+, esta versão é necessária para o a2a-python
- apiKey e baseURL do openai/openrouter. Estou usando o [OpenRouter](https://openrouter.ai/), que oferece mais opções de modelos.

## Processo Detalhado

### Criando o Projeto
```bash
uv init a2a-python-currency
cd a2a-python-currency
```

### Criando o Ambiente Virtual
```bash
uv venv
source .venv/bin/activate
```

### Adicionando Dependências
```bash
uv add a2a-sdk uvicorn dotenv click
```

### Configurando Variáveis de Ambiente
```bash
echo OPENROUTER_API_KEY=your_api_key >> .env
echo OPENROUTER_BASE_URL=your_base_url >> .env

# exemplo
OPENROUTER_API_KEY=sua_chave_api_do_OpenRouter
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

### Criando o Agent
O código completo é o seguinte:
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

Análise das funcionalidades principais e lógica de implementação:

#### 1. Funcionalidades Principais
- Especializado em processar consultas de conversão de moeda e taxas de câmbio
- Utiliza a API Frankfurter para obter dados de taxas de câmbio em tempo real
- Processa conversas através do modelo Claude 3.7 Sonnet via OpenRouter

#### 2. Arquitetura do Sistema
O Agent consiste em vários componentes principais:

##### 2.1 System Prompt
- Define o propósito específico do Agent: processar apenas consultas de conversão de moeda
- Define o formato de resposta: deve usar formato JSON
- Define o uso de ferramentas: usar a ferramenta `get_exchange_rate` para obter informações de taxas de câmbio

##### 2.2 Métodos Principais
1. **Método de Inicialização `__init__`**
   - Configura a chave API e URL base
   - Inicializa o histórico de conversas

2. **Método de Consulta de Taxa de Câmbio `get_exchange_rate`**
   - Parâmetros: moeda de origem, moeda de destino, data (mais recente por padrão)
   - Chama a API Frankfurter para obter dados de taxas de câmbio
   - Retorna informações de taxa de câmbio em formato JSON

3. **Método de Streaming `stream`**
   - Fornece funcionalidade de resposta em streaming
   - Retorna status de processamento e resultados em tempo real
   - Suporta feedback de estado intermediário para chamadas de ferramentas

#### 3. Fluxo de Trabalho
1. **Recebimento da Consulta do Usuário**
   - Adiciona a mensagem do usuário ao histórico de conversas

2. **Processamento do Modelo**
   - Envia o System Prompt e histórico de conversas para o modelo
   - Modelo analisa se precisa usar uma ferramenta

3. **Chamada de Ferramenta (se necessário)**
   - Se o modelo decide usar uma ferramenta, retorna uma solicitação de chamada de ferramenta
   - Executa a consulta de taxa de câmbio
   - Adiciona os resultados da consulta ao histórico de conversas

4. **Geração da Resposta Final**
   - Gera a resposta final com base nos resultados da chamada da ferramenta
   - Retorna uma resposta JSON formatada

#### 4. Formato de Resposta
As respostas do Agent sempre usam formato JSON com os seguintes estados:
- `completed`: tarefa concluída
- `input_required`: entrada do usuário necessária
- `error`: ocorreu um erro
- `tool_use`: uso de ferramenta necessário

#### 5. Tratamento de Erros
- Inclui mecanismo completo de tratamento de erros
- Trata falhas de chamadas de API
- Trata erros de parsing JSON
- Trata formatos de resposta inválidos

### Testando o Agent
O código de teste é o seguinte:
```python
import asyncio
import logging
from currency_agent import CurrencyAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    agent = CurrencyAgent()
    
    # Casos de teste
    test_queries = [
        "What is the current exchange rate from USD to EUR?",
        "Can you tell me the exchange rate between GBP and JPY?",
        "What's the weather like today?",  # Deve ser rejeitado por não ser relacionado a moeda
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: {query}")
        async for response in agent.stream(query, "test_session"):
            logger.info(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main()) 
```

Se tudo estiver configurado corretamente, especialmente a configuração do ambiente, você deve ver uma saída semelhante a:
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

### Implementando AgentExecutor
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

Análise da lógica deste código:
Esta é uma classe AgentExecutor chamada `CurrencyAgentExecutor` que lida principalmente com operações de agente relacionadas a moeda. Vamos analisar sua estrutura e funcionalidades em detalhes:

A lógica central para processar solicitações A2A e gerar respostas/eventos é implementada pelo AgentExecutor. O A2A Python SDK fornece uma classe base abstrata *a2a.server.agent_execution.AgentExecutor* que você precisa implementar.

A classe AgentExecutor define dois métodos principais:
- `async def execute(self, context: RequestContext, event_queue: EventQueue)`: lida com solicitações recebidas que requerem respostas ou fluxos de eventos. Ele processa a entrada do usuário (obtida através do contexto) e usa `event_queue` para enviar objetos Message, Task, TaskStatusUpdateEvent ou TaskArtifactUpdateEvent.
- `async def cancel(self, context: RequestContext, event_queue: EventQueue)`: lida com solicitações para cancelar tarefas em andamento.

O RequestContext fornece informações sobre a solicitação recebida, como a mensagem do usuário e quaisquer detalhes de tarefa existentes. O EventQueue é usado pelo agente para enviar eventos ao cliente.

### Implementando AgentServer

O código:
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

#### AgentSkill
O AgentSkill descreve as habilidades ou funcionalidades específicas que o agente pode executar. É um bloco de construção que informa ao cliente quais tipos de tarefas o agente é adequado para executar.
Atributos principais do AgentSkill (definidos em a2a.types):
- id: identificador único da habilidade
- name: nome legível para humanos
- description: explicação mais detalhada da funcionalidade da habilidade
- tags: palavras-chave para classificação e descoberta
- examples: exemplos de prompts ou casos de uso
- inputModes / outputModes: tipos MIME suportados para entrada e saída (por exemplo, "text/plain", "application/json")

Esta habilidade é muito simples: lidar com conversão de moeda, entrada e saída são `text`, definido no AgentCard.

#### AgentCard
O AgentCard é um documento JSON fornecido pelo servidor A2A, geralmente localizado no endpoint `.well-known/agent.json`. É como um cartão de visita digital do agente.
Atributos principais do AgentCard (definidos em a2a.types):
- name, description, version: informações básicas de identidade
- url: endpoint para acessar o serviço A2A
- capabilities: especifica recursos A2A suportados, como streaming ou pushNotifications
- defaultInputModes / defaultOutputModes: tipos MIME padrão do agente
- skills: lista de objetos AgentSkill fornecidos pelo agente

#### AgentServer

- DefaultRequestHandler:
O SDK fornece o DefaultRequestHandler. Este manipulador recebe uma implementação AgentExecutor (aqui CurrencyAgentExecutor) e um TaskStore (aqui InMemoryTaskStore).
Ele roteia chamadas RPC A2A recebidas para os métodos apropriados no agente (como execute ou cancel).
O TaskStore é usado pelo DefaultRequestHandler para gerenciar o ciclo de vida das tarefas, especialmente para interações com estado, streaming e re-subscrição.
Mesmo que o AgentExecutor seja simples, o manipulador precisa de um armazenamento de tarefas.

- A2AStarletteApplication:
A classe A2AStarletteApplication é instanciada usando agent_card e request_handler (chamado http_handler no construtor).
O agent_card é muito importante porque o servidor o exporá por padrão no endpoint `/.well-known/agent.json`.
O request_handler é responsável por processar todas as chamadas de método A2A recebidas através da interação com seu AgentExecutor.

- uvicorn.run(server_app_builder.build(), ...):
A2AStarletteApplication tem um método build() para construir o aplicativo [Starlette](https://www.starlette.io/) real.
Este aplicativo é então executado usando `uvicorn.run()`, tornando seu agente acessível via HTTP.
host='0.0.0.0' torna o servidor acessível em todas as interfaces de rede em sua máquina.
port=9999 especifica a porta para escutar. Isso corresponde à url no AgentCard.

### Execução

#### Executando o Servidor
```bash
uv run python main.py
```
Saída:
```bash
INFO:     Started server process [70842]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:10000 (Press CTRL+C to quit)
```

#### Executando o Cliente
O código do cliente é o seguinte:
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

Executando:
```bash
uv run python test_client.py
```

Fim do tutorial. 

[https://a2aprotocol.ai/blog/a2a-sdk-currency-agent-tutorial-pt](https://a2aprotocol.ai/blog/a2a-sdk-currency-agent-tutorial-pt)