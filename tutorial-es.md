# Implementación de CurrencyAgent con A2A Python SDK

El SDK oficial [a2a-python](https://github.com/google/a2a-python) de Google se ha actualizado frecuentemente, y nuestro tutorial necesita mantenerse al día. En este artículo, implementaremos un CurrencyAgent simple utilizando la versión `0.2.3` del SDK a2a-python.

## Tabla de Contenidos
- [Código Fuente](#código-fuente)
- [Prerrequisitos](#prerrequisitos)
- [Proceso Detallado](#proceso-detallado)
  - [Crear Proyecto](#crear-proyecto)
  - [Crear Entorno Virtual](#crear-entorno-virtual)
  - [Agregar Dependencias](#agregar-dependencias)
  - [Configurar Variables de Entorno](#configurar-variables-de-entorno)
  - [Crear Agent](#crear-agent)
    - [Funcionalidad Principal](#1-funcionalidad-principal)
    - [Arquitectura del Sistema](#2-arquitectura-del-sistema)
      - [System Prompt](#21-system-prompt)
      - [Métodos Principales](#22-métodos-principales)
    - [Flujo de Trabajo](#3-flujo-de-trabajo)
    - [Formato de Respuesta](#4-formato-de-respuesta)
    - [Manejo de Errores](#5-manejo-de-errores)
  - [Probar Agent](#probar-agent)
  - [Implementar AgentExecutor](#implementar-agentexecutor)
  - [Implementar AgentServer](#implementar-agentserver)
    - [AgentSkill](#agentskill)
    - [AgentCard](#agentcard)
    - [AgentServer](#agentserver-1)
  - [Ejecución](#ejecución)
    - [Ejecutar Servidor](#ejecutar-servidor)
    - [Ejecutar Cliente](#ejecutar-cliente)

## Código Fuente
El código fuente del proyecto está disponible en [a2a-python-currency](https://github.com/sing1ee/a2a-python-currency). ¡Las estrellas son bienvenidas!

## Prerrequisitos
- [uv](https://docs.astral.sh/uv/#installation) 0.7.2, para gestión de proyectos
- Python 3.13+, requerido por a2a-python
- apiKey y baseURL de openai/openrouter. Estoy usando [OpenRouter](https://openrouter.ai/), que ofrece más opciones de modelos.

## Proceso Detallado

### Crear Proyecto
```bash
uv init a2a-python-currency
cd a2a-python-currency
```

### Crear Entorno Virtual
```bash
uv venv
source .venv/bin/activate
```

### Agregar Dependencias
```bash
uv add a2a-sdk uvicorn dotenv click
```

### Configurar Variables de Entorno
```bash
echo OPENROUTER_API_KEY=your_api_key >> .env
echo OPENROUTER_BASE_URL=your_base_url >> .env

# ejemplo 
OPENROUTER_API_KEY=tu_clave_API_de_OpenRouter
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

### Crear Agent
Aquí está el código completo:
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

Analicemos su funcionalidad principal y lógica de implementación:

#### 1. Funcionalidad Principal
- Especializado en manejar consultas de conversión de moneda y tasas de cambio
- Utiliza la API de Frankfurter para obtener datos de tasas de cambio en tiempo real
- Procesa conversaciones a través de OpenRouter usando el modelo Claude 3.7 Sonnet

#### 2. Arquitectura del Sistema
El Agent consta de varios componentes principales:

##### 2.1 System Prompt
- Define el propósito especializado del Agent: manejar solo consultas de conversión de moneda
- Especifica el formato de respuesta: debe usar formato JSON
- Define el uso de herramientas: utiliza la herramienta `get_exchange_rate` para obtener información de tasas de cambio

##### 2.2 Métodos Principales
1. **Método de Inicialización `__init__`**
   - Configura la clave API y URL base
   - Inicializa el historial de conversación

2. **Método de Consulta de Tasa de Cambio `get_exchange_rate`**
   - Parámetros: moneda origen, moneda destino, fecha (por defecto es la más reciente)
   - Llama a la API de Frankfurter para obtener datos de tasa de cambio
   - Devuelve información de tasa de cambio en formato JSON

3. **Método de Streaming `stream`**
   - Proporciona funcionalidad de respuesta en streaming
   - Devuelve estado de procesamiento y resultados en tiempo real
   - Soporta retroalimentación de estado intermedio para llamadas a herramientas

#### 3. Flujo de Trabajo
1. **Recibir Consulta del Usuario**
   - Agregar mensaje del usuario al historial de conversación

2. **Procesamiento del Modelo**
   - Enviar system prompt e historial de conversación al modelo
   - El modelo analiza si se necesita usar una herramienta

3. **Llamada a Herramienta (si es necesario)**
   - Si el modelo decide usar una herramienta, devuelve una solicitud de llamada a herramienta
   - Ejecutar consulta de tasa de cambio
   - Agregar resultados de la consulta al historial de conversación

4. **Generar Respuesta Final**
   - Generar respuesta final basada en los resultados de la llamada a la herramienta
   - Devolver respuesta JSON formateada

#### 4. Formato de Respuesta
Las respuestas del Agent siempre usan formato JSON con los siguientes estados:
- `completed`: Tarea completada
- `input_required`: Se necesita entrada del usuario
- `error`: Ocurrió un error
- `tool_use`: Se necesita usar una herramienta

#### 5. Manejo de Errores
- Incluye un mecanismo completo de manejo de errores
- Maneja fallos en llamadas a API
- Maneja errores de análisis JSON
- Maneja formatos de respuesta inválidos

### Probar Agent
Aquí está el código de prueba:
```python
import asyncio
import logging
from currency_agent import CurrencyAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    agent = CurrencyAgent()
    
    # Casos de prueba
    test_queries = [
        "What is the current exchange rate from USD to EUR?",
        "Can you tell me the exchange rate between GBP and JPY?",
        "What's the weather like today?",  # Esto debería ser rechazado por no estar relacionado con moneda
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: {query}")
        async for response in agent.stream(query, "test_session"):
            logger.info(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main()) 
```

Si todo está configurado correctamente, especialmente la configuración del entorno, deberías ver una salida similar a esta:
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

### Implementar AgentExecutor
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

Analicemos la lógica de este código:
Esta es una clase AgentExecutor llamada `CurrencyAgentExecutor` que maneja principalmente operaciones de agentes relacionadas con moneda. Analicemos su estructura y funcionalidad en detalle:

La lógica principal para manejar solicitudes y generar respuestas/eventos en agentes A2A está implementada por AgentExecutor. El SDK A2A Python proporciona una clase base abstracta *a2a.server.agent_execution.AgentExecutor* que necesitas implementar.

La clase AgentExecutor define dos métodos principales:
- `async def execute(self, context: RequestContext, event_queue: EventQueue)`: Maneja solicitudes entrantes que necesitan respuestas o flujos de eventos. Procesa la entrada del usuario (obtenida a través del contexto) y usa `event_queue` para enviar objetos Message, Task, TaskStatusUpdateEvent o TaskArtifactUpdateEvent.
- `async def cancel(self, context: RequestContext, event_queue: EventQueue)`: Maneja solicitudes para cancelar tareas en curso.

RequestContext proporciona información sobre la solicitud entrante, como el mensaje del usuario y cualquier detalle de tarea existente. EventQueue es usado por el ejecutor para enviar eventos al cliente.

### Implementar AgentServer

Código:
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
AgentSkill describe capacidades o funciones específicas que el agente puede realizar. Es un bloque de construcción que le dice a los clientes qué tipos de tareas es adecuado para el agente.
Atributos clave de AgentSkill (definidos en a2a.types):
- id: Identificador único para la habilidad
- name: Nombre legible por humanos
- description: Explicación más detallada de la funcionalidad de la habilidad
- tags: Palabras clave para categorización y descubrimiento
- examples: Ejemplos de prompts o casos de uso
- inputModes / outputModes: Tipos MIME de entrada y salida soportados (por ejemplo, "text/plain", "application/json")

Esta habilidad es muy simple: manejar conversión de moneda, con entrada y salida siendo `text`, definido en el AgentCard.

#### AgentCard
AgentCard es un documento JSON proporcionado por el servidor A2A, típicamente ubicado en el endpoint `.well-known/agent.json`. Es como una tarjeta de presentación digital para el agente.
Atributos clave de AgentCard (definidos en a2a.types):
- name, description, version: Información básica de identidad
- url: Endpoint donde se puede acceder al servicio A2A
- capabilities: Especifica características A2A soportadas, como streaming o pushNotifications
- defaultInputModes / defaultOutputModes: Tipos MIME predeterminados para el agente
- skills: Lista de objetos AgentSkill proporcionados por el agente

#### AgentServer

- DefaultRequestHandler:
El SDK proporciona DefaultRequestHandler. Este manejador toma una implementación de AgentExecutor (aquí CurrencyAgentExecutor) y un TaskStore (aquí InMemoryTaskStore).
Enruta las llamadas RPC A2A entrantes a los métodos apropiados en el ejecutor (como execute o cancel).
TaskStore es usado por DefaultRequestHandler para gestionar los ciclos de vida de las tareas, especialmente para interacciones con estado, streaming y resubscripciones.
Incluso si el AgentExecutor es simple, el manejador necesita un almacén de tareas.

- A2AStarletteApplication:
La clase A2AStarletteApplication se instancia con agent_card y request_handler (llamado http_handler en su constructor).
agent_card es crucial porque el servidor lo expondrá por defecto en el endpoint `/.well-known/agent.json`.
request_handler es responsable de manejar todas las llamadas de método A2A entrantes interactuando con tu AgentExecutor.

- uvicorn.run(server_app_builder.build(), ...):
A2AStarletteApplication tiene un método build() que construye la aplicación [Starlette](https://www.starlette.io/) real.
Luego usa `uvicorn.run()` para ejecutar esta aplicación, haciendo que tu agente sea accesible vía HTTP.
host='0.0.0.0' hace que el servidor sea accesible en todas las interfaces de red de tu máquina.
port=9999 especifica el puerto en el que escuchar. Esto coincide con la url en el AgentCard.

### Ejecución

#### Ejecutar Servidor
```bash
uv run python main.py
```
Salida:
```bash
INFO:     Started server process [70842]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:10000 (Press CTRL+C to quit)
```

#### Ejecutar Cliente
Aquí está el código del cliente:
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

Ejecútalo así:
```bash
uv run python test_client.py
```

Fin del tutorial. 

[https://a2aprotocol.ai/blog/a2a-sdk-currency-agent-tutorial-es](https://a2aprotocol.ai/blog/a2a-sdk-currency-agent-tutorial-es)