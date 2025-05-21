# Implementierung eines CurrencyAgent mit dem A2A Python SDK

Das offizielle Google SDK [a2a-python](https://github.com/google/a2a-python) wird häufig aktualisiert, und unser Tutorial muss ebenfalls aktualisiert werden. In diesem Artikel implementieren wir einen einfachen CurrencyAgent mit Version `0.2.3` des a2a-python SDK.

## Inhaltsverzeichnis
- [Quellcode](#quellcode)
- [Vorbereitung](#vorbereitung)
- [Detaillierte Schritte](#detaillierte-schritte)
  - [Projekterstellung](#projekterstellung)
  - [Erstellung der virtuellen Umgebung](#erstellung-der-virtuellen-umgebung)
  - [Hinzufügen von Abhängigkeiten](#hinzufügen-von-abhängigkeiten)
  - [Umgebungsvariablen einrichten](#umgebungsvariablen-einrichten)
  - [Agent erstellen](#agent-erstellen)
    - [Hauptfunktionen](#1-hauptfunktionen)
    - [Systemarchitektur](#2-systemarchitektur)
      - [System Prompt](#21-system-prompt)
      - [Hauptmethoden](#22-hauptmethoden)
    - [Workflow](#3-workflow)
    - [Antwortformat](#4-antwortformat)
    - [Fehlerbehandlung](#5-fehlerbehandlung)
  - [Agent testen](#agent-testen)
  - [AgentExecutor-Implementierung](#agentexecutor-implementierung)
  - [AgentServer-Implementierung](#agentserver-implementierung)
    - [AgentSkill](#agentskill)
    - [AgentCard](#agentcard)
    - [AgentServer](#agentserver-1)
  - [Ausführung](#ausführung)
    - [Server ausführen](#server-ausführen)
    - [Client ausführen](#client-ausführen)

## Quellcode
Der Quellcode des Projekts ist unter [a2a-python-currency](https://github.com/sing1ee/a2a-python-currency) verfügbar. Ein Stern ist willkommen.

## Vorbereitung
- [uv](https://docs.astral.sh/uv/#installation) 0.7.2, für die Projektverwaltung
- Python 3.13+, erforderliche Version für a2a-python
- openai/openrouter apiKey und baseURL. Ich verwende [OpenRouter](https://openrouter.ai/), das mehr Modelloptionen bietet.

## Detaillierte Schritte

### Projekterstellung
```bash
uv init a2a-python-currency
cd a2a-python-currency
```

### Erstellung der virtuellen Umgebung
```bash
uv venv
source .venv/bin/activate
```

### Hinzufügen von Abhängigkeiten
```bash
uv add a2a-sdk uvicorn dotenv click
```

### Umgebungsvariablen einrichten
```bash
echo OPENROUTER_API_KEY=your_api_key >> .env
echo OPENROUTER_BASE_URL=your_base_url >> .env

# Beispiel
OPENROUTER_API_KEY=Ihr OpenRouter API-Schlüssel
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

### Agent erstellen
Der vollständige Code lautet wie folgt:
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
    """Ein Währungsagent mit OpenAI API."""

    SYSTEM_PROMPT = """Sie sind ein Assistent, der sich auf Währungsumrechnungen spezialisiert hat.
Ihr einziger Zweck ist es, das 'tool_use'-Tool zu verwenden, um Fragen zu Wechselkursen zu beantworten.
Wenn der Benutzer Fragen stellt, die nichts mit Währungsumrechnungen oder Wechselkursen zu tun haben,
teilen Sie höflich mit, dass Sie bei diesem Thema nicht helfen können und nur bei währungsbezogenen Fragen helfen können.
Beantworten Sie keine nicht relevanten Fragen und versuchen Sie nicht, das Tool für andere Zwecke zu verwenden.

Sie haben Zugriff auf folgende Tools:
- get_exchange_rate: Aktuelle Wechselkurse zwischen zwei Währungen abrufen

Wenn Sie ein Tool verwenden, antworten Sie im folgenden JSON-Format:
{
    "status": "completed" | "input_required" | "error",
    "message": "Ihre Antwortnachricht"
}

Wenn Sie ein Tool verwenden müssen, antworten Sie wie folgt:
{
    "status": "tool_use",
    "tool": "get_exchange_rate",
    "parameters": {
        "currency_from": "USD",
        "currency_to": "EUR",
        "currency_date": "latest"
    }
}
Hinweis: Antworten Sie nur im JSON-Format. Nur JSON ist erlaubt.
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
        """Aktuelle Wechselkurse zwischen Währungen abrufen."""
        try:
            response = httpx.get(
                f'https://api.frankfurter.app/{currency_date}',
                params={'from': currency_from, 'to': currency_to},
            )
            response.raise_for_status()
            data = response.json()
            if 'rates' not in data:
                logger.error(f'rates not found in response: {data}')
                return {'error': 'Ungültiges API-Antwortformat.'}
            logger.info(f'API response: {data}')
            return data
        except httpx.HTTPError as e:
            logger.error(f'API request failed: {e}')
            return {'error': f'API-Anfrage fehlgeschlagen: {e}'}
        except ValueError:
            logger.error('Invalid JSON response from API')
            return {'error': 'Ungültige JSON-Antwort von der API.'}

    async def _call_openai(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """OpenAI API über OpenRouter aufrufen."""
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
        """Antworten auf die angegebene Abfrage streamen."""
        # Benutzernachricht zum Gesprächsverlauf hinzufügen
        self.conversation_history.append({"role": "user", "content": query})

        # Nachrichten für API-Aufruf vorbereiten
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + self.conversation_history

        # Antwort von OpenAI abrufen
        response = await self._call_openai(messages)
        assistant_message = response["choices"][0]["message"]["content"]
        print(assistant_message)
        try:
            # Antwort als JSON parsen
            parsed_response = json.loads(assistant_message)
            
            # Bei Tool-Verwendungsanfrage
            if parsed_response.get("status") == "tool_use":
                tool_name = parsed_response["tool"]
                parameters = parsed_response["parameters"]
                
                # Tool-Verwendungsstatus ausgeben
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Suche Wechselkurse..."
                }
                
                if tool_name == "get_exchange_rate":
                    # Verarbeitungsstatus ausgeben
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": "Verarbeite Wechselkurse..."
                    }
                    
                    tool_result = await self.get_exchange_rate(**parameters)
                    
                    # Tool-Ergebnis zum Gesprächsverlauf hinzufügen
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": json.dumps({"tool_result": tool_result})
                    })
                    
                    # Endgültige Antwort nach Tool-Verwendung abrufen
                    final_response = await self._call_openai(messages)
                    final_message = final_response["choices"][0]["message"]["content"]
                    parsed_response = json.loads(final_message)

            # Assistentenantwort zum Gesprächsverlauf hinzufügen
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Endgültige Antwort ausgeben
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
                    "content": "Anfrage kann derzeit nicht verarbeitet werden. Bitte versuchen Sie es erneut."
                }

        except json.JSONDecodeError:
            # Bei ungültigem JSON in der Antwort Fehler zurückgeben
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "Ungültiges Antwortformat vom Modell."
            } 
```

Analyse der Hauptfunktionen und Implementierungslogik:

#### 1. Hauptfunktionen
- Spezialisiert auf Währungsumrechnungen und Wechselkursanfragen
- Verwendet Frankfurter API für Echtzeit-Wechselkursdaten
- Verarbeitet Konversationen über OpenRouter mit dem Claude 3.7 Sonnet-Modell

#### 2. Systemarchitektur
Der Agent besteht aus folgenden Hauptkomponenten:

##### 2.1 System Prompt
- Definiert den spezifischen Zweck des Agents: Verarbeitung nur von Währungsumrechnungsanfragen
- Definiert das Antwortformat: Muss im JSON-Format sein
- Definiert die Tool-Verwendung: Verwendet `get_exchange_rate`-Tool für Wechselkursinformationen

##### 2.2 Hauptmethoden
1. **Initialisierungsmethode `__init__`**
   - Setzt API-Schlüssel und Basis-URL
   - Initialisiert Gesprächsverlauf

2. **Wechselkursabrufmethode `get_exchange_rate`**
   - Parameter: Quellwährung, Zielwährung, Datum (Standard: neueste)
   - Ruft Frankfurter API für Wechselkursdaten auf
   - Gibt Wechselkursinformationen im JSON-Format zurück

3. **Streamingmethode `stream`**
   - Bietet Streaming-Antwortfunktionalität
   - Gibt Verarbeitungsstatus und Ergebnisse in Echtzeit zurück
   - Unterstützt Zwischenstatus bei Tool-Aufrufen

#### 3. Workflow
1. **Benutzeranfrage empfangen**
   - Fügt Benutzernachricht zum Gesprächsverlauf hinzu

2. **Modellverarbeitung**
   - Sendet System Prompt und Gesprächsverlauf an das Modell
   - Modell analysiert, ob Tool-Verwendung erforderlich ist

3. **Tool-Aufruf (falls erforderlich)**
   - Wenn Modell Tool-Verwendung entscheidet, gibt Tool-Aufrufanfrage zurück
   - Führt Wechselkursanfrage aus
   - Fügt Anfrageergebnis zum Gesprächsverlauf hinzu

4. **Endgültige Antwort generieren**
   - Generiert endgültige Antwort basierend auf Tool-Aufrufergebnis
   - Gibt Antwort im JSON-Format zurück

#### 4. Antwortformat
Agent-Antworten verwenden immer JSON-Format mit folgenden Status:
- `completed`: Aufgabe abgeschlossen
- `input_required`: Benutzereingabe erforderlich
- `error`: Fehler aufgetreten
- `tool_use`: Tool-Verwendung erforderlich

#### 5. Fehlerbehandlung
- Enthält vollständigen Fehlerbehandlungsmechanismus
- Behandelt API-Aufruf-Fehler
- Behandelt JSON-Parsing-Fehler
- Behandelt ungültige Antwortformate

### Agent testen
Der Testcode lautet wie folgt:
```python
import asyncio
import logging
from currency_agent import CurrencyAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    agent = CurrencyAgent()
    
    # Testfälle
    test_queries = [
        "What is the current exchange rate from USD to EUR?",
        "Can you tell me the exchange rate between GBP and JPY?",
        "What's the weather like today?",  # Sollte abgelehnt werden, da nicht währungsbezogen
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: {query}")
        async for response in agent.stream(query, "test_session"):
            logger.info(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main()) 
```

Bei korrekter Umgebungskonfiguration, insbesondere wenn die Umgebungskonfiguration korrekt ist, sollte folgende Ausgabe erscheinen:
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

### AgentExecutor-Implementierung
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
    """Beispiel für einen AgentExecutor für Währungen."""

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
        # Basisagent mit Streaming-Ergebnissen aufrufen
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
                            description='Ergebnis der Anfrage an den Agenten.',
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

Analyse der Logik dieses Codes:
Dies ist eine AgentExecutor-Klasse namens `CurrencyAgentExecutor`, die hauptsächlich währungsbezogene Agentenoperationen verarbeitet. Lassen Sie uns ihre Struktur und Funktionalität im Detail analysieren:

Die zentrale Logik für die Verarbeitung von A2A-Anfragen und die Generierung von Antworten/Ereignissen wird über den AgentExecutor implementiert. Das A2A Python SDK stellt eine abstrakte Basisklasse *a2a.server.agent_execution.AgentExecutor* bereit, die implementiert werden muss.

Die AgentExecutor-Klasse definiert zwei Hauptmethoden:
- `async def execute(self, context: RequestContext, event_queue: EventQueue)`: Verarbeitet eingehende Anfragen, die Antworten oder Ereignisströme erfordern. Verarbeitet Benutzereingaben aus dem Kontext und sendet Message-, Task-, TaskStatusUpdateEvent- oder TaskArtifactUpdateEvent-Objekte über die `event_queue`.
- `async def cancel(self, context: RequestContext, event_queue: EventQueue)`: Verarbeitet Anfragen zum Abbrechen des aktuellen Tasks.

RequestContext stellt Informationen über die eingehende Anfrage bereit, wie Benutzernachrichten oder bestehende Task-Details. EventQueue wird vom Agenten verwendet, um Ereignisse an den Client zu senden.

### AgentServer-Implementierung

Code:
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
    """Gibt die Agentenkarte für den Währungsagenten zurück."""
    capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
    skill = AgentSkill(
        id='convert_currency',
        name='Wechselkurs-Tool',
        description='Hilft bei Wechselkursen zwischen verschiedenen Währungen',
        tags=['Währungsumrechnung', 'Wechselkurs'],
        examples=['What is exchange rate between USD and GBP?'],
    )
    return AgentCard(
        name='Währungsagent',
        description='Hilft bei Wechselkursen',
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
AgentSkill beschreibt die Fähigkeiten oder spezifischen Funktionen, die ein Agent ausführen kann. Es ist ein Baustein, der Clients mitteilt, für welche Arten von Aufgaben der Agent geeignet ist.
Hauptattribute von AgentSkill (definiert in a2a.types):
- id: Eindeutiger Bezeichner für die Fähigkeit
- name: Menschenlesbarer Name
- description: Detailliertere Beschreibung der Funktionalität
- tags: Schlüsselwörter für Klassifizierung und Entdeckung
- examples: Beispiele für Prompts oder Anwendungsfälle
- inputModes / outputModes: Unterstützte MIME-Typen für Ein- und Ausgabe (z.B. "text/plain", "application/json")

Diese Fähigkeit ist sehr einfach: Verarbeitung von Währungsumrechnungen, Ein- und Ausgabe ist `text`, definiert in AgentCard.

#### AgentCard
AgentCard ist ein JSON-Dokument, das vom A2A-Server bereitgestellt wird, typischerweise am Endpunkt `.well-known/agent.json`. Es ist wie eine digitale Visitenkarte für den Agenten.
Hauptattribute von AgentCard (definiert in a2a.types):
- name, description, version: Grundlegende Identifikationsinformationen
- url: Endpunkt für den Zugriff auf den A2A-Dienst
- capabilities: Spezifiziert unterstützte A2A-Funktionen wie streaming und pushNotifications
- defaultInputModes / defaultOutputModes: Standard-MIME-Typen des Agenten
- skills: Liste von AgentSkill-Objekten, die der Agent bereitstellt

#### AgentServer

- DefaultRequestHandler:
Das SDK stellt DefaultRequestHandler bereit. Dieser Handler nimmt eine AgentExecutor-Implementierung (hier CurrencyAgentExecutor) und einen TaskStore (hier InMemoryTaskStore) entgegen.
Leitet eingehende A2A RPC-Aufrufe an die entsprechenden Methoden des Agenten wie execute oder cancel weiter.
TaskStore wird vom DefaultRequestHandler verwendet, um den Task-Lebenszyklus zu verwalten, insbesondere für zustandsbehaftete Interaktionen, Streaming und Wiederverbindung.
Auch wenn der AgentExecutor einfach ist, benötigt der Handler einen Task- Store.

- A2AStarletteApplication:
Die A2AStarletteApplication-Klasse wird mit agent_card und request_handler (im Konstruktor als http_handler bezeichnet) erstellt.
agent_card ist sehr wichtig. Der Server stellt dies standardmäßig am Endpunkt `/.well-known/agent.json` bereit.
request_handler ist für die Verarbeitung aller eingehenden A2A-Methodenaufrufe über seinen AgentExecutor verantwortlich.

- uvicorn.run(server_app_builder.build(), ...):
A2AStarletteApplication hat eine build()-Methode, die die eigentliche [Starlette](https://www.starlette.io/)-Anwendung erstellt.
Diese Anwendung wird mit `uvicorn.run()` ausgeführt, wodurch der Agent über HTTP zugänglich wird.
host='0.0.0.0' macht den Server auf allen Netzwerkschnittstellen der Maschine zugänglich.
port=9999 gibt den zu überwachenden Port an. Dies entspricht der url in AgentCard.

### Ausführung

#### Server ausführen
```bash
uv run python main.py
```
Ausgabe:
```bash
INFO:     Started server process [70842]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:10000 (Press CTRL+C to quit)
```

#### Client ausführen
Der Client-Code lautet wie folgt:
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
    """Hilfsfunktion zum Erstellen der Payload für Task-Sendungen."""
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
    """Hilfsfunktion zum Ausgeben der JSON-Darstellung der Antwort."""
    print(f'--- {description} ---')
    if hasattr(response, 'root'):
        print(f'{response.root.model_dump_json(exclude_none=True)}\n')
    else:
        print(f'{response.model_dump(mode="json", exclude_none=True)}\n')


async def run_single_turn_test(client: A2AClient) -> None:
    """Führt einen Single-Turn-Test ohne Streaming durch."""

    send_payload = create_send_message_payload(
        text='how much is 100 USD in CAD?'
    )
    request = SendMessageRequest(params=MessageSendParams(**send_payload))

    print('--- Single Turn Request ---')
    # Nachricht senden
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
    # Task abfragen
    get_request = GetTaskRequest(params=TaskQueryParams(id=task_id))
    get_response: GetTaskResponse = await client.get_task(get_request)
    print_json_response(get_response, 'Query Task Response')


async def run_streaming_test(client: A2AClient) -> None:
    """Führt einen Single-Turn-Test mit Streaming durch."""

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
    """Führt einen Multi-Turn-Test ohne Streaming durch."""
    print('--- Multi-Turn Request ---')
    # --- Erster Turn ---

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
        context_id = task.contextId  # Kontext-ID erfassen

        # --- Zweiter Turn (bei erforderlicher Eingabe) ---
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
    """Hauptfunktion zum Ausführen der Tests."""
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

Ausführung:
```bash
uv run python test_client.py
```

Tutorial beendet. 