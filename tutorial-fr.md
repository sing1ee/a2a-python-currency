# Implémentation de CurrencyAgent en utilisant le SDK A2A Python

Le SDK officiel de Google [a2a-python](https://github.com/google/a2a-python) reçoit des mises à jour fréquentes, et notre tutoriel doit également être mis à jour. Dans cet article, nous allons implémenter un CurrencyAgent simple en utilisant la version `0.2.3` du SDK a2a-python.

## Contenu
- [Code Source](#code-source)
- [Préparation](#préparation)
- [Processus Détaillé](#processus-détaillé)
  - [Création du Projet](#création-du-projet)
  - [Création de l'Environnement Virtuel](#création-de-lenvironnement-virtuel)
  - [Ajout des Dépendances](#ajout-des-dépendances)
  - [Configuration des Variables d'Environnement](#configuration-des-variables-denvironnement)
  - [Création de l'Agent](#création-de-lagent)
    - [Fonctionnalités Principales](#1-fonctionnalités-principales)
    - [Architecture du Système](#2-architecture-du-système)
      - [System Prompt](#21-system-prompt)
      - [Méthodes Principales](#22-méthodes-principales)
    - [Flux de Travail](#3-flux-de-travail)
    - [Format de Réponse](#4-format-de-réponse)
    - [Gestion des Erreurs](#5-gestion-des-erreurs)
  - [Test de l'Agent](#test-de-lagent)
  - [Implémentation de AgentExecutor](#implémentation-de-agentexecutor)
  - [Implémentation de AgentServer](#implémentation-de-agentserver)
    - [AgentSkill](#agentskill)
    - [AgentCard](#agentcard)
    - [AgentServer](#agentserver-1)
  - [Exécution](#exécution)
    - [Exécution du Serveur](#exécution-du-serveur)
    - [Exécution du Client](#exécution-du-client)

## Code Source
Le code source du projet est disponible sur [a2a-python-currency](https://github.com/sing1ee/a2a-python-currency). N'hésitez pas à donner une étoile.

## Préparation
- [uv](https://docs.astral.sh/uv/#installation) 0.7.2, pour la gestion du projet
- Python 3.13+, cette version est nécessaire pour a2a-python
- apiKey et baseURL d'openai/openrouter. J'utilise [OpenRouter](https://openrouter.ai/), qui offre plus d'options de modèles.

## Processus Détaillé

### Création du Projet
```bash
uv init a2a-python-currency
cd a2a-python-currency
```

### Création de l'Environnement Virtuel
```bash
uv venv
source .venv/bin/activate
```

### Ajout des Dépendances
```bash
uv add a2a-sdk uvicorn dotenv click
```

### Configuration des Variables d'Environnement
```bash
echo OPENROUTER_API_KEY=your_api_key >> .env
echo OPENROUTER_BASE_URL=your_base_url >> .env

# exemple
OPENROUTER_API_KEY=votre_clé_api_OpenRouter
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

### Création de l'Agent
Le code complet est le suivant :
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
    """Agent de Conversion de Devises utilisant l'API OpenAI."""

    SYSTEM_PROMPT = """Vous êtes un assistant spécialisé dans les conversions de devises.
Votre seul but est d'utiliser l'outil 'get_exchange_rate' pour répondre aux questions sur les taux de change.
Si l'utilisateur pose des questions sur autre chose que la conversion de devises ou les taux de change,
dites poliment que vous ne pouvez pas aider sur ce sujet et que vous ne pouvez assister que pour les questions liées aux devises.
N'essayez pas de répondre à des questions non liées ou d'utiliser des outils à d'autres fins.

Vous avez accès à l'outil suivant :
- get_exchange_rate : Obtenir le taux de change actuel entre deux devises

Lors de l'utilisation de l'outil, répondez dans le format JSON suivant :
{
    "status": "completed" | "input_required" | "error",
    "message": "votre message de réponse"
}

Si vous devez utiliser l'outil, répondez avec :
{
    "status": "tool_use",
    "tool": "get_exchange_rate",
    "parameters": {
        "currency_from": "USD",
        "currency_to": "EUR",
        "currency_date": "latest"
    }
}
Note : Retournez la réponse au format JSON, seul le json est autorisé.
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
        """Obtenir le taux de change actuel entre les devises."""
        try:
            response = httpx.get(
                f'https://api.frankfurter.app/{currency_date}',
                params={'from': currency_from, 'to': currency_to},
            )
            response.raise_for_status()
            data = response.json()
            if 'rates' not in data:
                logger.error(f'rates not found in response: {data}')
                return {'error': 'Format de réponse API invalide.'}
            logger.info(f'API response: {data}')
            return data
        except httpx.HTTPError as e:
            logger.error(f'API request failed: {e}')
            return {'error': f'Requête API échouée : {e}'}
        except ValueError:
            logger.error('Invalid JSON response from API')
            return {'error': 'Réponse JSON invalide de l\'API.'}

    async def _call_openai(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Appeler l'API OpenAI via OpenRouter."""
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
        """Diffuser la réponse pour une requête donnée."""
        # Ajouter le message de l'utilisateur à l'historique de conversation
        self.conversation_history.append({"role": "user", "content": query})

        # Préparer les messages pour l'appel API
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + self.conversation_history

        # Obtenir la réponse d'OpenAI
        response = await self._call_openai(messages)
        assistant_message = response["choices"][0]["message"]["content"]
        print(assistant_message)
        try:
            # Essayer de parser la réponse en JSON
            parsed_response = json.loads(assistant_message)
            
            # Si c'est une demande d'utilisation d'outil
            if parsed_response.get("status") == "tool_use":
                tool_name = parsed_response["tool"]
                parameters = parsed_response["parameters"]
                
                # Émettre le statut d'utilisation de l'outil
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Recherche des taux de change..."
                }
                
                if tool_name == "get_exchange_rate":
                    # Émettre le statut de traitement
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": "Traitement des taux de change..."
                    }
                    
                    tool_result = await self.get_exchange_rate(**parameters)
                    
                    # Ajouter le résultat de l'outil à l'historique de conversation
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": json.dumps({"tool_result": tool_result})
                    })
                    
                    # Obtenir la réponse finale après l'utilisation de l'outil
                    final_response = await self._call_openai(messages)
                    final_message = final_response["choices"][0]["message"]["content"]
                    parsed_response = json.loads(final_message)

            # Ajouter la réponse de l'assistant à l'historique de conversation
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Émettre la réponse finale
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
                    "content": "Nous ne pouvons pas traiter votre demande pour le moment. Veuillez réessayer."
                }

        except json.JSONDecodeError:
            # Si la réponse n'est pas un JSON valide, retourner une erreur
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "Format de réponse invalide du modèle."
            } 
```

Analyse des fonctionnalités principales et de la logique d'implémentation :

#### 1. Fonctionnalités Principales
- Spécialisé dans le traitement des requêtes de conversion de devises et de taux de change
- Utilise l'API Frankfurter pour obtenir des données de taux de change en temps réel
- Traite les conversations via le modèle Claude 3.7 Sonnet via OpenRouter

#### 2. Architecture du Système
L'Agent se compose de plusieurs composants principaux :

##### 2.1 System Prompt
- Définit le but spécifique de l'Agent : traiter uniquement les requêtes de conversion de devises
- Définit le format de réponse : doit utiliser le format JSON
- Définit l'utilisation des outils : utiliser l'outil `get_exchange_rate` pour obtenir des informations sur les taux de change

##### 2.2 Méthodes Principales
1. **Méthode d'Initialisation `__init__`**
   - Configure la clé API et l'URL de base
   - Initialise l'historique des conversations

2. **Méthode de Requête de Taux de Change `get_exchange_rate`**
   - Paramètres : devise source, devise cible, date (la plus récente par défaut)
   - Appelle l'API Frankfurter pour obtenir les données de taux de change
   - Retourne les informations de taux de change au format JSON

3. **Méthode de Streaming `stream`**
   - Fournit la fonctionnalité de réponse en streaming
   - Retourne le statut de traitement et les résultats en temps réel
   - Prend en charge le retour d'état intermédiaire pour les appels d'outils

#### 3. Flux de Travail
1. **Réception de la Requête Utilisateur**
   - Ajoute le message de l'utilisateur à l'historique des conversations

2. **Traitement du Modèle**
   - Envoie le System Prompt et l'historique des conversations au modèle
   - Le modèle analyse s'il doit utiliser un outil

3. **Appel d'Outil (si nécessaire)**
   - Si le modèle décide d'utiliser un outil, retourne une demande d'appel d'outil
   - Exécute la requête de taux de change
   - Ajoute les résultats de la requête à l'historique des conversations

4. **Génération de la Réponse Finale**
   - Génère la réponse finale basée sur les résultats de l'appel d'outil
   - Retourne une réponse formatée en JSON

#### 4. Format de Réponse
Les réponses de l'Agent utilisent toujours le format JSON avec les états suivants :
- `completed` : tâche terminée
- `input_required` : entrée utilisateur requise
- `error` : une erreur s'est produite
- `tool_use` : utilisation d'outil nécessaire

#### 5. Gestion des Erreurs
- Inclut un mécanisme complet de gestion des erreurs
- Gère les échecs d'appels API
- Gère les erreurs de parsing JSON
- Gère les formats de réponse invalides

### Test de l'Agent
Le code de test est le suivant :
```python
import asyncio
import logging
from currency_agent import CurrencyAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    agent = CurrencyAgent()
    
    # Cas de test
    test_queries = [
        "What is the current exchange rate from USD to EUR?",
        "Can you tell me the exchange rate between GBP and JPY?",
        "What's the weather like today?",  # Doit être rejeté car non lié aux devises
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: {query}")
        async for response in agent.stream(query, "test_session"):
            logger.info(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main()) 
```

Si tout est correctement configuré, en particulier la configuration de l'environnement, vous devriez voir une sortie similaire à :
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

### Implémentation de AgentExecutor
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
    """Exemple d'AgentExecutor pour les devises."""

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
        # invoquer l'agent sous-jacent, en utilisant les résultats en streaming
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
                            description='Résultat de la requête à l\'agent.',
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

Analyse de la logique de ce code :
C'est une classe AgentExecutor appelée `CurrencyAgentExecutor` qui gère principalement les opérations d'agent liées aux devises. Analysons sa structure et ses fonctionnalités en détail :

La logique centrale pour traiter les requêtes A2A et générer des réponses/événements est implémentée par l'AgentExecutor. Le SDK A2A Python fournit une classe de base abstraite *a2a.server.agent_execution.AgentExecutor* que vous devez implémenter.

La classe AgentExecutor définit deux méthodes principales :
- `async def execute(self, context: RequestContext, event_queue: EventQueue)` : gère les requêtes reçues qui nécessitent des réponses ou des flux d'événements. Il traite l'entrée utilisateur (obtenue via le contexte) et utilise `event_queue` pour envoyer des objets Message, Task, TaskStatusUpdateEvent ou TaskArtifactUpdateEvent.
- `async def cancel(self, context: RequestContext, event_queue: EventQueue)` : gère les requêtes pour annuler les tâches en cours.

Le RequestContext fournit des informations sur la requête reçue, comme le message de l'utilisateur et tous les détails de tâche existants. L'EventQueue est utilisé par l'agent pour envoyer des événements au client.

### Implémentation de AgentServer

Le code :
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
    """Retourne la Carte Agent pour l'Agent de Devises."""
    capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
    skill = AgentSkill(
        id='convert_currency',
        name='Outil de Taux de Change',
        description='Aide avec les valeurs d\'échange entre différentes devises',
        tags=['conversion de devises', 'échange de devises'],
        examples=['What is exchange rate between USD and GBP?'],
    )
    return AgentCard(
        name='Agent de Devises',
        description='Aide avec les taux de change pour les devises',
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
L'AgentSkill décrit les compétences ou fonctionnalités spécifiques que l'agent peut exécuter. C'est un bloc de construction qui informe le client des types de tâches pour lesquels l'agent est adapté.
Attributs principaux de l'AgentSkill (définis dans a2a.types) :
- id : identifiant unique de la compétence
- name : nom lisible par l'homme
- description : explication plus détaillée de la fonctionnalité de la compétence
- tags : mots-clés pour la classification et la découverte
- examples : exemples de prompts ou cas d'utilisation
- inputModes / outputModes : types MIME supportés pour l'entrée et la sortie (par exemple, "text/plain", "application/json")

Cette compétence est très simple : gérer la conversion de devises, l'entrée et la sortie sont `text`, défini dans l'AgentCard.

#### AgentCard
L'AgentCard est un document JSON fourni par le serveur A2A, généralement situé à l'endpoint `.well-known/agent.json`. C'est comme une carte de visite numérique de l'agent.
Attributs principaux de l'AgentCard (définis dans a2a.types) :
- name, description, version : informations d'identité de base
- url : endpoint pour accéder au service A2A
- capabilities : spécifie les fonctionnalités A2A supportées, comme le streaming ou pushNotifications
- defaultInputModes / defaultOutputModes : types MIME par défaut de l'agent
- skills : liste des objets AgentSkill fournis par l'agent

#### AgentServer

- DefaultRequestHandler :
Le SDK fournit le DefaultRequestHandler. Ce gestionnaire reçoit une implémentation AgentExecutor (ici CurrencyAgentExecutor) et un TaskStore (ici InMemoryTaskStore).
Il route les appels RPC A2A reçus vers les méthodes appropriées de l'agent (comme execute ou cancel).
Le TaskStore est utilisé par le DefaultRequestHandler pour gérer le cycle de vie des tâches, en particulier pour les interactions avec état, le streaming et la ré-abonnement.
Même si l'AgentExecutor est simple, le gestionnaire a besoin d'un stockage de tâches.

- A2AStarletteApplication :
La classe A2AStarletteApplication est instanciée en utilisant agent_card et request_handler (appelé http_handler dans le constructeur).
L'agent_card est très important car le serveur l'exposera par défaut à l'endpoint `/.well-known/agent.json`.
Le request_handler est responsable du traitement de tous les appels de méthode A2A reçus via l'interaction avec son AgentExecutor.

- uvicorn.run(server_app_builder.build(), ...) :
A2AStarletteApplication a une méthode build() pour construire l'application [Starlette](https://www.starlette.io/) réelle.
Cette application est ensuite exécutée en utilisant `uvicorn.run()`, rendant votre agent accessible via HTTP.
host='0.0.0.0' rend le serveur accessible sur toutes les interfaces réseau de votre machine.
port=9999 spécifie le port à écouter. Cela correspond à l'url dans l'AgentCard.

### Exécution

#### Exécution du Serveur
```bash
uv run python main.py
```
Sortie :
```bash
INFO:     Started server process [70842]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:10000 (Press CTRL+C to quit)
```

#### Exécution du Client
Le code du client est le suivant :
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
    """Fonction d'aide pour créer le payload pour l'envoi d'une tâche."""
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
    """Fonction d'aide pour imprimer la représentation JSON d'une réponse."""
    print(f'--- {description} ---')
    if hasattr(response, 'root'):
        print(f'{response.root.model_dump_json(exclude_none=True)}\n')
    else:
        print(f'{response.model_dump(mode="json", exclude_none=True)}\n')


async def run_single_turn_test(client: A2AClient) -> None:
    """Exécute un test non-streaming à un seul tour."""

    send_payload = create_send_message_payload(
        text='how much is 100 USD in CAD?'
    )
    request = SendMessageRequest(params=MessageSendParams(**send_payload))

    print('--- Single Turn Request ---')
    # Envoyer Message
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
    # interroger la tâche
    get_request = GetTaskRequest(params=TaskQueryParams(id=task_id))
    get_response: GetTaskResponse = await client.get_task(get_request)
    print_json_response(get_response, 'Query Task Response')


async def run_streaming_test(client: A2AClient) -> None:
    """Exécute un test streaming à un seul tour."""

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
    """Exécute un test non-streaming à plusieurs tours."""
    print('--- Multi-Turn Request ---')
    # --- Premier Tour ---

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
        context_id = task.contextId  # Capturer l'ID de contexte

        # --- Second Tour (si entrée requise) ---
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
    """Fonction principale pour exécuter les tests."""
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

Exécution :
```bash
uv run python test_client.py
```

Fin du tutoriel. 

[https://a2aprotocol.ai/blog/a2a-sdk-currency-agent-tutorial-fr](https://a2aprotocol.ai/blog/a2a-sdk-currency-agent-tutorial-fr)