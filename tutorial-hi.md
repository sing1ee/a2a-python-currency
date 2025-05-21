# A2A Python SDK का उपयोग करके CurrencyAgent का कार्यान्वयन

Google का आधिकारिक [a2a-python](https://github.com/google/a2a-python) SDK हाल ही में लगातार अपडेट हो रहा है, और हमारे ट्यूटोरियल को भी अपडेट करने की आवश्यकता है। इस लेख में, हम a2a-python SDK के `0.2.3` संस्करण का उपयोग करके एक सरल CurrencyAgent का कार्यान्वयन करेंगे।

## विषय सूची
- [स्रोत कोड](#स्रोत-कोड)
- [आवश्यकताएँ](#आवश्यकताएँ)
- [विस्तृत प्रक्रिया](#विस्तृत-प्रक्रिया)
  - [प्रोजेक्ट बनाना](#प्रोजेक्ट-बनाना)
  - [वर्चुअल एनवायरनमेंट बनाना](#वर्चुअल-एनवायरनमेंट-बनाना)
  - [डिपेंडेंसी जोड़ना](#डिपेंडेंसी-जोड़ना)
  - [एनवायरनमेंट वेरिएबल्स कॉन्फ़िगर करना](#एनवायरनमेंट-वेरिएबल्स-कॉन्फ़िगर-करना)
  - [Agent बनाना](#agent-बनाना)
    - [मुख्य कार्यक्षमता](#1-मुख्य-कार्यक्षमता)
    - [सिस्टम आर्किटेक्चर](#2-सिस्टम-आर्किटेक्चर)
      - [सिस्टम प्रॉम्प्ट](#21-सिस्टम-प्रॉम्प्ट)
      - [मुख्य विधियाँ](#22-मुख्य-विधियाँ)
    - [कार्य प्रवाह](#3-कार्य-प्रवाह)
    - [प्रतिक्रिया प्रारूप](#4-प्रतिक्रिया-प्रारूप)
    - [त्रुटि प्रबंधन](#5-त्रुटि-प्रबंधन)
  - [Agent का परीक्षण](#agent-का-परीक्षण)
  - [AgentExecutor का कार्यान्वयन](#agentexecutor-का-कार्यान्वयन)
  - [AgentServer का कार्यान्वयन](#agentserver-का-कार्यान्वयन)
    - [AgentSkill](#agentskill)
    - [AgentCard](#agentcard)
    - [AgentServer](#agentserver-1)
  - [चलाना](#चलाना)
    - [सर्वर चलाना](#सर्वर-चलाना)
    - [क्लाइंट चलाना](#क्लाइंट-चलाना)

## स्रोत कोड
प्रोजेक्ट का स्रोत कोड [a2a-python-currency](https://github.com/sing1ee/a2a-python-currency) पर उपलब्ध है। स्टार करने के लिए आपका स्वागत है।

## आवश्यकताएँ
- [uv](https://docs.astral.sh/uv/#installation) 0.7.2, प्रोजेक्ट प्रबंधन के लिए
- Python 3.13+, a2a-python की आवश्यकता
- openai/openrouter का apiKey और baseURL। मैं [OpenRouter](https://openrouter.ai/) का उपयोग कर रहा हूँ, जो अधिक मॉडल विकल्प प्रदान करता है।

## विस्तृत प्रक्रिया

### प्रोजेक्ट बनाना
```bash
uv init a2a-python-currency
cd a2a-python-currency
```

### वर्चुअल एनवायरनमेंट बनाना
```bash
uv venv
source .venv/bin/activate
```

### डिपेंडेंसी जोड़ना
```bash
uv add a2a-sdk uvicorn dotenv click
```

### एनवायरनमेंट वेरिएबल्स कॉन्फ़िगर करना
```bash
echo OPENROUTER_API_KEY=your_api_key >> .env
echo OPENROUTER_BASE_URL=your_base_url >> .env

# उदाहरण 
OPENROUTER_API_KEY=आपकी_OpenRouter_API_कुंजी
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

### Agent बनाना
पूरा कोड इस प्रकार है:
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

इसकी मुख्य कार्यक्षमता और कार्यान्वयन तर्क का विश्लेषण करें:

#### 1. मुख्य कार्यक्षमता
- मुद्रा रूपांतरण और विनिमय दर प्रश्नों को संभालने के लिए विशेष
- फ्रैंकफर्टर API का उपयोग करके वास्तविक समय की विनिमय दर डेटा प्राप्त करना
- OpenRouter के माध्यम से Claude 3.7 Sonnet मॉडल का उपयोग करके बातचीत प्रसंस्करण

#### 2. सिस्टम आर्किटेक्चर
Agent में कई मुख्य घटक होते हैं:

##### 2.1 सिस्टम प्रॉम्प्ट
- Agent का विशेष उद्देश्य परिभाषित करता है: केवल मुद्रा रूपांतरण प्रश्नों को संभालना
- प्रतिक्रिया प्रारूप निर्दिष्ट करता है: JSON प्रारूप का उपयोग करना चाहिए
- उपकरणों के उपयोग को परिभाषित करता है: विनिमय दर जानकारी प्राप्त करने के लिए `get_exchange_rate` उपकरण का उपयोग करता है

##### 2.2 मुख्य विधियाँ
1. **इनिशियलाइज़ेशन मेथड `__init__`**
   - API कुंजी और बेस URL कॉन्फ़िगर करता है
   - बातचीत इतिहास को इनिशियलाइज़ करता है

2. **विनिमय दर क्वेरी मेथड `get_exchange_rate`**
   - पैरामीटर्स: स्रोत मुद्रा, लक्ष्य मुद्रा, तिथि (डिफ़ॉल्ट रूप से नवीनतम)
   - विनिमय दर डेटा प्राप्त करने के लिए फ्रैंकफर्टर API को कॉल करता है
   - JSON प्रारूप में विनिमय दर जानकारी लौटाता है

3. **स्ट्रीमिंग मेथड `stream`**
   - स्ट्रीमिंग प्रतिक्रिया कार्यक्षमता प्रदान करता है
   - वास्तविक समय में प्रसंस्करण स्थिति और परिणाम लौटाता है
   - उपकरण कॉल के लिए मध्यवर्ती स्थिति प्रतिक्रिया का समर्थन करता है

#### 3. कार्य प्रवाह
1. **उपयोगकर्ता क्वेरी प्राप्त करना**
   - उपयोगकर्ता संदेश को बातचीत इतिहास में जोड़ना

2. **मॉडल प्रसंस्करण**
   - सिस्टम प्रॉम्प्ट और बातचीत इतिहास को मॉडल को भेजना
   - मॉडल विश्लेषण करता है कि क्या उपकरण का उपयोग करने की आवश्यकता है

3. **उपकरण कॉल (यदि आवश्यक हो)**
   - यदि मॉडल उपकरण का उपयोग करने का निर्णय लेता है, तो उपकरण कॉल अनुरोध लौटाता है
   - विनिमय दर क्वेरी निष्पादित करना
   - क्वेरी परिणामों को बातचीत इतिहास में जोड़ना

4. **अंतिम प्रतिक्रिया उत्पन्न करना**
   - उपकरण कॉल परिणामों के आधार पर अंतिम उत्तर उत्पन्न करना
   - स्वरूपित JSON प्रतिक्रिया लौटाना

#### 4. प्रतिक्रिया प्रारूप
Agent की प्रतिक्रियाएँ हमेशा निम्नलिखित स्थितियों के साथ JSON प्रारूप का उपयोग करती हैं:
- `completed`: कार्य पूरा हुआ
- `input_required`: उपयोगकर्ता इनपुट की आवश्यकता है
- `error`: त्रुटि हुई
- `tool_use`: उपकरण का उपयोग करने की आवश्यकता है

#### 5. त्रुटि प्रबंधन
- पूर्ण त्रुटि प्रबंधन तंत्र शामिल है
- API कॉल विफलताओं को संभालता है
- JSON पार्सिंग त्रुटियों को संभालता है
- अमान्य प्रतिक्रिया प्रारूपों को संभालता है

### Agent का परीक्षण
परीक्षण कोड इस प्रकार है:
```python
import asyncio
import logging
from currency_agent import CurrencyAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    agent = CurrencyAgent()
    
    # परीक्षण केस
    test_queries = [
        "What is the current exchange rate from USD to EUR?",
        "Can you tell me the exchange rate between GBP and JPY?",
        "What's the weather like today?",  # यह मुद्रा से संबंधित नहीं होने के कारण अस्वीकार किया जाना चाहिए
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: {query}")
        async for response in agent.stream(query, "test_session"):
            logger.info(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main()) 
```

यदि सब कुछ सही ढंग से कॉन्फ़िगर किया गया है, विशेष रूप से एनवायरनमेंट कॉन्फ़िगरेशन, तो आपको इसी तरह का आउटपुट दिखाई देगा:
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

### AgentExecutor का कार्यान्वयन
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

इस कोड की तर्क का विश्लेषण करें:
यह एक `CurrencyAgentExecutor` नामक AgentExecutor वर्ग है जो मुख्य रूप से मुद्रा से संबंधित एजेंट संचालनों को संभालता है। आइए इसकी संरचना और कार्यक्षमता का विस्तार से विश्लेषण करें:

A2A एजेंट अनुरोधों और प्रतिक्रियाओं/घटनाओं को संभालने की मुख्य तर्क AgentExecutor द्वारा कार्यान्वित की जाती है। A2A Python SDK एक अमूर्त आधार वर्ग *a2a.server.agent_execution.AgentExecutor* प्रदान करता है जिसे आपको कार्यान्वित करना होगा।

AgentExecutor वर्ग दो मुख्य विधियों को परिभाषित करता है:
- `async def execute(self, context: RequestContext, event_queue: EventQueue)`: आने वाले अनुरोधों को संभालता है जिन्हें प्रतिक्रियाओं या घटना प्रवाह की आवश्यकता होती है। यह उपयोगकर्ता इनपुट (संदर्भ के माध्यम से प्राप्त) को संसाधित करता है और `event_queue` का उपयोग Message, Task, TaskStatusUpdateEvent या TaskArtifactUpdateEvent वस्तुओं को भेजने के लिए करता है।
- `async def cancel(self, context: RequestContext, event_queue: EventQueue)`: चल रहे कार्यों को रद्द करने के अनुरोधों को संभालता है।

RequestContext आने वाले अनुरोध के बारे में जानकारी प्रदान करता है, जैसे उपयोगकर्ता का संदेश और कोई मौजूदा कार्य विवरण। EventQueue का उपयोग एजेंट द्वारा क्लाइंट को घटनाएँ भेजने के लिए किया जाता है।

### AgentServer का कार्यान्वयन

कोड:
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
AgentSkill एजेंट द्वारा किए जा सकने वाले विशिष्ट कौशल या कार्यों का वर्णन करता है। यह एक बिल्डिंग ब्लॉक है जो क्लाइंट को बताता है कि एजेंट किस प्रकार के कार्यों के लिए उपयुक्त है।
AgentSkill के मुख्य गुण (a2a.types में परिभाषित):
- id: कौशल के लिए अद्वितीय पहचानकर्ता
- name: मानव-पठनीय नाम
- description: कौशल की कार्यक्षमता का अधिक विस्तृत विवरण
- tags: वर्गीकरण और खोज के लिए कीवर्ड
- examples: प्रॉम्प्ट या उपयोग के मामलों के उदाहरण
- inputModes / outputModes: समर्थित इनपुट और आउटपुट MIME प्रकार (उदाहरण के लिए, "text/plain", "application/json")

यह कौशल बहुत सरल है: मुद्रा रूपांतरण को संभालना, इनपुट और आउटपुट `text` है, AgentCard में परिभाषित।

#### AgentCard
AgentCard A2A सर्वर द्वारा प्रदान किया गया एक JSON दस्तावेज़ है, जो आमतौर पर `.well-known/agent.json` एंडपॉइंट पर स्थित होता है। यह एजेंट के लिए एक डिजिटल व्यवसाय कार्ड की तरह है।
AgentCard के मुख्य गुण (a2a.types में परिभाषित):
- name, description, version: बुनियादी पहचान जानकारी
- url: A2A सेवा तक पहुंचने के लिए एंडपॉइंट
- capabilities: समर्थित A2A सुविधाओं को निर्दिष्ट करता है, जैसे streaming या pushNotifications
- defaultInputModes / defaultOutputModes: एजेंट के लिए डिफ़ॉल्ट MIME प्रकार
- skills: एजेंट द्वारा प्रदान किए गए AgentSkill वस्तुओं की सूची

#### AgentServer

- DefaultRequestHandler:
SDK DefaultRequestHandler प्रदान करता है। यह हैंडलर एक AgentExecutor कार्यान्वयन (यहाँ CurrencyAgentExecutor) और एक TaskStore (यहाँ InMemoryTaskStore) लेता है।
यह आने वाले A2A RPC कॉल को एजेंट पर उचित विधियों (जैसे execute या cancel) पर रूट करता है।
TaskStore का उपयोग DefaultRequestHandler द्वारा कार्यों के जीवनचक्र को प्रबंधित करने के लिए किया जाता है, विशेष रूप से स्थिति-संबंधित इंटरैक्शन, स्ट्रीमिंग और पुनः सब्सक्रिप्शन के लिए।
भले ही AgentExecutor सरल हो, हैंडलर को एक टास्क स्टोर की आवश्यकता होती है।

- A2AStarletteApplication:
A2AStarletteApplication वर्ग agent_card और request_handler (इसके कंस्ट्रक्टर में http_handler कहा जाता है) के साथ इनिशियलाइज़ किया जाता है।
agent_card महत्वपूर्ण है क्योंकि सर्वर इसे डिफ़ॉल्ट रूप से `/.well-known/agent.json` एंडपॉइंट पर एक्सपोज़ करेगा।
request_handler आपके AgentExecutor के साथ इंटरैक्ट करके सभी आने वाले A2A मेथड कॉल को संभालने के लिए जिम्मेदार है।

- uvicorn.run(server_app_builder.build(), ...):
A2AStarletteApplication में एक build() मेथड है जो वास्तविक [Starlette](https://www.starlette.io/) एप्लिकेशन बनाता है।
फिर यह `uvicorn.run()` का उपयोग करके इस एप्लिकेशन को चलाता है, जिससे आपका एजेंट HTTP के माध्यम से सुलभ हो जाता है।
host='0.0.0.0' सर्वर को आपकी मशीन के सभी नेटवर्क इंटरफेस पर सुलभ बनाता है।
port=9999 सुनने के लिए पोर्ट निर्दिष्ट करता है। यह AgentCard में url से मेल खाता है।

### चलाना

#### सर्वर चलाना
```bash
uv run python main.py
```
आउटपुट:
```bash
INFO:     Started server process [70842]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:10000 (Press CTRL+C to quit)
```

#### क्लाइंट चलाना
क्लाइंट कोड इस प्रकार है:
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

इसे इस तरह चलाएँ:
```bash
uv run python test_client.py
```

ट्यूटोरियल समाप्त। 