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
        async with httpx.AsyncClient(timeout=30.0) as client:
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