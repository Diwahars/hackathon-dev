import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv(
        "BIFROST_BASE_URL",
        "http://localhost:8080/v1",
    ),
    api_key=os.getenv("BIFROST_API_KEY"),
)

MODEL = os.getenv(
    "BIFROST_MODEL",
    "openai/gpt-4o-mini",
)


def chat_text(messages, temperature=0.2):
    """Send a chat completion request to the Bifrost API.

    Args:
        messages: List of message objects for the conversation.
        tools: Optional list of tool definitions for function calling.
        temperature: Sampling temperature for response generation (default: 0.2).

    Returns:
        The chat completion response from the API.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )

    return (response.choices[0].message.content or "").strip()