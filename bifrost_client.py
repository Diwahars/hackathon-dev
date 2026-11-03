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


def chat(messages, tools=None, temperature=0.2):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        temperature=temperature,
    )

    return response