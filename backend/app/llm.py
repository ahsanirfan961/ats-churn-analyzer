from langchain_openai import ChatOpenAI

from app.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL


def chat_model(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY or "missing-key",
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
    )
