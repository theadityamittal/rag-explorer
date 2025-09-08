import ollama

from src.utils.settings import OLLAMA_MODEL  # <- add

MODEL_NAME = OLLAMA_MODEL


def llm_chat(system_prompt: str, user_prompt: str) -> str:
    """
    General chat helper for normal prompts.
    Uses conservative options to reduce randomness.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={
            "temperature": 0.0,
            "top_p": 0.0,
        },
    )
    return resp["message"]["content"].strip()


def llm_echo(text: str) -> str:
    """
    Simple health check to wake the model up'
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a healthcheck. "
                "If you are awake, respond with 'Yeah Yeah! I'm awake!'"
            ),
        },
        {"role": "user", "content": text},
    ]
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={"temperature": 0.0, "top_p": 0.0, "repeat_penalty": 1.0},
    )
    return resp["message"]["content"].strip()
