import os
import ollama
from src.settings import OLLAMA_MODEL  # <- add

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
    Force the model to output EXACTLY the provided text.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a healthcheck. "
                "You must output EXACTLY the user content and nothing else—no punctuation, no commentary."
            ),
        },
        {"role": "user", "content": text},
    ]
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={
            "temperature": 0.0,
            "top_p": 0.0,
            "repeat_penalty": 1.0
        },
    )
    return resp["message"]["content"].strip()
