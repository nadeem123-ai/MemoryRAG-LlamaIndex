"""
llm.py
------
LLM setup supporting both Ollama (local) and OpenAI (cloud).
Compatible with llama-index-llms-ollama and llama-index-llms-openai >= 0.1.x
"""

import os
from dotenv import load_dotenv
from llama_index.core.llms import LLM

load_dotenv()


def get_llm(
    provider: str = "ollama",
    model: str = "mistral",
    temperature: float = 0.0,
) -> LLM:
    """
    Return a LlamaIndex LLM based on the chosen provider.

    Args:
        provider:    'ollama' for local inference, 'openai' for cloud.
        model:       Model name.
                     Ollama  → 'mistral', 'llama3', 'gemma3:4b'
                     OpenAI  → 'gpt-4o', 'gpt-4o-mini'
        temperature: 0.0 = deterministic, 1.0 = creative.

    Returns:
        LlamaIndex LLM object (same interface regardless of provider).
    """
    if provider == "ollama":
        from llama_index.llms.ollama import Ollama

        print(f"Using Ollama model: '{model}'")
        llm = Ollama(
            model=model,
            temperature=temperature,
            request_timeout=300.0,
        )
        print("✓ Ollama LLM ready")
        return llm

    elif provider == "openai":
        from llama_index.llms.openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found.\n"
                "Create a .env file with:\n"
                "  OPENAI_API_KEY=sk-..."
            )
        print(f"Using OpenAI model: '{model}'")
        llm = OpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )
        print("✓ OpenAI LLM ready")
        return llm

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Use 'ollama' or 'openai'."
        )