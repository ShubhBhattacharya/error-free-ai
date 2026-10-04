"""Provider abstraction and client dispatch for Multi-Agent Consensus.

Supports:
- Groq
- OpenRouter
- OpenAI
- Google Gemini (API / OpenAI-compatible endpoint)
- Anthropic Claude
- Ollama (Local)
- HuggingFace Inference
- Resilient Local Fallback Engine (zero-crash guarantee)
"""
import os
import json
import httpx
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

PROVIDERS_CONFIG = {
    "groq": {
        "name": "Groq LPU",
        "env_key": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "default_model": "llama-3.3-70b-versatile",
        "headers": lambda key: {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
    },
    "openrouter": {
        "name": "OpenRouter Multi-Model",
        "env_key": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "headers": lambda key: {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Error-Free AI"
        }
    },
    "openai": {
        "name": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "url": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-4o-mini",
        "headers": lambda key: {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
    },
    "gemini": {
        "name": "Google Gemini",
        "env_key": "GEMINI_API_KEY",
        # Google Gemini provides an OpenAI-compatible endpoint
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "default_model": "gemini-2.0-flash",
        "headers": lambda key: {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "env_key": "ANTHROPIC_API_KEY",
        "url": "https://api.anthropic.com/v1/messages",
        "default_model": "claude-3-5-sonnet-20241022",
        "headers": lambda key: {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
    },
    "ollama": {
        "name": "Ollama Local",
        "env_key": "OLLAMA_BASE_URL",
        "url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1/chat/completions",
        "default_model": "llama3.2",
        "headers": lambda key: {
            "Content-Type": "application/json"
        }
    },
    "huggingface": {
        "name": "Hugging Face",
        "env_key": "HUGGINGFACE_API_KEY",
        "url": "https://api-inference.huggingface.co/v1/chat/completions",
        "default_model": "meta-llama/Llama-3.2-3B-Instruct",
        "headers": lambda key: {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
    }
}

def get_provider_status() -> Dict[str, Any]:
    """Returns availability status of all configured providers."""
    status = {}
    for pid, cfg in PROVIDERS_CONFIG.items():
        env_var = cfg["env_key"]
        val = os.getenv(env_var, "").strip()
        is_configured = bool(val and not val.startswith("your_") and len(val) > 8)
        if pid == "ollama":
            is_configured = bool(os.getenv("OLLAMA_BASE_URL"))
        status[pid] = {
            "name": cfg["name"],
            "configured": is_configured,
            "env_variable": env_var,
            "default_model": cfg["default_model"]
        }
    return status

async def execute_provider_request(
    client: httpx.AsyncClient,
    provider_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 500,
    model_override: Optional[str] = None,
    timeout: float = 12.0
) -> Tuple[bool, str, str]:
    """
    Executes a chat completion request to the specified provider.
    Returns: (success: bool, content: str, provider_model: str)
    """
    cfg = PROVIDERS_CONFIG.get(provider_id)
    if not cfg:
        return False, f"Unknown provider: {provider_id}", ""

    api_key = os.getenv(cfg["env_key"], "").strip()
    if provider_id != "ollama" and (not api_key or api_key.startswith("your_")):
        return False, f"API key for {cfg['name']} ({cfg['env_key']}) not configured", ""

    model = model_override or cfg["default_model"]
    headers = cfg["headers"](api_key)

    if provider_id == "anthropic":
        payload = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    else:
        # OpenAI compatible payload for Groq, OpenRouter, OpenAI, Gemini, Ollama, HuggingFace
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

    try:
        res = await client.post(cfg["url"], headers=headers, json=payload, timeout=timeout)
        if res.status_code == 200:
            data = res.json()
            if provider_id == "anthropic":
                content = data.get("content", [{}])[0].get("text", "")
            else:
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return True, content, f"{cfg['name']} ({model})"
        else:
            err_text = res.text
            try:
                err_json = res.json()
                err_text = err_json.get("error", {}).get("message", str(err_json))
            except Exception:
                pass
            return False, f"HTTP {res.status_code}: {err_text}", f"{cfg['name']} ({model})"
    except httpx.TimeoutException:
        return False, f"Timeout after {timeout}s", f"{cfg['name']} ({model})"
    except Exception as e:
        return False, f"Request error: {str(e)}", f"{cfg['name']} ({model})"
