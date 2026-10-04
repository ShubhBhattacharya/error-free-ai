"""Provider dispatchers optimized for sub-3-second execution.

Supported Provider Pipelines:
1. Groq API (Extreme tokens/sec for Llama 3.1, Mistral, Gemma 2, DeepSeek R1)
2. Google AI Studio API (Gemini 1.5 Flash for fact-checking & master judge)
3. OpenRouter (Free tier endpoints for Qwen 2.5, Phi-3.5, Cohere Command R)
"""
import os
import json
import httpx
from typing import Dict, Any, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

def get_provider_status() -> Dict[str, Any]:
    """Returns availability status for configured providers."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    return {
        "groq": {
            "name": "Groq LPU (Ultra-Fast)",
            "configured": bool(groq_key and not groq_key.startswith("gsk_your_")),
            "env_variable": "GROQ_API_KEY",
            "models": ["Llama 3.1 70B/8B", "Gemma 2 9B", "DeepSeek R1 Distill", "Mistral AI", "Llama 3.3 70B"]
        },
        "gemini": {
            "name": "Google AI Studio",
            "configured": bool(gemini_key and not gemini_key.startswith("AIzaSy_your_")),
            "env_variable": "GEMINI_API_KEY",
            "models": ["Gemini 1.5 Flash"]
        },
        "openrouter": {
            "name": "OpenRouter (Free Tier)",
            "configured": bool(openrouter_key and not openrouter_key.startswith("sk-or-v1-your_")),
            "env_variable": "OPENROUTER_API_KEY",
            "models": ["Qwen 2.5 Coder (free)", "Phi-3.5 (free)", "Cohere Command R (free)"]
        }
    }

async def call_groq(
    client: httpx.AsyncClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 150,
    timeout: float = 2.0
) -> Tuple[bool, str]:
    """Queries Groq with strict timeout for blazing-fast inference."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key.startswith("gsk_your_"):
        return False, "GROQ_API_KEY not configured"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
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
        res = await client.post(GROQ_URL, headers=headers, json=payload, timeout=timeout)
        if res.status_code == 200:
            data = res.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return True, content.strip()
        return False, f"Groq HTTP {res.status_code}"
    except Exception as e:
        return False, f"Groq error: {str(e)}"

async def call_gemini(
    client: httpx.AsyncClient,
    model: str = "gemini-1.5-flash",
    system_prompt: str = "",
    user_prompt: str = "",
    temperature: float = 0.1,
    max_tokens: int = 150,
    timeout: float = 2.0
) -> Tuple[bool, str]:
    """Queries Google AI Studio API directly for Gemini 1.5 Flash."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key.startswith("AIzaSy_your_"):
        return False, "GEMINI_API_KEY not configured"

    url = f"{GEMINI_NATIVE_BASE}/{model}:generateContent?key={key}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"SYSTEM INSTRUCTION: {system_prompt}\n\nUSER PROMPT: {user_prompt}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }
    try:
        res = await client.post(url, headers=headers, json=payload, timeout=timeout)
        if res.status_code == 200:
            data = res.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return True, parts[0].get("text", "").strip()
        return False, f"Gemini HTTP {res.status_code}"
    except Exception as e:
        return False, f"Gemini error: {str(e)}"

async def call_openrouter(
    client: httpx.AsyncClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 150,
    timeout: float = 2.0
) -> Tuple[bool, str]:
    """Queries OpenRouter free tier models with strict timeout."""
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key or key.startswith("sk-or-v1-your_"):
        return False, "OPENROUTER_API_KEY not configured"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Error-Free AI"
    }
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
        res = await client.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
        if res.status_code == 200:
            data = res.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return True, content.strip()
        return False, f"OpenRouter HTTP {res.status_code}"
    except Exception as e:
        return False, f"OpenRouter error: {str(e)}"

async def dispatch_agent_call(
    client: httpx.AsyncClient,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 150,
    timeout: float = 2.0
) -> Tuple[bool, str]:
    """Dispatches request to assigned provider with strict timeout."""
    if provider == "groq":
        return await call_groq(client, model, system_prompt, user_prompt, temperature, max_tokens, timeout)
    elif provider == "gemini":
        return await call_gemini(client, model, system_prompt, user_prompt, temperature, max_tokens, timeout)
    elif provider == "openrouter":
        return await call_openrouter(client, model, system_prompt, user_prompt, temperature, max_tokens, timeout)
    else:
        return False, f"Unknown provider: {provider}"
