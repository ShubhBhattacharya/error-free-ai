"""Provider dispatchers optimized for sub-3-second execution.

Supported Provider Pipelines:
1. Groq API (Extreme tokens/sec for Llama, Mistral, Gemma, Qwen, DeepSeek)
2. Google GenAI / Gemini API (Supporting new AQ. and AIza keys, Gemini 3.5/3.8 Flash)
3. OpenRouter (Free tier model endpoints for Qwen, Phi, Nemotron, LFM)
"""
import os
import json
import httpx
from typing import Dict, Any, Tuple, Optional, List
from dotenv import load_dotenv

load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

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
            "models": ["Meta Llama", "Mistral AI", "Gemma 2", "Qwen 3.8", "DeepSeek"]
        },
        "gemini": {
            "name": "Google GenAI (Gemini Flash)",
            "configured": bool(gemini_key and not (gemini_key.startswith("AQ.your_") or gemini_key.startswith("AIzaSy_your_"))),
            "env_variable": "GEMINI_API_KEY",
            "models": ["Gemini 3.5 Flash Lite", "Gemini 3.8 Flash", "Gemini Flash Latest"]
        },
        "openrouter": {
            "name": "OpenRouter (Free Tier)",
            "configured": bool(openrouter_key and not openrouter_key.startswith("sk-or-v1-your_")),
            "env_variable": "OPENROUTER_API_KEY",
            "models": ["Nemotron 3.5 Lightning", "Qwen 2.5/3.8", "Phi-3.5", "LFM 2.5"]
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
    """Queries Groq with strict timeout and automatic model fallback."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key.startswith("gsk_your_"):
        return False, "GROQ_API_KEY not configured"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    # Primary model + resilient Groq fallbacks
    models_to_try = [model]
    if model != "qwen/qwen3.8-27b":
        models_to_try.append("qwen/qwen3.8-27b")
    if "openai/gpt-oss-120b" not in models_to_try:
        models_to_try.append("openai/gpt-oss-120b")

    for m in models_to_try:
        payload = {
            "model": m,
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
                if content:
                    return True, content.strip()
        except Exception:
            continue

    return False, "Groq execution unavailable"

async def call_gemini(
    client: httpx.AsyncClient,
    model: str = "gemini-3.5-flash-lite",
    system_prompt: str = "",
    user_prompt: str = "",
    temperature: float = 0.1,
    max_tokens: int = 150,
    timeout: float = 2.5
) -> Tuple[bool, str]:
    """
    Queries Google GenAI API for Gemini models.
    Supports both new AQ. key formats and traditional AIza keys.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key.startswith("AQ.your_") or key.startswith("AIzaSy_your_"):
        return False, "GEMINI_API_KEY not configured"

    headers = {
        "x-goog-api-key": key,
        "Content-Type": "application/json"
    }
    
    # Priority order for active Gemini Flash models
    candidates_models = [model, "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-flash-latest"]
    tried = set()

    for m in candidates_models:
        if m in tried:
            continue
        tried.add(m)

        url = f"{GEMINI_BASE_URL}/{m}:generateContent?key={key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"System Directive: {system_prompt}\n\nTask: {user_prompt}"}
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
                        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                        if text.strip():
                            return True, text.strip()
        except Exception:
            continue

    return False, "Gemini execution unavailable"

async def call_openrouter(
    client: httpx.AsyncClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 150,
    timeout: float = 2.0
) -> Tuple[bool, str]:
    """Queries OpenRouter models with strict timeout and fallback across active free models."""
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key or key.startswith("sk-or-v1-your_"):
        return False, "OPENROUTER_API_KEY not configured"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Error-Free AI"
    }

    # Primary model with immediate fast fallback
    models_to_try = [model]
    if model != "nvidia/nemotron-3.5-lightning:free":
        models_to_try.append("nvidia/nemotron-3.5-lightning:free")

    per_try_timeout = max(0.9, timeout / len(models_to_try))

    for m in models_to_try:
        payload = {
            "model": m,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        try:
            res = await client.post(OPENROUTER_URL, headers=headers, json=payload, timeout=per_try_timeout)
            if res.status_code == 200:
                data = res.json()
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content") or msg.get("reasoning") or ""
                    if content.strip():
                        return True, content.strip()
        except Exception:
            continue

    return False, "OpenRouter execution unavailable"

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
