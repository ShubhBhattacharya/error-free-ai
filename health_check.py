"""Health-check script to verify authentication and connectivity for all 3 AI providers.

Checks:
1. Groq API (Meta Llama, Mistral, Gemma, Qwen, DeepSeek)
2. Google GenAI / Gemini API (AQ. key format, Gemini 3.5/3.8 Flash)
3. OpenRouter API (Free Tier models: Nemotron, Qwen, Phi, LFM)
"""
import os
import sys
import time
import asyncio
import httpx
from dotenv import load_dotenv

# Ensure utf-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

async def ping_groq(client: httpx.AsyncClient) -> dict:
    if not GROQ_KEY or GROQ_KEY.startswith("gsk_your_"):
        return {"status": "FAIL", "error": "GROQ_API_KEY missing or placeholder in .env"}

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen/qwen3.8-27b",
        "messages": [{"role": "user", "content": "Respond strictly with: PONG"}],
        "max_tokens": 15,
        "temperature": 0.0
    }
    start = time.time()
    try:
        res = await client.post(url, headers=headers, json=payload, timeout=5.0)
        latency = round(time.time() - start, 3)
        if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"].strip()
            return {"status": "OK", "model": "qwen/qwen3.8-27b (Groq LPU)", "latency": f"{latency}s", "reply": content}
        return {"status": "FAIL", "http_code": res.status_code, "error": res.text[:200]}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

async def ping_gemini(client: httpx.AsyncClient) -> dict:
    if not GEMINI_KEY or GEMINI_KEY.startswith("AQ.your_") or GEMINI_KEY.startswith("AIzaSy_your_"):
        return {"status": "FAIL", "error": "GEMINI_API_KEY missing or placeholder in .env"}

    headers = {
        "x-goog-api-key": GEMINI_KEY,
        "Content-Type": "application/json"
    }
    # Using the active Gemini Flash model endpoint supporting AQ. key format
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [{"parts": [{"text": "Respond strictly with: PONG"}]}],
        "generationConfig": {"maxOutputTokens": 15, "temperature": 0.0}
    }
    start = time.time()
    try:
        res = await client.post(url, headers=headers, json=payload, timeout=6.0)
        latency = round(time.time() - start, 3)
        if res.status_code == 200:
            candidates = res.json().get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                return {"status": "OK", "model": "gemini-3.5-flash-lite (Google GenAI)", "latency": f"{latency}s", "reply": text}
        return {"status": "FAIL", "http_code": res.status_code, "error": res.text[:200]}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

async def ping_openrouter(client: httpx.AsyncClient) -> dict:
    if not OPENROUTER_KEY or OPENROUTER_KEY.startswith("sk-or-v1-your_"):
        return {"status": "FAIL", "error": "OPENROUTER_API_KEY missing or placeholder in .env"}

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Error-Free AI"
    }
    payload = {
        "model": "nvidia/nemotron-3.5-lightning:free",
        "messages": [{"role": "user", "content": "Respond strictly with: PONG"}],
        "max_tokens": 15,
        "temperature": 0.0
    }
    start = time.time()
    try:
        res = await client.post(url, headers=headers, json=payload, timeout=6.0)
        latency = round(time.time() - start, 3)
        if res.status_code == 200:
            choices = res.json().get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = (msg.get("content") or msg.get("reasoning") or "").strip()
                return {"status": "OK", "model": "nemotron-3.5-lightning:free", "latency": f"{latency}s", "reply": content[:30]}
        return {"status": "FAIL", "http_code": res.status_code, "error": res.text[:200]}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

async def run_health_check():
    print("=" * 65)
    print("🏥 ERROR-FREE AI PROVIDER HEALTH CHECK")
    print("=" * 65)
    print(f"1. GROQ_API_KEY:       {'[Configured - ' + GROQ_KEY[:8] + '...]' if GROQ_KEY else '[MISSING]'}")
    print(f"2. GEMINI_API_KEY:     {'[Configured - ' + GEMINI_KEY[:8] + '...]' if GEMINI_KEY else '[MISSING]'}")
    print(f"3. OPENROUTER_API_KEY: {'[Configured - ' + OPENROUTER_KEY[:10] + '...]' if OPENROUTER_KEY else '[MISSING]'}")
    print("-" * 65)
    print("Pinging all 3 providers concurrently...\n")

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            ping_groq(client),
            ping_gemini(client),
            ping_openrouter(client),
            return_exceptions=True
        )

    groq_res, gemini_res, openrouter_res = results

    print("📊 PROVIDER CONNECTIVITY REPORT:")
    
    # Groq
    if isinstance(groq_res, dict) and groq_res.get("status") == "OK":
        print(f"  ✅ GROQ:       AUTHENTICATED ({groq_res['model']} in {groq_res['latency']}) -> '{groq_res['reply']}'")
    else:
        print(f"  ❌ GROQ:       FAILED -> {groq_res}")

    # Gemini
    if isinstance(gemini_res, dict) and gemini_res.get("status") == "OK":
        print(f"  ✅ GEMINI:     AUTHENTICATED ({gemini_res['model']} in {gemini_res['latency']}) -> '{gemini_res['reply']}'")
    else:
        print(f"  ❌ GEMINI:     FAILED -> {gemini_res}")

    # OpenRouter
    if isinstance(openrouter_res, dict) and openrouter_res.get("status") == "OK":
        print(f"  ✅ OPENROUTER: AUTHENTICATED ({openrouter_res['model']} in {openrouter_res['latency']}) -> '{openrouter_res['reply']}'")
    else:
        print(f"  ❌ OPENROUTER: FAILED -> {openrouter_res}")

    print("=" * 65)
    all_ok = (
        isinstance(groq_res, dict) and groq_res.get("status") == "OK" and
        isinstance(gemini_res, dict) and gemini_res.get("status") == "OK" and
        isinstance(openrouter_res, dict) and openrouter_res.get("status") == "OK"
    )
    if all_ok:
        print("🎉 ALL 3 PROVIDERS AUTHENTICATED & OPERATIONAL!")
        print("   Ready for 10-Agent Parallel Consensus & Gemini Synthesis.")
    else:
        print("⚠️ One or more providers failed to authenticate.")
    print("=" * 65)
    return all_ok

if __name__ == "__main__":
    success = asyncio.run(run_health_check())
    sys.exit(0 if success else 1)
