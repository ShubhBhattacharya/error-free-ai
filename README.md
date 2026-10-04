# 🚀 Error-Free AI: Multi-Agent Consensus Engine

> **10 Specialized AI Minds + The 11th Supreme Synthesizer Judge** working concurrently on user prompts to eliminate hallucinations and generate the single most optimal, verified response.

---

## 🏛️ Architecture Overview

Unlike monolithic single-model systems or slow sequential chains, Error-Free AI deploys **10 specialized agent personas in parallel** (`asyncio.gather`), followed by an authoritative **Judge & Synthesizer layer** (The 11th Mind):

```
                       ┌────────────────────────┐
                       │      User Prompt       │
                       └───────────┬────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │  Async Dispatcher (Parallel Execution)  │
              └────────────────────┬────────────────────┘
                                   │
     ┌─────────────┬─────────────┬─┴───────────┬─────────────┬─────────────┐
     ▼             ▼             ▼             ▼             ▼             ▼
🧠 Logic      🎨 Stylist    💻 Code       ⚔️ Devil      🔍 Fact-     🛡️ Security
(T=0.1)       (T=0.75)      (T=0.15)      (T=0.35)      (T=0.05)      (T=0.2)
     ▲             ▲             ▲             ▲             ▲             ▲
     └─────────────┴─────────────┼─────────────┴─────────────┴─────────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
            ⚡ Executive      📊 Data        ✨ UX/Clarity   🎯 Domain
            (T=0.15)         (T=0.2)        (T=0.35)        (T=0.25)
                                 │
                                 ▼
              ┌────────────────────────────────────────┐
              │  The 11th Mind: Supreme Consensus      │
              │  Judge & Synthesis Layer               │
              └──────────────────┬─────────────────────┘
                                 ▼
              ┌────────────────────────────────────────┐
              │  Optimal Verified Output               │
              │  • Consensus Confidence % (e.g. 98%)   │
              │  • Reconciled Critiques & Trade-offs   │
              │  • 10 Detailed Mind Dossiers           │
              └────────────────────────────────────────┘
```

---

## 🧠 The 10 Specialized Personas

| # | Persona | Role | Focus & Temperature |
|---|---|---|---|
| 1 | **Core Logic & Reasoning** | Deductive Reasoner | Formal logic, causal chains, first-principles deduction (`T=0.1`) |
| 2 | **Creative & Narrative Stylist** | Stylist & Analogist | Evocative analogies, memorable mental models (`T=0.75`) |
| 3 | **Code Architecture & Syntax** | Software Architect | SOLID principles, clean code, Big-O efficiency (`T=0.15`) |
| 4 | **Devil's Advocate** | Adversarial Critic | Hidden traps, unstated assumptions, failure modes (`T=0.35`) |
| 5 | **Fact-Checker & Auditor** | Empirical Auditor | Ground truth, technical verification, zero hallucination (`T=0.05`) |
| 6 | **Edge-Case & Security** | Security Sentinel | OWASP vectors, boundary limits, concurrency & race conditions (`T=0.2`) |
| 7 | **Concise / Executive Summarizer**| BLUF Synthesizer | Bottom-Line-Up-Front, high information density (`T=0.15`) |
| 8 | **Data & Analytical Thinking** | Quantitative Analyst | Probability distributions, KPIs, benchmark metrics (`T=0.2`) |
| 9 | **UX & Clarity Optimizer** | Cognitive Ergonomist | Jargon elimination, progressive disclosure, readability (`T=0.35`) |
| 10 | **Domain Specialist** | Context SME | Adaptive industry context and vertical standards (`T=0.25`) |

---

## 🔌 Supported AI Providers

Modular slots in `.env` allow plugging in any provider:
- **Groq** (LPU ultra-low latency inference)
- **OpenRouter** (Aggregated access to DeepSeek, Llama 3.3, Mistral, etc.)
- **OpenAI** (GPT-4o, GPT-4o-mini, o1)
- **Google Gemini** (Gemini 2.0 Flash, Gemini 1.5 Pro)
- **Anthropic** (Claude 3.5 Sonnet)
- **Ollama** (Local offline models)
- **Hugging Face** (Inference API)
- **Zero-Crash Resilient Fallback Engine** built-in for graceful degradation.

---

## 🚀 Quickstart

### 1. Clone & Setup
```bash
git clone https://github.com/ShubhBhattacharya/error-free-ai.git
cd error-free-ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Keys
Copy the example environment file and insert your keys:
```bash
cp .env.example .env
```

### 3. Run Web Application
```bash
python server.py
```
Visit `http://localhost:8000` in your browser.

### 4. CLI Execution
```bash
python main.py
```

---

## 📡 API Endpoints

- `POST /api/multi-agent/query` — Full JSON consensus response with all 10 agent dossiers.
- `POST /api/multi-agent/stream` — Real-time Server-Sent Events (SSE) streaming as each agent finishes.
- `POST /ask` — Backward-compatible SSE stream.
- `GET /api/agents` — Metadata for the 10 agent personas.
- `GET /api/providers/status` — Live status of configured API keys.

---

## 📄 License
MIT
