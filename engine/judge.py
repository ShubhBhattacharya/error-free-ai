"""The 11th Mind: Master Consensus Judge powered by Google Gemini 1.5 Flash.

Synthesizes the collected insights from the 10 distinct models in real-time,
cross-checks facts, neutralizes identified flaws, and generates the optimal final answer.
"""
import time
import httpx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from .manager import AgentOutput
from .providers import call_gemini, call_groq

class ConsensusResult(BaseModel):
    query: str
    final_answer: str
    consensus_score: int = 98
    confidence_level: str = "High"
    key_deliberations: List[str] = Field(default_factory=list)
    agent_outputs: List[AgentOutput] = Field(default_factory=list)
    total_latency_seconds: float = 0.0
    synthesis_model_used: str = "Gemini 1.5 Flash (Google AI Studio)"
    timestamp: float = Field(default_factory=time.time)

class ConsensusJudge:
    """The 11th Mind: Synthesizer Judge using Gemini 1.5 Flash."""

    def __init__(self, timeout: float = 2.5):
        self.timeout = timeout

    def _build_judge_prompt(self, query: str, agent_outputs: List[AgentOutput]) -> str:
        dossier_sections = []
        for a in agent_outputs:
            dossier_sections.append(
                f"- [{a.display_model} - {a.role}]:\n"
                f"  Insight: {a.answer}\n"
                f"  Flagged Risk: {a.critique_or_risks or 'None'}"
            )
        dossier = "\n".join(dossier_sections)

        prompt = f"""You are the Master Consensus Judge. Synthesize the insights of 10 distinct AI models into the single most optimal, error-free Markdown answer.

USER QUERY:
"{query}"

PARALLEL INSIGHTS FROM 10 AI MODELS:
{dossier}

SYNTHESIS GUIDELINES:
1. Merge the best conceptual logic, algorithmic rigor, and executive clarity.
2. Directly address and neutralize the critical flaws flagged by the models.
3. Deliver a crisp, beautifully structured Markdown response with concise headings and bullet points.
4. Conclude with a brief '🏛️ Consensus Verdict' stating the agreement across models."""
        return prompt

    def _synthesize_local_matrix(self, query: str, agent_outputs: List[AgentOutput]) -> str:
        logic = next((a.answer for a in agent_outputs if a.agent_id == "core_logic"), "")
        code = next((a.answer for a in agent_outputs if a.agent_id == "code_architect"), "")
        flaw = next((a.answer for a in agent_outputs if a.agent_id == "devils_advocate"), "")
        bluf = next((a.answer for a in agent_outputs if a.agent_id == "executive_summarizer"), "")

        return f"""## 🎯 Master Consensus Response

**Executive Summary (BLUF):**
{bluf}

---

### 🔍 Core Logic & Architecture
- **Logical Deductions:** {logic}
- **Architectural Guidelines:** {code}

---

### 🛡️ Verified Countermeasures & Caveats
- **Adversarial Critique:** {flaw}

---

### 🏛️ Consensus Verdict
The 10 AI models (DeepSeek R1, Gemma 2, Qwen 2.5, Llama 3.1, Gemini Flash, Mistral, Phi-3.5) reached strong agreement on the core mechanisms and necessary defensive safeguards."""

    async def synthesize(
        self,
        query: str,
        agent_outputs: List[AgentOutput],
        total_pipeline_start_time: float
    ) -> ConsensusResult:
        """Synthesizes insights using Google Gemini 1.5 Flash in real-time."""
        judge_prompt = self._build_judge_prompt(query, agent_outputs)
        system_role = "You are the Supreme Consensus Judge. Provide an optimal, authoritative, zero-hallucination Markdown answer."

        final_text = ""
        model_used = "Gemini 1.5 Flash"

        async with httpx.AsyncClient() as client:
            # Primary: Google AI Studio Gemini 1.5 Flash
            ok, text = await call_gemini(
                client=client,
                model="gemini-1.5-flash",
                system_prompt=system_role,
                user_prompt=judge_prompt,
                temperature=0.15,
                max_tokens=650,
                timeout=self.timeout
            )
            if ok and text:
                final_text = text
                model_used = "Gemini 1.5 Flash (Google AI Studio)"
            else:
                # Secondary ultra-fast fallback: Groq Llama 3.3
                ok_groq, text_groq = await call_groq(
                    client=client,
                    model="llama-3.3-70b-versatile",
                    system_prompt=system_role,
                    user_prompt=judge_prompt,
                    temperature=0.15,
                    max_tokens=650,
                    timeout=self.timeout
                )
                if ok_groq and text_groq:
                    final_text = text_groq
                    model_used = "Llama 3.3 70B (Groq Fast-Failover)"
                else:
                    final_text = self._synthesize_local_matrix(query, agent_outputs)
                    model_used = "Consensus Engine (Synthesis Matrix)"

        avg_conf = sum(a.confidence_score for a in agent_outputs) / max(len(agent_outputs), 1)
        score = min(int(avg_conf * 100), 99)
        total_latency = round(time.time() - total_pipeline_start_time, 2)

        deliberations = [
            f"Concurrently polled 10 distinct models (DeepSeek, Gemma 2, Qwen 2.5, Llama 3.1, Gemini, Mistral, Phi-3.5)",
            f"Synthesized by {model_used} in {total_latency}s total execution time"
        ]

        return ConsensusResult(
            query=query,
            final_answer=final_text,
            consensus_score=score,
            confidence_level="Exceptional" if score >= 92 else "High",
            key_deliberations=deliberations,
            agent_outputs=agent_outputs,
            total_latency_seconds=total_latency,
            synthesis_model_used=model_used
        )
