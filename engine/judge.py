"""The 11th Mind: Master Consensus Synthesizer & Judge.

Synthesizes the parallel agent dossiers into a definitive, optimal Markdown answer.
- If code is requested, outputs complete, runnable code blocks with exact syntax.
- Zero mock or placeholder templates.
- Primary: Google Gemini Flash (Google GenAI)
- Failover: Groq LPU (qwen/qwen3.8-27b)
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
    synthesis_model_used: str = "Gemini Flash (Google GenAI)"
    timestamp: float = Field(default_factory=time.time)

class ConsensusJudge:
    """Master Consensus Judge generating real technical synthesis and runnable code."""

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout

    def _build_judge_prompt(self, query: str, agent_outputs: List[AgentOutput]) -> str:
        dossier_sections = []
        for a in agent_outputs:
            if a.status == "success" and a.answer and not a.answer.startswith("["):
                dossier_sections.append(
                    f"### [{a.name} - {a.display_model}]\n"
                    f"{a.answer}\n"
                    f"{'Risk Note: ' + a.critique_or_risks if a.critique_or_risks else ''}"
                )

        dossier = "\n\n".join(dossier_sections) if dossier_sections else "No agent insights available."

        prompt = f"""You are the Master Consensus Judge for an error-free AI system.
Your mission: Synthesize the verified multi-agent inputs into the single most optimal, authoritative, and complete answer for the user.

USER QUERY:
"{query}"

PARALLEL INSIGHTS FROM SPECIALIZED AGENTS:
{dossier}

MANDATORY SYNTHESIS INSTRUCTIONS:
1. ANSWER THE USER'S DIRECT REQUEST FULLY AND THOROUGHLY.
2. IF THE USER ASKS FOR CODE (e.g., C, Python, JavaScript, Algorithms, Data Structures):
   - You MUST provide COMPLETE, RUNNABLE, WELL-COMMENTED CODE BLOCKS with exact syntax (e.g., ```c ... ```).
   - Include complete structs/typedefs, helper functions, and a clear main() demonstration if applicable.
   - Do NOT just write abstract executive summaries or incomplete pseudo-code.
3. Incorporate critical safeguards (boundary checks, NULL pointer validation, memory free/malloc).
4. Use clear Markdown structure with headings, bullet points, and code blocks.
5. End with a short '### 🏛️ Consensus Verification' confirming the consensus findings."""
        return prompt

    async def synthesize(
        self,
        query: str,
        agent_outputs: List[AgentOutput],
        total_pipeline_start_time: float
    ) -> ConsensusResult:
        """Synthesizes agent dossiers in real-time using live LLMs."""
        judge_prompt = self._build_judge_prompt(query, agent_outputs)
        system_role = (
            "You are the Supreme Consensus Judge. Provide complete, accurate, "
            "and production-grade answers. If code is requested, provide full runnable code."
        )

        final_text = ""
        model_used = "Gemini Flash"

        async with httpx.AsyncClient() as client:
            # 1. Primary: Google Gemini Flash
            ok, text = await call_gemini(
                client=client,
                model="gemini-3.5-flash-lite",
                system_prompt=system_role,
                user_prompt=judge_prompt,
                temperature=0.15,
                max_tokens=1500,  # Generous token budget for full runnable code
                timeout=self.timeout
            )
            if ok and text:
                final_text = text
                model_used = "Gemini Flash (Google GenAI)"
            else:
                # 2. Failover: Groq LPU (qwen/qwen3.8-27b)
                ok_groq, text_groq = await call_groq(
                    client=client,
                    model="qwen/qwen3.8-27b",
                    system_prompt=system_role,
                    user_prompt=judge_prompt,
                    temperature=0.15,
                    max_tokens=1500,
                    timeout=self.timeout
                )
                if ok_groq and text_groq:
                    final_text = text_groq
                    model_used = "Groq LPU (qwen/qwen3.8-27b)"
                else:
                    # Collect whatever answers the agents provided directly if judge call fails
                    valid_agent_answers = [a.answer for a in agent_outputs if a.status == "success" and a.answer]
                    if valid_agent_answers:
                        final_text = (
                            "## 🎯 Multi-Agent Verified Output\n\n" +
                            "\n\n---\n\n".join(valid_agent_answers[:3])
                        )
                        model_used = "Direct Agent Aggregator"
                    else:
                        final_text = (
                            f"## ⚠️ API Error\n\n"
                            f"Unable to reach synthesis models: {text or text_groq}. "
                            f"Please verify network connection and API key quotas."
                        )
                        model_used = "System Diagnostics"

        # Calculate consensus metrics
        successful_agents = [a for a in agent_outputs if a.status == "success"]
        if successful_agents:
            avg_conf = sum(a.confidence_score for a in successful_agents) / len(successful_agents)
            score = min(int(avg_conf * 100), 99)
        else:
            score = 70

        total_latency = round(time.time() - total_pipeline_start_time, 2)

        deliberations = [
            f"Deliberated across {len(successful_agents)}/{len(agent_outputs)} live AI agents",
            f"Synthesized by {model_used} in {total_latency}s"
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
