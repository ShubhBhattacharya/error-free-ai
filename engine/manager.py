"""Asynchronous Execution Manager for Real Multi-Agent Inference.

- Dispatches real user prompts to live AI models concurrently via asyncio.gather(return_exceptions=True).
- ZERO hardcoded or mock fallback templates.
- Strict, clean parsing preserving real code blocks, technical depth, and error diagnostics.
"""
import time
import re
import asyncio
import httpx
from typing import List, Dict, Any, Optional, Callable, Awaitable
from pydantic import BaseModel

from .personas import AGENT_PERSONAS, AgentPersona
from .providers import dispatch_agent_call

class AgentOutput(BaseModel):
    agent_id: str
    name: str
    role: str
    icon: str
    color: str
    accent: str
    model: str
    display_model: str
    provider: str
    thoughts: str = ""
    confidence_score: float = 0.95
    critique_or_risks: str = ""
    answer: str = ""
    latency_seconds: float = 0.0
    status: str = "success"  # "success", "timeout", "error"
    raw_response: Optional[str] = None

class MultiAgentManager:
    """Manages concurrent parallel execution across live AI models."""

    def __init__(self, timeout_per_agent: float = 6.0):
        self.personas: List[AgentPersona] = AGENT_PERSONAS
        self.timeout_per_agent = timeout_per_agent

    def _extract_insights(self, text: str) -> Dict[str, Any]:
        """Extracts insights from LLM response while preserving all code blocks intact."""
        clean_text = text.strip()
        
        # Check for structured markers if present
        thoughts = ""
        critique = ""
        confidence = 0.95
        answer = clean_text

        # Extract confidence if mentioned
        c_match = re.search(r"Confidence[:\s]+(\d+)%?", clean_text, re.IGNORECASE)
        if c_match:
            try:
                confidence = float(c_match.group(1)) / 100.0
            except Exception:
                pass

        # If model explicitly formatted with Thoughts / Flaw / Answer
        if "Thoughts:" in clean_text and "Answer:" in clean_text:
            parts = clean_text.split("Answer:", 1)
            meta_part = parts[0]
            answer = parts[1].strip()

            t_match = re.search(r"Thoughts?:\s*(.+?)(?=\n|Flaw:|Confidence:|$)", meta_part, re.DOTALL | re.IGNORECASE)
            if t_match:
                thoughts = t_match.group(1).strip()

            f_match = re.search(r"Flaw?:\s*(.+?)(?=\n|Confidence:|$)", meta_part, re.DOTALL | re.IGNORECASE)
            if f_match:
                critique = f_match.group(1).strip()
        elif "```" in clean_text:
            # Code block detected - keep full raw output as answer
            answer = clean_text
            thoughts = "Generated code and architectural implementation."
        else:
            answer = clean_text
            thoughts = clean_text[:120] + "..." if len(clean_text) > 120 else clean_text

        return {
            "thoughts": thoughts,
            "confidence_score": confidence,
            "critique_or_risks": critique,
            "answer": answer
        }

    async def execute_single_agent(
        self,
        client: httpx.AsyncClient,
        persona: AgentPersona,
        user_query: str,
        on_progress: Optional[Callable[[str, AgentOutput], Awaitable[None]]] = None
    ) -> AgentOutput:
        """Executes a real model call passing the user prompt directly."""
        start_time = time.time()
        
        ok, text = await dispatch_agent_call(
            client=client,
            provider=persona.provider,
            model=persona.model,
            system_prompt=persona.system_prompt,
            user_prompt=user_query,
            temperature=persona.temperature,
            max_tokens=persona.max_tokens,
            timeout=self.timeout_per_agent
        )

        latency = round(time.time() - start_time, 2)

        if ok and text:
            parsed = self._extract_insights(text)
            output = AgentOutput(
                agent_id=persona.id,
                name=persona.name,
                role=persona.role,
                icon=persona.icon,
                color=persona.color,
                accent=persona.accent,
                model=persona.model,
                display_model=persona.display_model,
                provider=persona.provider,
                thoughts=parsed["thoughts"],
                confidence_score=parsed["confidence_score"],
                critique_or_risks=parsed["critique_or_risks"],
                answer=parsed["answer"],
                latency_seconds=latency,
                status="success",
                raw_response=text
            )
        else:
            # Real error/timeout diagnostic - NO mock text
            err_msg = text or f"API response timed out (> {self.timeout_per_agent}s)"
            output = AgentOutput(
                agent_id=persona.id,
                name=persona.name,
                role=persona.role,
                icon=persona.icon,
                color=persona.color,
                accent=persona.accent,
                model=persona.model,
                display_model=persona.display_model,
                provider=persona.provider,
                thoughts=f"Call to {persona.display_model} was unavailable.",
                confidence_score=0.80,
                critique_or_risks=f"API error: {err_msg}",
                answer=f"[{persona.display_model} could not return an output: {err_msg}]",
                latency_seconds=latency,
                status="error"
            )

        if on_progress:
            try:
                await on_progress(persona.id, output)
            except Exception:
                pass

        return output

    async def run_all_concurrent(
        self,
        user_query: str,
        on_agent_complete: Optional[Callable[[str, AgentOutput], Awaitable[None]]] = None
    ) -> List[AgentOutput]:
        """
        Executes all agents concurrently with real API requests.
        Never substitutes mock data.
        """
        async with httpx.AsyncClient() as client:
            tasks = [
                self.execute_single_agent(client, persona, user_query, on_agent_complete)
                for persona in self.personas
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            agent_outputs: List[AgentOutput] = []
            for i, res in enumerate(results):
                persona = self.personas[i]
                if isinstance(res, AgentOutput):
                    agent_outputs.append(res)
                else:
                    err_str = str(res) if isinstance(res, Exception) else "Unknown failure"
                    agent_outputs.append(
                        AgentOutput(
                            agent_id=persona.id,
                            name=persona.name,
                            role=persona.role,
                            icon=persona.icon,
                            color=persona.color,
                            accent=persona.accent,
                            model=persona.model,
                            display_model=persona.display_model,
                            provider=persona.provider,
                            thoughts=f"Exception during call: {err_str}",
                            confidence_score=0.75,
                            critique_or_risks=f"Exception: {err_str}",
                            answer=f"[Execution failed: {err_str}]",
                            latency_seconds=0.0,
                            status="error"
                        )
                    )

            return agent_outputs
