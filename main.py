"""Error-Free AI CLI Runner.

Demonstrates parallel multi-agent execution and master consensus synthesis.
"""
import os
import time
import asyncio
from dotenv import load_dotenv

from engine.manager import MultiAgentManager, AgentOutput
from engine.judge import ConsensusJudge

import sys
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

async def error_free_pipeline(user_query: str):
    print(f"\n=======================================================")
    print(f"🚀 [Query]: {user_query}")
    print(f"=======================================================")
    print("\n[Phase 1/2] Launching 10 Specialized AI Minds in Parallel...")

    manager = MultiAgentManager(timeout_per_agent=12.0)
    judge = ConsensusJudge(timeout=15.0)

    start_time = time.time()

    async def on_agent_done(agent_id: str, output: AgentOutput):
        print(f"  -> {output.icon} [{output.name}] finished ({int(output.confidence_score * 100)}% conf | {output.latency_seconds}s)")

    agent_outputs = await manager.run_all_concurrent(user_query, on_agent_complete=on_agent_done)

    print("\n[Phase 2/2] The 11th Mind (Supreme Consensus Judge) Synthesizing Matrix...")
    consensus = await judge.synthesize(user_query, agent_outputs, total_pipeline_start_time=start_time)

    print("\n" + "=" * 65)
    print(f"🏛️ FINAL 10-AGENT CONSENSUS SYNTHESIS (Score: {consensus.consensus_score}%)")
    print("=" * 65)
    print(consensus.final_answer)
    print("-" * 65)
    print(f"⏱️ Total Latency: {consensus.total_latency_seconds}s | Model: {consensus.synthesis_model_used}")
    print("=" * 65 + "\n")

    return consensus

if __name__ == "__main__":
    test_question = "What is the time complexity of QuickSort in worst case and how does Randomized QuickSort fix it?"
    asyncio.run(error_free_pipeline(test_question))