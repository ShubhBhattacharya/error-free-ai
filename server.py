"""Error-Free AI: Multi-Agent Consensus Engine Server.

Orchestrates 10 specialized agent minds + 1 supreme synthesis judge.
Provides SSE streaming, structured REST endpoints, and UI serving.
"""
import os
import time
import json
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from engine.personas import AGENT_PERSONAS, AgentPersona
from engine.providers import get_provider_status
from engine.manager import MultiAgentManager, AgentOutput
from engine.judge import ConsensusJudge, ConsensusResult

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

app = FastAPI(
    title="Error-Free AI: Multi-Agent Consensus Engine",
    description="10 Specialized AI Minds + The 11th Synthesizer Judge for Zero-Hallucination Answers",
    version="2.0.0"
)

# Enable CORS for flexible integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_manager = MultiAgentManager(timeout_per_agent=5.0)
consensus_judge = ConsensusJudge(timeout=10.0)

class QueryRequest(BaseModel):
    query: str

# -------------------------------------------------------------
# 1. Backward-Compatible Streaming Endpoint (/ask)
# -------------------------------------------------------------
async def stream_multi_agent_pipeline(user_query: str):
    start_time = time.time()
    yield f"data: 🚀 Launching 10 Specialized AI Minds in parallel...\n\n"
    await asyncio.sleep(0.02)

    # Queue to stream progress as each agent finishes
    progress_queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(agent_id: str, output: AgentOutput):
        await progress_queue.put(output)

    # Launch execution task in background
    run_task = asyncio.create_task(
        agent_manager.run_all_concurrent(user_query, on_agent_complete=on_progress)
    )

    completed_count = 0
    total_agents = len(AGENT_PERSONAS)

    # Stream out agent completions as they happen in real-time
    while completed_count < total_agents:
        try:
            agent_out = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
            completed_count += 1
            status_symbol = "✅" if agent_out.status == "success" else "⚠️"
            yield (
                f"data: {status_symbol} {agent_out.icon} [{agent_out.display_model}] "
                f"verified ({int(agent_out.confidence_score * 100)}% conf | {agent_out.latency_seconds}s)\n\n"
            )
        except asyncio.TimeoutError:
            # Check if execution finished
            if run_task.done():
                while not progress_queue.empty():
                    agent_out = progress_queue.get_nowait()
                    completed_count += 1
                    yield f"data: ✅ {agent_out.icon} [{agent_out.name}] finished\n\n"
                break
            await asyncio.sleep(0.02)

    agent_outputs = await run_task

    yield "data: ⚖️ The 11th Mind (Consensus Judge) synthesizing multi-agent matrix...\n\n"
    await asyncio.sleep(0.05)

    # Run synthesis layer
    consensus_result = await consensus_judge.synthesize(
        query=user_query,
        agent_outputs=agent_outputs,
        total_pipeline_start_time=start_time
    )

    # Send result marker for existing UI parsing
    yield f"data: [RESULT_START]{consensus_result.final_answer}[RESULT_END]\n\n"
    yield (
        f"data: ⏱️ 10-Agent Consensus reached in {consensus_result.total_latency_seconds}s | "
        f"Consensus Score: {consensus_result.consensus_score}% ({consensus_result.confidence_level})\n\n"
    )

@app.post("/ask")
async def ask_legacy(req: QueryRequest):
    """Compatible with legacy chat interface while executing full 10-agent consensus."""
    return StreamingResponse(
        stream_multi_agent_pipeline(req.query),
        media_type="text/event-stream"
    )

# -------------------------------------------------------------
# 2. Modern Multi-Agent REST API Endpoint
# -------------------------------------------------------------
@app.post("/api/multi-agent/query")
async def multi_agent_query(req: QueryRequest):
    """
    Primary API endpoint returning full JSON consensus breakdown:
    - Synthesized optimal answer
    - Consensus confidence metrics
    - Individual breakdown of all 10 specialized agent dossiers
    """
    start_time = time.time()
    agent_outputs = await agent_manager.run_all_concurrent(req.query)
    consensus_result = await consensus_judge.synthesize(
        query=req.query,
        agent_outputs=agent_outputs,
        total_pipeline_start_time=start_time
    )
    return JSONResponse(content=consensus_result.model_dump())

# -------------------------------------------------------------
# 3. Rich Event-Stream API Endpoint for Modern UI
# -------------------------------------------------------------
async def event_stream_pipeline(user_query: str):
    start_time = time.time()
    progress_queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(agent_id: str, output: AgentOutput):
        await progress_queue.put(output)

    yield f"event: init\ndata: {json.dumps({'status': 'started', 'total_agents': len(AGENT_PERSONAS)})}\n\n"

    run_task = asyncio.create_task(
        agent_manager.run_all_concurrent(user_query, on_agent_complete=on_progress)
    )

    completed_count = 0
    total_agents = len(AGENT_PERSONAS)

    while completed_count < total_agents:
        try:
            agent_out = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
            completed_count += 1
            yield f"event: agent_complete\ndata: {json.dumps(agent_out.model_dump())}\n\n"
        except asyncio.TimeoutError:
            if run_task.done():
                while not progress_queue.empty():
                    agent_out = progress_queue.get_nowait()
                    completed_count += 1
                    yield f"event: agent_complete\ndata: {json.dumps(agent_out.model_dump())}\n\n"
                break
            await asyncio.sleep(0.02)

    agent_outputs = await run_task

    yield f"event: judging\ndata: {json.dumps({'status': 'synthesizing', 'judge': 'The 11th Mind'})}\n\n"

    consensus_result = await consensus_judge.synthesize(
        query=user_query,
        agent_outputs=agent_outputs,
        total_pipeline_start_time=start_time
    )

    yield f"event: consensus_complete\ndata: {json.dumps(consensus_result.model_dump())}\n\n"

@app.post("/api/multi-agent/stream")
async def multi_agent_stream(req: QueryRequest):
    """SSE endpoint streaming structured JSON events for each agent and final synthesis."""
    return StreamingResponse(
        event_stream_pipeline(req.query),
        media_type="text/event-stream"
    )

# -------------------------------------------------------------
# 4. Agent Metadata & System Status Endpoints
# -------------------------------------------------------------
@app.get("/api/agents")
async def list_agents():
    """Lists all 10 specialized agent personas and their configuration."""
    return JSONResponse(content=[p.model_dump() for p in AGENT_PERSONAS])

@app.get("/api/providers/status")
async def provider_status():
    """Returns provider key configuration and system status."""
    return JSONResponse(content=get_provider_status())

# -------------------------------------------------------------
# 5. UI Serving
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)