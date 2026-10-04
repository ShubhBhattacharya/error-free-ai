"""Multi-Agent Consensus Engine Package."""
from .personas import AGENT_PERSONAS, AgentPersona
from .manager import MultiAgentManager
from .judge import ConsensusJudge

__all__ = ["AGENT_PERSONAS", "AgentPersona", "MultiAgentManager", "ConsensusJudge"]
