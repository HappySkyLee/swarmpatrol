#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

provider = os.getenv("AGENT_LLM_PROVIDER", "gemini").strip().lower()
print(f"Provider: {provider}")
print(f"Manus API Key present: {bool(os.getenv('MANUS_API_KEY'))}")
print(f"Manus Model: {os.getenv('MANUS_MODEL')}")

# Try to import and verify orchestrator
from agent_orchestrator import CommandAgentOrchestrator
orch = CommandAgentOrchestrator()
print(f"Orchestrator provider: {orch.provider}")
print(f"Manus client initialized: {orch.manus_client is not None}")
print(f"Manus tools count: {len(orch.manus_tools)}")
print(f"Gemini LLM initialized: {orch.gemini_llm is not None}")
print(f"Manus tool map keys: {list(orch.manus_tool_map.keys())[:5]}")
