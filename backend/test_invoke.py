#!/usr/bin/env python
import traceback
from dotenv import load_dotenv
load_dotenv()
from agent_orchestrator import CommandAgentOrchestrator

try:
    orch = CommandAgentOrchestrator()
    result = orch.invoke('Test')
    print('Invoke succeeded')
    print('Result keys:', list(result.keys()))
    intermed = result.get('intermediate_steps')
    print('Intermediate steps type:', type(intermed))
    print('Intermediate steps length:', len(intermed) if intermed else 'None')
except Exception as e:
    print('Error:', e)
    traceback.print_exc()
