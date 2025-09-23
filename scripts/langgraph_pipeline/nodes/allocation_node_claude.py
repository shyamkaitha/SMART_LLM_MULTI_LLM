"""
Robot Allocation Agent using Claude API instead of OpenAI.
"""

from typing import Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from scripts.langgraph_pipeline.state.schema import PipelineState, AgentResponse
from scripts.langgraph_pipeline.utils import format_feedback_for_prompt
import sys
sys.path.append(".")
import resources.actions as actions


class RobotAllocationAgentClaude:
    """Robot allocation agent using Claude API."""
    
    def __init__(self, model_name="claude-3-haiku-20240307", api_key=None):
        self.llm = ChatAnthropic(
            model=model_name,
            anthropic_api_key=api_key,
            temperature=0,
            max_tokens=400
        )
    
    def __call__(self, state: PipelineState) -> PipelineState:
        """Process robot allocation using Claude."""
        
        print("🤖 LangGraph Stage 2: Robot Allocation...")
        
        # Increment attempt counter
        state['allocation_attempts'] += 1
        
        # Check max attempts
        if state['allocation_attempts'] > state.get('max_attempts_per_agent', 5):
            state['current_stage'] = 'failed'
            state['errors'].append("Maximum allocation attempts exceeded")
            print(f"❌ Max allocation attempts ({state['max_attempts_per_agent']}) exceeded")
            return state
        
        # Build prompt with feedback if available
        feedback_text = format_feedback_for_prompt(
            [msg for msg in state['feedback_messages'] if msg['to_agent'] == 'allocation']
        )
        
        # System prompt
        system_prompt = """You are a robot allocation expert. Your job is to assign robots to subtasks based on their skills and capabilities.

Instructions:
1. Analyze the decomposed subtasks and required skills
2. Match robots to subtasks based on their available skills
3. Consider robot mass capacity for object manipulation
4. Provide clear reasoning for your assignments

Format your response as:
# ROBOT ALLOCATION
# SubTask 1: Robot X (Reasoning: [why this robot])
# SubTask 2: Robot Y (Reasoning: [why this robot])"""

        # User prompt
        user_prompt = f"""{feedback_text}

Training Examples:
{state['alloc_examples']}

Task Decomposition:
{state['decomposed_plan']}

Available Robots:
{state['available_robots']}

Please allocate robots to subtasks with clear reasoning."""

        try:
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Call Claude
            response = self.llm.invoke(messages)
            allocation_plan = response.content
            
            # Update state
            state['allocation_plan'] = allocation_plan
            state['current_stage'] = 'validation'
            
            print(f"✅ Allocation completed (attempt {state['allocation_attempts']})")
            return state
            
        except Exception as e:
            error_msg = f"Allocation failed: {str(e)}"
            state['errors'].append(error_msg)
            print(f"❌ Allocation error: {error_msg}")
            
            # If we haven't hit max attempts, try again
            if state['allocation_attempts'] < state.get('max_attempts_per_agent', 5):
                return state  # Will retry
            else:
                state['current_stage'] = 'failed'
                return state
