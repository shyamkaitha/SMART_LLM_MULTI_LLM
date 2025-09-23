"""
Decomposition Agent Node for LangGraph SMART-LLM Pipeline.
Specializes in breaking down complex tasks into logical subtasks.
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from scripts.langgraph_pipeline.state.schema import PipelineState, AgentResponse
from scripts.langgraph_pipeline.utils import format_feedback_for_prompt
import sys
sys.path.append(".")
import resources.actions as actions


class DecompositionAgent:
    """
    Agent specialized in task decomposition.
    Uses existing train_task_decompose.py examples with feedback enhancement.
    """
    
    def __init__(self, model_name="gpt-4", api_key=None):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0,
            max_tokens=1300
        )
        
        self.system_prompt = """You are an expert at breaking down complex household tasks into logical subtasks. 
        Focus on task dependencies, required skills, and parallelization opportunities. 
        Analyze what can be done in parallel vs sequentially.
        
        IMPORTANT: Follow the exact patterns from the training examples provided.
        Generate clean, well-structured task decomposition with clear subtasks and skill requirements."""
    
    def __call__(self, state: PipelineState) -> PipelineState:
        """Execute task decomposition with feedback integration."""
        print("🔍 LangGraph Stage 1: Task Decomposition...")
        
        # Increment attempt counter
        state["decomposition_attempts"] += 1
        state["current_stage"] = "decomposition"
        
        # Build prompt with feedback if available
        feedback_text = format_feedback_for_prompt([
            msg for msg in state["feedback_messages"] 
            if msg["to_agent"] == "decomposition"
        ])
        
        # Build prompt using existing pattern from multi_llm_pipeline.py
        prompt = f"from skills import {actions.ai2thor_actions}\n"
        prompt += "import time\nimport threading\n"
        prompt += f"\nobjects = {state['objects']}\n\n"
        prompt += state['decomp_examples']
        
        # Add feedback if this is a retry
        if feedback_text:
            prompt += f"\n{feedback_text}"
        
        prompt += f"\n\n# Task Description: {state['original_task']}"
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            decomposed_plan = response.content.strip()
            
            # Update state with successful result
            state['decomposed_plan'] = decomposed_plan
            state['current_stage'] = 'allocation'
            
            print(f"✅ Decomposition completed (attempt {state['decomposition_attempts']})")
            
        except Exception as e:
            error_msg = f"Decomposition error (attempt {state['decomposition_attempts']}): {str(e)}"
            state['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            
            # Check if we should retry or fail
            if state['decomposition_attempts'] >= state['max_attempts_per_agent']:
                state['current_stage'] = 'failed'
                print(f"🚫 Decomposition failed after {state['max_attempts_per_agent']} attempts")
            else:
                print(f"🔄 Will retry decomposition (attempt {state['decomposition_attempts'] + 1})")
        
        # Update iteration count
        state['iteration_count'] += 1
        
        return state


def create_decomposition_agent(model_name="gpt-4", api_key=None) -> DecompositionAgent:
    """Factory function to create decomposition agent."""
    return DecompositionAgent(model_name, api_key)
