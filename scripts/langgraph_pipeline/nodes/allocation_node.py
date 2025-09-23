"""
Allocation Agent Node for LangGraph SMART-LLM Pipeline.
Specializes in robot task allocation based on skills and mass capacity.
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from scripts.langgraph_pipeline.state.schema import PipelineState, AgentResponse
from scripts.langgraph_pipeline.utils import format_feedback_for_prompt
import sys
sys.path.append(".")
import resources.actions as actions


class AllocationAgent:
    """
    Agent specialized in robot allocation.
    Uses existing train_task_allocation_solution.py examples with feedback enhancement.
    """
    
    def __init__(self, model_name="gpt-4", api_key=None):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0,
            max_tokens=400,
            model_kwargs={"frequency_penalty": 0.69}
        )
        
        self.system_prompt = """You are a robot task allocation specialist. Your expertise is in matching robot 
        capabilities to task requirements, optimizing for efficiency and constraints. Consider both skills 
        and mass capacity when forming teams.
        
        IMPORTANT: 
        - Ensure robots have ALL required skills for their assigned tasks
        - Check mass capacity constraints for pickup operations
        - Form teams when individual robots lack required capabilities
        - Follow the exact patterns from training examples"""
    
    def __call__(self, state: PipelineState) -> PipelineState:
        """Execute robot allocation with feedback integration."""
        print("🤖 LangGraph Stage 2: Robot Allocation...")
        
        # Increment attempt counter
        state["allocation_attempts"] += 1
        state["current_stage"] = "allocation"
        
        # Build prompt with feedback if available
        feedback_text = format_feedback_for_prompt([
            msg for msg in state["feedback_messages"] 
            if msg["to_agent"] == "allocation"
        ])
        
        # Build focused prompt for allocation (same as multi_llm_pipeline.py)
        prompt = f"from skills import {actions.ai2thor_actions}\n"
        prompt += "import time\nimport threading\n"
        
        # Include subset of examples to reduce token count
        example_lines = state['alloc_examples'].split('\n')
        short_examples = '\n'.join(example_lines[:100])
        prompt += short_examples + "\n\n"
        
        prompt += state['decomposed_plan']
        prompt += "\n# TASK ALLOCATION\n"
        prompt += f"# Scenario: There are {len(state['available_robots'])} robots available. "
        prompt += "Robots should be assigned to subtasks that match their skills and mass capacity.\n"
        prompt += f"robots = {state['available_robots']}\n"
        
        # Reduce objects list to save tokens
        object_names = [obj['name'] for obj in state['objects']]
        prompt += f"object_types = {object_names}\n"
        
        # Add feedback if this is a retry
        if feedback_text:
            prompt += f"\n{feedback_text}"
        
        prompt += "# SOLUTION\n"
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            allocation_plan = response.content.strip()
            
            # Update state with successful result
            state['allocation_plan'] = allocation_plan
            state['current_stage'] = 'code_generation'
            
            print(f"✅ Allocation completed (attempt {state['allocation_attempts']})")
            
        except Exception as e:
            error_msg = f"Allocation error (attempt {state['allocation_attempts']}): {str(e)}"
            state['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            
            # Check if we should retry or fail
            if state['allocation_attempts'] >= state['max_attempts_per_agent']:
                state['current_stage'] = 'failed'
                print(f"🚫 Allocation failed after {state['max_attempts_per_agent']} attempts")
            else:
                print(f"🔄 Will retry allocation (attempt {state['allocation_attempts'] + 1})")
        
        # Update iteration count
        state['iteration_count'] += 1
        
        return state


def create_allocation_agent(model_name="gpt-4", api_key=None) -> AllocationAgent:
    """Factory function to create allocation agent."""
    return AllocationAgent(model_name, api_key)
