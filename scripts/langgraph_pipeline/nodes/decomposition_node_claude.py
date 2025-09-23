"""
Task Decomposition Agent using Claude API instead of OpenAI.
"""

from typing import Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from scripts.langgraph_pipeline.state.schema import PipelineState, AgentResponse
from scripts.langgraph_pipeline.utils import format_feedback_for_prompt
import sys
sys.path.append(".")
import resources.actions as actions


class TaskDecompositionAgentClaude:
    """Task decomposition agent using Claude API."""
    
    def __init__(self, model_name="claude-3-haiku-20240307", api_key=None):
        self.llm = ChatAnthropic(
            model=model_name,
            anthropic_api_key=api_key,
            temperature=0,
            max_tokens=1300
        )
    
    def __call__(self, state: PipelineState) -> PipelineState:
        """Process task decomposition using Claude."""
        
        print("🔍 LangGraph Stage 1: Task Decomposition...")
        
        # Increment attempt counter
        state['decomposition_attempts'] += 1
        
        # Check max attempts
        if state['decomposition_attempts'] > state.get('max_attempts_per_agent', 5):
            state['current_stage'] = 'failed'
            state['errors'].append("Maximum decomposition attempts exceeded")
            print(f"❌ Max decomposition attempts ({state['max_attempts_per_agent']}) exceeded")
            return state
        
        # Build prompt with feedback if available
        feedback_text = format_feedback_for_prompt(
            [msg for msg in state['feedback_messages'] if msg['to_agent'] == 'decomposition']
        )
        
        # System prompt
        system_prompt = f"""You are a task decomposition expert for household robotics. Your job is to break down complex tasks into subtasks that can be executed by robots with specific skills.

Available AI2Thor Actions: {', '.join(actions.ai2thor_actions)}

Instructions:
1. Break the task into clear, sequential subtasks
2. For each subtask, specify the required skills from the available actions
3. Identify opportunities for parallel execution
4. Provide a CODE section with function definitions

Format your response exactly like this:
# GENERAL TASK DECOMPOSITION
# Independent subtasks:
# SubTask 1: [description]. (Skills Required: [skills])
# SubTask 2: [description]. (Skills Required: [skills])
# We can perform SubTask 1 and SubTask 2 in [parallel/sequence].

# CODE
```python
def subtask_1_function(robot):
    # Implementation
    pass

def subtask_2_function(robot):
    # Implementation  
    pass
```"""

        # User prompt with training examples and feedback
        user_prompt = f"""{feedback_text}

Training Examples:
{state['decomp_examples']}

Task to decompose: {state['original_task']}
Available objects: {[obj.get('name', obj.get('objectType')) for obj in state['objects']]}

Please decompose this task following the format above."""

        try:
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Call Claude
            response = self.llm.invoke(messages)
            decomposed_plan = response.content
            
            # Update state
            state['decomposed_plan'] = decomposed_plan
            state['current_stage'] = 'validation'
            
            print(f"✅ Decomposition completed (attempt {state['decomposition_attempts']})")
            return state
            
        except Exception as e:
            error_msg = f"Decomposition failed: {str(e)}"
            state['errors'].append(error_msg)
            print(f"❌ Decomposition error: {error_msg}")
            
            # If we haven't hit max attempts, try again
            if state['decomposition_attempts'] < state.get('max_attempts_per_agent', 5):
                return state  # Will retry
            else:
                state['current_stage'] = 'failed'
                return state
