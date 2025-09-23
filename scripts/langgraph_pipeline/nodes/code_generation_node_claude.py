"""
Code Generation Agent using Claude API instead of OpenAI.
"""

from typing import Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from scripts.langgraph_pipeline.state.schema import PipelineState, AgentResponse
from scripts.langgraph_pipeline.utils import format_feedback_for_prompt
import sys
sys.path.append(".")
import resources.actions as actions


class CodeGenerationAgentClaude:
    """Code generation agent using Claude API."""
    
    def __init__(self, model_name="claude-3-haiku-20240307", api_key=None):
        self.llm = ChatAnthropic(
            model=model_name,
            anthropic_api_key=api_key,
            temperature=0,
            max_tokens=1400
        )
    
    def __call__(self, state: PipelineState) -> PipelineState:
        """Process code generation using Claude."""
        
        print("💻 LangGraph Stage 3: Code Generation...")
        
        # Increment attempt counter
        state['code_generation_attempts'] += 1
        
        # Check max attempts
        if state['code_generation_attempts'] > state.get('max_attempts_per_agent', 5):
            state['current_stage'] = 'failed'
            state['errors'].append("Maximum code generation attempts exceeded")
            print(f"❌ Max code generation attempts ({state['max_attempts_per_agent']}) exceeded")
            return state
        
        # Build prompt with feedback if available
        feedback_text = format_feedback_for_prompt(
            [msg for msg in state['feedback_messages'] if msg['to_agent'] == 'code_generation']
        )
        
        # System prompt
        system_prompt = f"""You are a robotics code generation expert. Your job is to convert task decompositions and robot allocations into executable Python code.

Available AI2Thor Actions: {', '.join(actions.ai2thor_actions)}

Instructions:
1. Generate clean, executable Python code
2. Use proper function definitions for each subtask
3. Include robot execution calls
4. Handle parallel execution with threading if needed
5. Use exact object names as they appear in the floor plan

Code Requirements:
- Use functions like: GoToObject(robot, 'ObjectName')
- Access robots as: robots[0], robots[1], etc.
- Include proper error handling
- Follow Python best practices"""

        # User prompt
        user_prompt = f"""{feedback_text}

Training Examples:
{state['codegen_examples']}

Task Decomposition:
{state['decomposed_plan']}

Robot Allocation:
{state['allocation_plan']}

Available Objects: {[obj.get('name', obj.get('objectType')) for obj in state['objects']]}

Generate executable Python code for this task."""

        try:
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Call Claude
            response = self.llm.invoke(messages)
            generated_code = response.content
            
            # Clean up code (remove markdown formatting if present)
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()
            
            # Update state
            state['generated_code'] = generated_code
            state['current_stage'] = 'validation'
            
            print(f"✅ Code generation completed (attempt {state['code_generation_attempts']})")
            return state
            
        except Exception as e:
            error_msg = f"Code generation failed: {str(e)}"
            state['errors'].append(error_msg)
            print(f"❌ Code generation error: {error_msg}")
            
            # If we haven't hit max attempts, try again
            if state['code_generation_attempts'] < state.get('max_attempts_per_agent', 5):
                return state  # Will retry
            else:
                state['current_stage'] = 'failed'
                return state
