"""
Code Generation Agent Node for LangGraph SMART-LLM Pipeline.
Specializes in generating executable Python code for robot tasks.
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from scripts.langgraph_pipeline.state.schema import PipelineState, AgentResponse
from scripts.langgraph_pipeline.utils import format_feedback_for_prompt
import sys
sys.path.append(".")
import resources.actions as actions


class CodeGenerationAgent:
    """
    Agent specialized in code generation.
    Uses existing train_task_allocation_code.py examples with feedback enhancement.
    """
    
    def __init__(self, model_name="gpt-4", api_key=None):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0,
            max_tokens=1000,
            model_kwargs={"frequency_penalty": 0.8}
        )
        
        self.system_prompt = """You are a Python code generation expert specializing in robotics. 
        Generate clean, executable code using the provided action APIs. Focus on proper threading, 
        error handling, and efficient execution.
        
        CRITICAL REQUIREMENTS:
        - Follow EXACT patterns from training examples
        - Use proper indentation and syntax
        - Include proper threading when parallel execution is needed
        - Robot parameter FIRST in function calls: GoToObject(robot, 'Object')
        - Function signature: def task_name(robot):
        - Execution: task_name(robots[0])"""
    
    def __call__(self, state: PipelineState) -> PipelineState:
        """Execute code generation with feedback integration."""
        print("💻 LangGraph Stage 3: Code Generation...")
        
        # Increment attempt counter
        state["code_generation_attempts"] += 1
        state["current_stage"] = "code_generation"
        
        # Build prompt with feedback if available
        feedback_text = format_feedback_for_prompt([
            msg for msg in state["feedback_messages"] 
            if msg["to_agent"] == "code_generation"
        ])
        
        # PRIORITY: Training examples FIRST - LLM learns from patterns
        example_lines = state['codegen_examples'].split('\n')
        # Use clean examples - they're already minimal and perfect
        prompt = '\n'.join(example_lines) + "\n\n"
        
        # Add only essential context
        prompt += "# FOLLOW THE PATTERNS ABOVE EXACTLY - NO DEVIATIONS\n"
        prompt += "# Robot parameter FIRST: GoToObject(robot, 'Object')\n"
        prompt += "# Function signature: def task_name(robot):\n"
        prompt += "# Execution: task_name(robots[0])\n\n"
        
        # Add feedback if this is a retry
        if feedback_text:
            prompt += feedback_text
        
        # Add current task context
        prompt += f"# NEW TASK TO SOLVE:\n{state['decomposed_plan']}\n"
        prompt += f"# AVAILABLE ROBOTS: {state['available_robots']}\n"
        prompt += f"# ALLOCATION: {state['allocation_plan']}\n"
        prompt += "# GENERATE CODE FOLLOWING THE EXACT PATTERNS ABOVE:\n"
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            generated_code = response.content.strip()
            
            # Update state with successful result
            state['generated_code'] = generated_code
            state['current_stage'] = 'validation'
            
            print(f"✅ Code generation completed (attempt {state['code_generation_attempts']})")
            
        except Exception as e:
            error_msg = f"Code generation error (attempt {state['code_generation_attempts']}): {str(e)}"
            state['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            
            # Check if we should retry or fail
            if state['code_generation_attempts'] >= state['max_attempts_per_agent']:
                state['current_stage'] = 'failed'
                print(f"🚫 Code generation failed after {state['max_attempts_per_agent']} attempts")
            else:
                print(f"🔄 Will retry code generation (attempt {state['code_generation_attempts'] + 1})")
        
        # Update iteration count
        state['iteration_count'] += 1
        
        return state


def create_code_generation_agent(model_name="gpt-4", api_key=None) -> CodeGenerationAgent:
    """Factory function to create code generation agent."""
    return CodeGenerationAgent(model_name, api_key)
