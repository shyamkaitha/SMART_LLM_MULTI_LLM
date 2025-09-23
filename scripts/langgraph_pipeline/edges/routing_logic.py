"""
Routing logic for the LangGraph SMART-LLM pipeline.
Determines which agent to call next based on current state and validation results.
"""

from typing import Literal
from scripts.langgraph_pipeline.state.schema import PipelineState


def route_next_agent(state: PipelineState) -> Literal["decomposition", "allocation", "code_generation", "validation", "complete", "failed"]:
    """
    Determine the next agent to call based on current state.
    Implements the core routing logic for iterative refinement.
    """
    
    current_stage = state.get('current_stage', 'decomposition')
    iteration_count = state.get('iteration_count', 0)
    max_iterations = state.get('max_iterations', 9)
    
    # Check global iteration limit
    if iteration_count >= max_iterations:
        print(f"🚫 Maximum iterations ({max_iterations}) reached - failing")
        return "failed"
    
    # Check individual agent attempt limits
    max_attempts = state.get('max_attempts_per_agent', 3)
    decomp_attempts = state.get('decomposition_attempts', 0)
    alloc_attempts = state.get('allocation_attempts', 0)
    codegen_attempts = state.get('code_generation_attempts', 0)
    
    # Route based on current stage
    if current_stage == "decomposition":
        if decomp_attempts >= max_attempts:
            print(f"🚫 Decomposition max attempts ({max_attempts}) reached - failing")
            return "failed"
        return "decomposition"
    
    elif current_stage == "allocation":
        if alloc_attempts >= max_attempts:
            print(f"🚫 Allocation max attempts ({max_attempts}) reached - failing")
            return "failed"
        return "allocation"
    
    elif current_stage == "code_generation":
        if codegen_attempts >= max_attempts:
            print(f"🚫 Code generation max attempts ({max_attempts}) reached - failing")
            return "failed"
        return "code_generation"
    
    elif current_stage == "validation":
        return "validation"
    
    elif current_stage == "complete":
        return "complete"
    
    elif current_stage == "failed":
        return "failed"
    
    else:
        # Default fallback - start with decomposition
        print(f"⚠️ Unknown stage '{current_stage}' - defaulting to decomposition")
        return "decomposition"


def should_continue(state: PipelineState) -> bool:
    """
    Determine if the pipeline should continue or terminate.
    Used by LangGraph to decide when to stop the workflow.
    """
    
    current_stage = state.get('current_stage', 'decomposition')
    iteration_count = state.get('iteration_count', 0)
    max_iterations = state.get('max_iterations', 9)
    
    # Stop conditions
    if current_stage in ["complete", "failed"]:
        return False
    
    if iteration_count >= max_iterations:
        state['current_stage'] = 'failed'
        state['final_stage_reached'] = f'failed_at_{current_stage}_after_{iteration_count}_iterations'
        return False
    
    # Continue if we haven't reached terminal states
    return True


def get_next_node_name(state: PipelineState) -> str:
    """
    Get the name of the next node to execute.
    Maps routing logic to actual node names in the graph.
    """
    
    next_agent = route_next_agent(state)
    
    # Map agent types to node names (OpenAI pipeline naming)
    node_mapping = {
        "decomposition": "decompose_task",
        "allocation": "allocate_robots", 
        "code_generation": "generate_code",
        "validation": "validate_decomposition",  # Will be handled by conditional edges
        "complete": "END",
        "failed": "END"
    }
    
    node_name = node_mapping.get(next_agent, "END")
    
    print(f"🔄 Routing: {state.get('current_stage', 'unknown')} → {next_agent} (node: {node_name})")
    
    return node_name


def create_conditional_edge_function():
    """
    Create the conditional edge function for LangGraph.
    This function will be used by LangGraph to determine routing.
    """
    def conditional_edge(state: PipelineState):
        return get_next_node_name(state)
    
    return conditional_edge
