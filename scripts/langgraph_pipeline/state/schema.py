"""
State schema for the LangGraph SMART-LLM pipeline.
Defines the shared state that persists across all agent interactions.
"""

from typing import TypedDict, List, Dict, Any, Literal, Optional


class FeedbackMessage(TypedDict):
    """Structured feedback message between agents."""
    from_agent: str
    to_agent: str
    error_type: str
    message: str
    context: Optional[Dict[str, Any]]


class PipelineState(TypedDict):
    """
    Shared state for the LangGraph SMART-LLM pipeline.
    This state persists across all agent calls and enables iterative refinement.
    """
    
    # ===== ORIGINAL INPUTS (IMMUTABLE) =====
    original_task: str
    available_robots: List[Dict[str, Any]]
    objects: List[Dict[str, Any]]
    ground_truth: List[Dict[str, Any]]
    trans: int
    max_trans: int
    floor_plan: int
    
    # ===== TRAINING EXAMPLES (IMMUTABLE) =====
    decomp_examples: str
    alloc_examples: str
    codegen_examples: str
    
    # ===== WORK PRODUCTS (MUTABLE) =====
    decomposed_plan: str
    allocation_plan: str
    generated_code: str
    
    # ===== PROCESS CONTROL =====
    current_stage: Literal["decomposition", "allocation", "code_generation", "validation", "complete", "failed"]
    iteration_count: int
    max_iterations: int
    
    # ===== AGENT ITERATION TRACKING =====
    decomposition_attempts: int
    allocation_attempts: int
    code_generation_attempts: int
    max_attempts_per_agent: int
    
    # ===== FEEDBACK SYSTEM =====
    feedback_messages: List[FeedbackMessage]
    errors: List[str]
    validation_results: Dict[str, Any]
    
    # ===== SUCCESS TRACKING =====
    is_successful: bool
    final_stage_reached: str


class ValidationResult(TypedDict):
    """Result of validation checks."""
    is_valid: bool
    error_type: Optional[str]
    error_message: Optional[str]
    suggestions: List[str]
    line_number: Optional[int]
    context: Optional[Dict[str, Any]]


class AgentResponse(TypedDict):
    """Standard response format from agents."""
    success: bool
    content: str
    error_message: Optional[str]
    needs_retry: bool
    feedback_for_previous: Optional[FeedbackMessage]
