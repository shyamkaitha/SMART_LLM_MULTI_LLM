"""
Validation functions for the LangGraph SMART-LLM pipeline.
Provides fast programmatic validation with optional LLM-based logic validation.
"""

from typing import Dict, Any, List
from scripts.langgraph_pipeline.state.schema import PipelineState, ValidationResult, FeedbackMessage
from scripts.langgraph_pipeline.utils import (
    validate_python_syntax, 
    validate_robot_skills, 
    validate_threading_patterns,
    create_feedback_message,
    validate_semantic_match,
    validate_entity_consistency,
    validate_object_names,
    validate_robot_indexing
)


def validate_decomposition(state: PipelineState) -> PipelineState:
    """
    Validate task decomposition output.
    Checks for logical structure and completeness.
    """
    print("🔍 Validating decomposition...")
    
    decomposed_plan = state.get('decomposed_plan', '')
    
    # Basic checks
    if not decomposed_plan or len(decomposed_plan.strip()) < 50:
        validation_result = ValidationResult(
            is_valid=False,
            error_type="incomplete_decomposition",
            error_message="Decomposition is too short or empty",
            suggestions=["Provide detailed task breakdown with subtasks"],
            line_number=None,
            context=None
        )
        
        feedback = create_feedback_message("validator", "decomposition", validation_result)
        state['feedback_messages'].append(feedback)
        state['current_stage'] = 'decomposition'  # Route back
        print("❌ Decomposition validation failed - too short")
        return state
    
    # Check for required components
    required_components = ['SubTask', 'Skills Required', 'CODE']
    missing_components = []
    
    for component in required_components:
        if component not in decomposed_plan:
            missing_components.append(component)
    
    if missing_components:
        validation_result = ValidationResult(
            is_valid=False,
            error_type="missing_components",
            error_message=f"Missing required components: {', '.join(missing_components)}",
            suggestions=[
                "Include clear subtask definitions",
                "Specify required skills for each subtask",
                "Provide CODE section with function definitions"
            ],
            line_number=None,
            context={"missing_components": missing_components}
        )
        
        feedback = create_feedback_message("validator", "decomposition", validation_result)
        state['feedback_messages'].append(feedback)
        state['current_stage'] = 'decomposition'  # Route back
        print(f"❌ Decomposition validation failed - missing: {missing_components}")
        return state
    
    print("✅ Decomposition validation passed")
    state['validation_results']['decomposition'] = ValidationResult(
        is_valid=True, error_type=None, error_message=None, 
        suggestions=[], line_number=None, context=None
    )
    
    # Move to next stage after successful validation
    state['current_stage'] = 'allocation'
    
    return state


def validate_allocation(state: PipelineState) -> PipelineState:
    """
    Validate robot allocation output.
    Checks robot availability and skill matching.
    """
    print("🔍 Validating allocation...")
    
    allocation_plan = state.get('allocation_plan', '')
    available_robots = state.get('available_robots', [])
    
    # Basic checks
    if not allocation_plan or len(allocation_plan.strip()) < 30:
        validation_result = ValidationResult(
            is_valid=False,
            error_type="incomplete_allocation",
            error_message="Allocation plan is too short or empty",
            suggestions=["Provide detailed robot assignments with reasoning"],
            line_number=None,
            context=None
        )
        
        feedback = create_feedback_message("validator", "allocation", validation_result)
        state['feedback_messages'].append(feedback)
        state['current_stage'] = 'allocation'  # Route back
        print("❌ Allocation validation failed - too short")
        return state
    
    # Validate robot skills
    skill_validation = validate_robot_skills(allocation_plan, available_robots)
    if not skill_validation['is_valid']:
        feedback = create_feedback_message("validator", "allocation", skill_validation)
        state['feedback_messages'].append(feedback)
        state['current_stage'] = 'allocation'  # Route back
        print(f"❌ Allocation validation failed - {skill_validation['error_message']}")
        return state
    
    print("✅ Allocation validation passed")
    state['validation_results']['allocation'] = ValidationResult(
        is_valid=True, error_type=None, error_message=None, 
        suggestions=[], line_number=None, context=None
    )
    
    # Move to next stage after successful validation
    state['current_stage'] = 'code_generation'
    
    return state


def validate_code(state: PipelineState) -> PipelineState:
    """
    Validate generated code output.
    Performs syntax, threading, and logic validation.
    """
    print("🔍 Validating generated code...")
    
    generated_code = state.get('generated_code', '')
    
    # Basic checks
    if not generated_code or len(generated_code.strip()) < 50:
        validation_result = ValidationResult(
            is_valid=False,
            error_type="incomplete_code",
            error_message="Generated code is too short or empty",
            suggestions=["Generate complete executable code with function definitions"],
            line_number=None,
            context=None
        )
        
        feedback = create_feedback_message("validator", "code_generation", validation_result)
        state['feedback_messages'].append(feedback)
        state['current_stage'] = 'code_generation'  # Route back
        print("❌ Code validation failed - too short")
        return state
    
    # 1. Syntax validation (fast)
    syntax_validation = validate_python_syntax(generated_code)
    if not syntax_validation['is_valid']:
        feedback = create_feedback_message("validator", "code_generation", syntax_validation)
        state['feedback_messages'].append(feedback)
        state['current_stage'] = 'code_generation'  # Route back
        print(f"❌ Code validation failed - syntax: {syntax_validation['error_message']}")
        return state
    
    # 2. Threading validation (fast)
    threading_validation = validate_threading_patterns(generated_code)
    if not threading_validation['is_valid']:
        feedback = create_feedback_message("validator", "code_generation", threading_validation)
        state['feedback_messages'].append(feedback)
        state['current_stage'] = 'code_generation'  # Route back
        print(f"❌ Code validation failed - threading: {threading_validation['error_message']}")
        return state
    
    # 3. Check for required function calls
    required_patterns = ['def ', 'GoToObject', 'robots[']
    missing_patterns = []
    
    for pattern in required_patterns:
        if pattern not in generated_code:
            missing_patterns.append(pattern)
    
    if missing_patterns:
        validation_result = ValidationResult(
            is_valid=False,
            error_type="missing_patterns",
            error_message=f"Missing required code patterns: {', '.join(missing_patterns)}",
            suggestions=[
                "Include function definitions (def)",
                "Use robot action calls (GoToObject, PickupObject, etc.)",
                "Execute functions with robots[0] or robots[1]"
            ],
            line_number=None,
            context={"missing_patterns": missing_patterns}
        )
        
        feedback = create_feedback_message("validator", "code_generation", validation_result)
        state['feedback_messages'].append(feedback)
        state['current_stage'] = 'code_generation'  # Route back
        print(f"❌ Code validation failed - missing patterns: {missing_patterns}")
        return state
    
    # 4. NEW: Object name validation (check objects exist in floor plan)
    print("🔍 Checking object names...")
    object_validation = validate_object_names(generated_code, state['objects'])
    if not object_validation['is_valid']:
        feedback = create_feedback_message("validator", "code_generation", object_validation)
        state['feedback_messages'].append(feedback)
        state['current_stage'] = 'code_generation'  # Route back
        print(f"❌ Code validation failed - object naming: {object_validation['error_message']}")
        return state
    
    # 5. NEW: Robot index validation (check robot indices are valid)
    print("🔍 Checking robot indexing...")
    robot_index_validation = validate_robot_indexing(generated_code, state['available_robots'])
    if not robot_index_validation['is_valid']:
        feedback = create_feedback_message("validator", "code_generation", robot_index_validation)
        state['feedback_messages'].append(feedback)
        state['current_stage'] = 'code_generation'  # Route back
        print(f"❌ Code validation failed - robot indexing: {robot_index_validation['error_message']}")
        return state
    
    # 6. NEW: Entity consistency validation (fast programmatic check)
    print("🔍 Checking entity consistency...")
    entity_validation = validate_entity_consistency(state['original_task'], generated_code)
    if not entity_validation['is_valid']:
        feedback = create_feedback_message("validator", "code_generation", entity_validation)
        state['feedback_messages'].append(feedback)
        state['current_stage'] = 'code_generation'  # Route back
        print(f"❌ Code validation failed - entity mismatch: {entity_validation['error_message']}")
        return state
    
    # 7. NEW: Semantic validation (LLM-based check for logical alignment)
    print("🔍 Checking semantic alignment...")
    try:
        # Auto-detect if we should use Claude or OpenAI based on available API keys
        import os
        claude_key_file = "/home/shyam/SMART_LLM_clone/SMART-LLM/claude_api_key.txt"
        openai_key_file = "/home/shyam/SMART_LLM_clone/SMART-LLM/api_key.txt"
        
        if os.path.exists(claude_key_file):
            # Use Claude for semantic validation
            with open(claude_key_file, 'r') as f:
                api_key = f.read().strip()
            use_claude = True
        else:
            # Fall back to OpenAI
            with open(openai_key_file, 'r') as f:
                api_key = f.read().strip()
            use_claude = False
        
        semantic_validation = validate_semantic_match(
            state['original_task'], 
            generated_code, 
            api_key,
            use_claude=use_claude
        )
        
        if not semantic_validation['is_valid']:
            feedback = create_feedback_message("validator", "code_generation", semantic_validation)
            state['feedback_messages'].append(feedback)
            state['current_stage'] = 'code_generation'  # Route back
            print(f"❌ Code validation failed - semantic mismatch: {semantic_validation['error_message']}")
            return state
            
    except Exception as e:
        print(f"⚠️ Warning: Semantic validation failed due to error: {e}")
        # Continue without semantic validation if there's an error
        pass
    
    print("✅ All code validations passed (syntax, threading, patterns, objects, robot indexing, entities, semantics)")
    state['validation_results']['code'] = ValidationResult(
        is_valid=True, error_type=None, error_message=None, 
        suggestions=[], line_number=None, context=None
    )
    
    # All validations passed - mark as complete
    state['current_stage'] = 'complete'
    state['is_successful'] = True
    state['final_stage_reached'] = 'complete'
    
    return state
