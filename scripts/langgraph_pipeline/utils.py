"""
Utility functions for the LangGraph SMART-LLM pipeline.
Shared functionality across agents and validation logic.
"""

import os
import json
import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
import ai2thor.controller

from scripts.langgraph_pipeline.state.schema import ValidationResult, FeedbackMessage


def load_training_examples(decomp_prompt_set="train_task_decompose", alloc_prompt_set="train_task_allocation"):
    """Load training examples from existing files - IDENTICAL to multi_llm_pipeline.py"""
    base_path = os.getcwd() + "/data/pythonic_plans/"
    
    # Task decomposition examples
    with open(f"{base_path}{decomp_prompt_set}.py", "r") as f:
        decomp_examples = f.read()
        
    # Allocation examples  
    with open(f"{base_path}{alloc_prompt_set}_solution.py", "r") as f:
        alloc_examples = f.read()
        
    # Code generation examples
    with open(f"{base_path}{alloc_prompt_set}_code.py", "r") as f:
        codegen_examples = f.read()
        
    return decomp_examples, alloc_examples, codegen_examples


def get_ai2_thor_objects(floor_plan_id):
    """Get objects from AI2Thor environment - IDENTICAL to multi_llm_pipeline.py"""
    controller = ai2thor.controller.Controller(scene=f"FloorPlan{floor_plan_id}")
    objs = [{"name": obj["objectType"], "mass": obj["mass"]} 
            for obj in controller.last_event.metadata["objects"]]
    controller.stop()
    return objs


def validate_python_syntax(code: str) -> ValidationResult:
    """
    Validate Python code syntax using ast.parse.
    Fast programmatic check - no LLM calls.
    """
    try:
        ast.parse(code)
        return ValidationResult(
            is_valid=True,
            error_type=None,
            error_message=None,
            suggestions=[],
            line_number=None,
            context=None
        )
    except SyntaxError as e:
        return ValidationResult(
            is_valid=False,
            error_type="syntax_error",
            error_message=f"Syntax error: {str(e)}",
            suggestions=[
                "Check indentation and brackets",
                "Ensure proper function definitions",
                "Verify threading syntax matches training examples"
            ],
            line_number=e.lineno,
            context={"column": e.offset, "text": e.text}
        )
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_type="parsing_error",
            error_message=f"Code parsing error: {str(e)}",
            suggestions=["Review code structure and syntax"],
            line_number=None,
            context=None
        )


def validate_robot_skills(allocation_plan: str, available_robots: List[Dict]) -> ValidationResult:
    """
    Validate that allocated robots have required skills.
    Programmatic check - no LLM calls.
    """
    try:
        # Extract robot assignments from allocation plan
        robot_pattern = r"robot(\d+)"
        skill_pattern = r"(GoToObject|OpenObject|CloseObject|BreakObject|SliceObject|SwitchOn|SwitchOff|PickupObject|PutObject|DropHandObject|ThrowObject|PushObject|PullObject)"
        
        robot_matches = re.findall(robot_pattern, allocation_plan)
        skill_matches = re.findall(skill_pattern, allocation_plan)
        
        # Check if robots exist and have required skills
        for robot_num in robot_matches:
            robot_idx = int(robot_num) - 1
            if robot_idx >= len(available_robots):
                return ValidationResult(
                    is_valid=False,
                    error_type="robot_not_found",
                    error_message=f"Robot {robot_num} not available (only {len(available_robots)} robots)",
                    suggestions=[f"Use robots 1-{len(available_robots)} only"],
                    line_number=None,
                    context={"robot_number": robot_num, "available_count": len(available_robots)}
                )
            
            robot_skills = available_robots[robot_idx].get('skills', [])
            for skill in skill_matches:
                if skill not in robot_skills:
                    return ValidationResult(
                        is_valid=False,
                        error_type="skill_mismatch",
                        error_message=f"Robot {robot_num} does not have skill '{skill}'",
                        suggestions=[
                            f"Use a robot with '{skill}' skill",
                            "Form a team of robots with complementary skills",
                            "Choose different robots for this task"
                        ],
                        line_number=None,
                        context={"robot_number": robot_num, "missing_skill": skill, "robot_skills": robot_skills}
                    )
        
        return ValidationResult(
            is_valid=True,
            error_type=None,
            error_message=None,
            suggestions=[],
            line_number=None,
            context=None
        )
        
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_type="validation_error",
            error_message=f"Error validating robot skills: {str(e)}",
            suggestions=["Review robot allocation format"],
            line_number=None,
            context=None
        )


def validate_threading_patterns(code: str) -> ValidationResult:
    """
    Validate threading patterns in generated code.
    Programmatic check - no LLM calls.
    """
    try:
        # Check for proper threading patterns from training examples
        has_threading_import = "import threading" in code or "from threading import" in code
        has_thread_creation = "threading.Thread" in code
        has_thread_start = ".start()" in code
        has_thread_join = ".join()" in code
        
        if has_thread_creation and not has_threading_import:
            return ValidationResult(
                is_valid=False,
                error_type="missing_import",
                error_message="Threading used but 'import threading' missing",
                suggestions=["Add 'import threading' at the top"],
                line_number=None,
                context=None
            )
        
        if has_thread_creation and has_thread_start and not has_thread_join:
            return ValidationResult(
                is_valid=False,
                error_type="incomplete_threading",
                error_message="Thread started but not joined",
                suggestions=["Add .join() calls to wait for threads to complete"],
                line_number=None,
                context=None
            )
        
        return ValidationResult(
            is_valid=True,
            error_type=None,
            error_message=None,
            suggestions=[],
            line_number=None,
            context=None
        )
        
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_type="threading_validation_error",
            error_message=f"Error validating threading: {str(e)}",
            suggestions=["Review threading pattern against training examples"],
            line_number=None,
            context=None
        )


def create_feedback_message(from_agent: str, to_agent: str, validation_result: ValidationResult) -> FeedbackMessage:
    """Create structured feedback message from validation result."""
    return FeedbackMessage(
        from_agent=from_agent,
        to_agent=to_agent,
        error_type=validation_result["error_type"] or "general_error",
        message=validation_result["error_message"] or "Please review and improve the output",
        context=validation_result.get("context")
    )


def format_feedback_for_prompt(feedback_messages: List[FeedbackMessage]) -> str:
    """Format feedback messages for inclusion in LLM prompts."""
    if not feedback_messages:
        return ""
    
    feedback_text = "\n# FEEDBACK FROM PREVIOUS ATTEMPTS:\n"
    for i, feedback in enumerate(feedback_messages[-3:]):  # Only last 3 feedback messages
        feedback_text += f"## Feedback {i+1} ({feedback['error_type']}):\n"
        feedback_text += f"{feedback['message']}\n"
        if feedback.get('context'):
            feedback_text += f"Context: {json.dumps(feedback['context'], indent=2)}\n"
        feedback_text += "\n"
    
    feedback_text += "# PLEASE ADDRESS THE ABOVE FEEDBACK IN YOUR RESPONSE.\n\n"
    return feedback_text


def extract_entities_from_text(text: str) -> List[str]:
    """Extract key entities (objects, actions) from text."""
    import re
    
    # Common household objects and actions
    objects = [
        'mug', 'coffee', 'machine', 'bread', 'knife', 'toaster', 'plate', 'fork', 'spoon',
        'apple', 'tomato', 'lettuce', 'potato', 'fridge', 'sink', 'faucet', 'microwave',
        'stove', 'burner', 'pan', 'pot', 'bowl', 'cup', 'garbage', 'trash', 'can',
        'cabinet', 'drawer', 'counter', 'table', 'chair', 'light', 'switch', 'broom',
        'floor', 'spatula', 'egg', 'salt', 'pepper'
    ]
    
    actions = [
        'put', 'place', 'slice', 'cut', 'toast', 'cook', 'wash', 'clean', 'throw',
        'pick', 'pickup', 'take', 'get', 'open', 'close', 'switch', 'turn', 'break',
        'heat', 'chill', 'cool', 'move', 'go', 'navigate'
    ]
    
    found_entities = []
    text_lower = text.lower()
    
    # Extract objects
    for obj in objects:
        if obj in text_lower:
            found_entities.append(obj)
    
    # Extract actions
    for action in actions:
        if action in text_lower:
            found_entities.append(action)
    
    return list(set(found_entities))  # Remove duplicates


def validate_semantic_match(task: str, generated_code: str, api_key: str, use_claude: bool = False) -> ValidationResult:
    """
    Use LLM to validate semantic alignment between task and generated code.
    This catches logical mismatches that syntax validation misses.
    """
    try:
        if use_claude:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
        else:
            client = OpenAI(api_key=api_key)
        
        prompt = f"""You are a code validation expert. Your job is to determine if generated code actually solves the given task.

ORIGINAL TASK: {task}

GENERATED CODE:
{generated_code}

ANALYSIS QUESTIONS:
1. Does the generated code actually solve the original task?
2. Are the objects mentioned in the code relevant to the task?
3. Are the actions in the code appropriate for the task?

Respond with ONLY:
- "VALID" if the code matches the task intent
- "INVALID: [specific reason]" if there's a mismatch

Examples:
- Task: "Put mug in coffee machine" + Code about mugs and coffee machines = "VALID"
- Task: "Toast bread" + Code about cleaning floors = "INVALID: Code is about floor cleaning, not bread toasting"
- Task: "Slice apple" + Code about slicing tomatoes = "INVALID: Code slices wrong object (tomato vs apple)"

Your response:"""

        if use_claude:
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text.strip()
        else:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0
            )
            result = response.choices[0].message.content.strip()
        
        if result.startswith("VALID"):
            return ValidationResult(
                is_valid=True,
                error_type=None,
                error_message=None,
                suggestions=[],
                line_number=None,
                context=None
            )
        elif result.startswith("INVALID:"):
            error_message = result[8:].strip()  # Remove "INVALID: " prefix
            return ValidationResult(
                is_valid=False,
                error_type="semantic_mismatch",
                error_message=f"Task-code mismatch: {error_message}",
                suggestions=[
                    "Generate code that directly addresses the original task",
                    "Ensure objects in code match objects mentioned in task",
                    "Verify actions in code are appropriate for the task goal"
                ],
                line_number=None,
                context={"task": task, "validation_response": result}
            )
        else:
            # Fallback for unexpected response format
            return ValidationResult(
                is_valid=False,
                error_type="semantic_validation_error",
                error_message="Could not determine semantic alignment",
                suggestions=["Review code against original task requirements"],
                line_number=None,
                context={"validation_response": result}
            )
            
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_type="semantic_validation_error",
            error_message=f"Error during semantic validation: {str(e)}",
            suggestions=["Review code manually for task alignment"],
            line_number=None,
            context=None
        )


def validate_entity_consistency(task: str, code: str) -> ValidationResult:
    """
    Extract and compare entities between task and generated code.
    Fast programmatic check for basic semantic alignment.
    """
    try:
        task_entities = extract_entities_from_text(task)
        code_entities = extract_entities_from_text(code)
        
        if not task_entities:
            # If we can't extract entities from task, skip this validation
            return ValidationResult(
                is_valid=True,
                error_type=None,
                error_message=None,
                suggestions=[],
                line_number=None,
                context=None
            )
        
        # Check for overlap between task and code entities
        common_entities = set(task_entities) & set(code_entities)
        
        # If there's no overlap, it's likely a mismatch
        if not common_entities:
            return ValidationResult(
                is_valid=False,
                error_type="entity_mismatch",
                error_message=f"No common entities found between task and code. Task entities: {task_entities}, Code entities: {code_entities}",
                suggestions=[
                    "Ensure code uses objects mentioned in the task",
                    "Verify the generated code addresses the correct task",
                    "Include relevant actions and objects from the original task"
                ],
                line_number=None,
                context={
                    "task_entities": task_entities,
                    "code_entities": code_entities,
                    "common_entities": list(common_entities)
                }
            )
        
        # Check if at least 30% of task entities are present in code
        overlap_ratio = len(common_entities) / len(task_entities)
        if overlap_ratio < 0.3:
            return ValidationResult(
                is_valid=False,
                error_type="low_entity_overlap",
                error_message=f"Low entity overlap ({overlap_ratio:.1%}). Task and code may not be aligned.",
                suggestions=[
                    "Include more relevant objects and actions from the original task",
                    "Review task requirements and ensure code addresses them"
                ],
                line_number=None,
                context={
                    "task_entities": task_entities,
                    "code_entities": code_entities,
                    "overlap_ratio": overlap_ratio
                }
            )
        
        return ValidationResult(
            is_valid=True,
            error_type=None,
            error_message=None,
            suggestions=[],
            line_number=None,
            context={
                "task_entities": task_entities,
                "code_entities": code_entities,
                "common_entities": list(common_entities)
            }
        )
        
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_type="entity_validation_error",
            error_message=f"Error during entity validation: {str(e)}",
            suggestions=["Review code against task requirements"],
            line_number=None,
            context=None
        )


def validate_object_names(generated_code: str, floor_plan_objects: List[Dict]) -> ValidationResult:
    """
    Validate that all objects referenced in generated code actually exist in the floor plan.
    This catches object naming mismatches like 'TrashCan' vs 'GarbageCan'.
    """
    try:
        
        # Extract object names from the floor plan
        available_objects = set()
        object_synonyms = {}
        
        for obj in floor_plan_objects:
            obj_name = obj.get('objectType', obj.get('name'))  # Handle both formats
            if obj_name:  # Only add if not None
                available_objects.add(obj_name)
            
            # Build synonym mappings for common mismatches
            if obj_name == 'GarbageCan':
                object_synonyms['TrashCan'] = 'GarbageCan'
                object_synonyms['Trash'] = 'GarbageCan'
            elif obj_name == 'Fridge':
                object_synonyms['Refrigerator'] = 'Fridge'
            elif obj_name == 'CoffeeMachine':
                object_synonyms['CoffeeMaker'] = 'CoffeeMachine'
            elif obj_name == 'CounterTop':
                object_synonyms['Counter'] = 'CounterTop'
                object_synonyms['Countertop'] = 'CounterTop'
            elif obj_name == 'LightSwitch':
                object_synonyms['Switch'] = 'LightSwitch'
                object_synonyms['Light'] = 'LightSwitch'
        
        # Extract object references from generated code
        import re
        # Look for patterns like GoToObject(robot, 'ObjectName') or 'ObjectName'
        object_pattern = r"['\"]([A-Z][a-zA-Z]*)['\"]"
        referenced_objects = re.findall(object_pattern, generated_code)
        
        # Filter to likely object names (start with capital, reasonable length)
        likely_objects = [obj for obj in referenced_objects 
                         if len(obj) > 2 and obj[0].isupper() and obj not in ['True', 'False', 'None']]
        
        
        # Check for missing objects
        missing_objects = []
        suggested_replacements = {}
        
        for obj in set(likely_objects):
            if obj not in available_objects:
                # Check if there's a synonym available
                if obj in object_synonyms:
                    suggested_replacements[obj] = object_synonyms[obj]
                else:
                    # Look for similar objects (fuzzy matching)
                    similar_objects = [avail_obj for avail_obj in available_objects 
                                     if avail_obj.lower() in obj.lower() or obj.lower() in avail_obj.lower()]
                    if similar_objects:
                        suggested_replacements[obj] = similar_objects[0]
                    else:
                        missing_objects.append(obj)
        
        # If we have missing objects with no suggestions, it's an error
        if missing_objects and not suggested_replacements:
            return ValidationResult(
                is_valid=False,
                error_type="missing_objects",
                error_message=f"Objects not found in floor plan: {missing_objects}",
                suggestions=[
                    f"Available objects: {sorted(list(available_objects))}",
                    "Use exact object names from the floor plan",
                    "Check object spelling and capitalization"
                ],
                line_number=None,
                context={
                    "missing_objects": missing_objects,
                    "available_objects": sorted(list(available_objects))
                }
            )
        
        # If we have suggested replacements, provide helpful feedback
        if suggested_replacements:
            replacement_suggestions = []
            for wrong_name, correct_name in suggested_replacements.items():
                replacement_suggestions.append(f"Replace '{wrong_name}' with '{correct_name}'")
            
            return ValidationResult(
                is_valid=False,
                error_type="object_name_mismatch",
                error_message=f"Object naming mismatches found: {list(suggested_replacements.keys())}",
                suggestions=replacement_suggestions + [
                    "Use exact object names as they appear in the floor plan",
                    "Common mismatches: TrashCan→GarbageCan, Refrigerator→Fridge"
                ],
                line_number=None,
                context={
                    "replacements": suggested_replacements,
                    "available_objects": sorted(list(available_objects))
                }
            )
        
        # All objects found - validation passes
        return ValidationResult(
            is_valid=True,
            error_type=None,
            error_message=None,
            suggestions=[],
            line_number=None,
            context={
                "referenced_objects": list(set(likely_objects)),
                "all_objects_found": True
            }
        )
        
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_type="object_validation_error",
            error_message=f"Error during object name validation: {str(e)}",
            suggestions=["Review object names in generated code"],
            line_number=None,
            context=None
        )


def validate_robot_indexing(generated_code: str, available_robots: List[Dict]) -> ValidationResult:
    """
    Validate that robot indices in generated code match available robots.
    Catches errors like robots[1] when only 1 robot exists.
    """
    try:
        import re
        
        # Find all robot index references in the code
        robot_index_pattern = r"robots\[(\d+)\]"
        robot_indices = re.findall(robot_index_pattern, generated_code)
        
        if not robot_indices:
            # No robot indexing found, that's fine
            return ValidationResult(
                is_valid=True,
                error_type=None,
                error_message=None,
                suggestions=[],
                line_number=None,
                context={"robot_indices_found": []}
            )
        
        # Convert to integers and check bounds
        max_available_index = len(available_robots) - 1
        invalid_indices = []
        
        for idx_str in robot_indices:
            idx = int(idx_str)
            if idx > max_available_index:
                invalid_indices.append(idx)
        
        if invalid_indices:
            return ValidationResult(
                is_valid=False,
                error_type="robot_index_error",
                error_message=f"Robot index out of range: found indices {invalid_indices}, but only {len(available_robots)} robots available (max index: {max_available_index})",
                suggestions=[
                    f"Change robots[{max(invalid_indices)}] to robots[{max_available_index}]" if invalid_indices else "",
                    f"Available robot indices: 0 to {max_available_index}",
                    "Ensure robot indexing matches the number of allocated robots"
                ],
                line_number=None,
                context={
                    "invalid_indices": invalid_indices,
                    "max_valid_index": max_available_index,
                    "available_robots": len(available_robots)
                }
            )
        
        return ValidationResult(
            is_valid=True,
            error_type=None,
            error_message=None,
            suggestions=[],
            line_number=None,
            context={
                "robot_indices_found": [int(idx) for idx in robot_indices],
                "all_indices_valid": True
            }
        )
        
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_type="validation_error",
            error_message=f"Error during robot index validation: {str(e)}",
            suggestions=["Review robot indexing in generated code"],
            line_number=None,
            context=None
        )
