#!/usr/bin/env python3
"""
Test script to validate the new semantic validation functions.
Tests both entity extraction and LLM-based semantic matching.
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to Python path
sys.path.append("/home/shyam/SMART_LLM_clone/SMART-LLM/scripts")

from langgraph_pipeline.utils import (
    extract_entities_from_text, 
    validate_entity_consistency, 
    validate_semantic_match
)

def test_entity_extraction():
    """Test entity extraction functionality."""
    print("🧪 Testing Entity Extraction")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        "Toast a slice of the breadloaf",
        "Put mug in coffee machine", 
        "Slice the tomato and wash the knife",
        "Clean the floor with a broom"
    ]
    
    for task in test_cases:
        entities = extract_entities_from_text(task)
        print(f"Task: '{task}'")
        print(f"Entities: {entities}")
        print()

def test_entity_consistency():
    """Test entity consistency validation."""
    print("🧪 Testing Entity Consistency Validation")
    print("=" * 50)
    
    # Test case 1: Good match (toast bread task)
    task1 = "Toast a slice of the breadloaf"
    code1 = """
def slice_breadloaf(robot):
    GoToObject(robot, 'Knife')
    PickupObject(robot, 'Knife')
    GoToObject(robot, 'Bread')
    SliceObject(robot, 'Bread')

def toast_bread_slice(robot):
    GoToObject(robot, 'BreadSliced')
    PickupObject(robot, 'BreadSliced')
    GoToObject(robot, 'Toaster')
    PutObject(robot, 'BreadSliced', 'Toaster')
    SwitchOn(robot, 'Toaster')
"""
    
    result1 = validate_entity_consistency(task1, code1)
    print(f"Task: '{task1}'")
    print(f"Result: {'✅ VALID' if result1['is_valid'] else '❌ INVALID'}")
    if not result1['is_valid']:
        print(f"Error: {result1['error_message']}")
    print()
    
    # Test case 2: Bad match (toast bread vs clean floor)
    task2 = "Toast a slice of the breadloaf"
    code2 = """
def clean_floor_with_broom(robot):
    GoToObject(robot, 'Broom')
    PickupObject(robot, 'Broom')
    GoToObject(robot, 'Floor')
    CleanObject(robot, 'Floor')
    
def turn_on_light(robot):
    GoToObject(robot, 'LightSwitch')
    SwitchOn(robot, 'LightSwitch')
"""
    
    result2 = validate_entity_consistency(task2, code2)
    print(f"Task: '{task2}'")
    print(f"Result: {'✅ VALID' if result2['is_valid'] else '❌ INVALID'}")
    if not result2['is_valid']:
        print(f"Error: {result2['error_message']}")
    print()

def test_semantic_validation():
    """Test LLM-based semantic validation."""
    print("🧪 Testing LLM-Based Semantic Validation")
    print("=" * 50)
    
    # Get API key
    api_key_file = "/home/shyam/SMART_LLM_clone/SMART-LLM/api_key.txt"
    if not os.path.exists(api_key_file):
        print("❌ API key file not found. Skipping LLM tests.")
        return
        
    with open(api_key_file, 'r') as f:
        api_key = f.read().strip()
    
    # Test case 1: Good match
    task1 = "Put mug in coffee machine"
    code1 = """
def put_mug_in_coffee_machine(robot):
    GoToObject(robot, 'Mug')
    PickupObject(robot, 'Mug')
    GoToObject(robot, 'CoffeeMachine')
    PutObject(robot, 'Mug', 'CoffeeMachine')
"""
    
    print("Testing GOOD match...")
    result1 = validate_semantic_match(task1, code1, api_key)
    print(f"Task: '{task1}'")
    print(f"Result: {'✅ VALID' if result1['is_valid'] else '❌ INVALID'}")
    if not result1['is_valid']:
        print(f"Error: {result1['error_message']}")
    print()
    
    # Test case 2: Bad match (the problematic toast bread case)
    task2 = "Toast a slice of the breadloaf"
    code2 = """
def clean_floor_with_broom(robot):
    GoToObject(robot, 'Broom')
    PickupObject(robot, 'Broom')
    GoToObject(robot, 'Floor')
    CleanObject(robot, 'Floor')
    
def turn_on_light(robot):
    GoToObject(robot, 'LightSwitch')
    SwitchOn(robot, 'LightSwitch')

# Execute SubTask 1
clean_floor_with_broom(robots[0])

# Execute SubTask 2  
turn_on_light(robots[0])
"""
    
    print("Testing BAD match (the bread vs floor issue)...")
    result2 = validate_semantic_match(task2, code2, api_key)
    print(f"Task: '{task2}'")
    print(f"Result: {'✅ VALID' if result2['is_valid'] else '❌ INVALID'}")
    if not result2['is_valid']:
        print(f"Error: {result2['error_message']}")
    print()

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Semantic Validation System")
    print("=" * 60)
    print()
    
    test_entity_extraction()
    test_entity_consistency()
    test_semantic_validation()
    
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main()
