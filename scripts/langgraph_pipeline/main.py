#!/usr/bin/env python3
"""
LangGraph SMART-LLM Main Entry Point
Drop-in replacement for multi_llm_pipeline.py with iterative refinement capabilities.

Usage:
    python -m scripts.langgraph_pipeline.main --floor-plan 21 --test-set final_test
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(".")

import resources.actions as actions
import resources.robots as robots
from scripts.langgraph_pipeline.graph_builder import create_langgraph_pipeline
from scripts.langgraph_pipeline.utils import load_training_examples, get_ai2_thor_objects


def main():
    """Main entry point with identical interface to multi_llm_pipeline.py"""
    
    parser = argparse.ArgumentParser(description="LangGraph SMART-LLM Pipeline")
    parser.add_argument("--floor-plan", type=int, required=True, help="Floor plan ID")
    parser.add_argument("--openai-api-key-file", type=str, default="api_key", help="OpenAI API key file")
    parser.add_argument("--decomp-model", type=str, default="gpt-4", help="Model for task decomposition")
    parser.add_argument("--alloc-model", type=str, default="gpt-4", help="Model for robot allocation") 
    parser.add_argument("--codegen-model", type=str, default="gpt-4", help="Model for code generation")
    parser.add_argument("--test-set", type=str, default="final_test", help="Test dataset")
    parser.add_argument("--prompt-decompse-set", type=str, default="train_task_decompose", 
                        choices=['train_task_decompose'])
    parser.add_argument("--prompt-allocation-set", type=str, default="train_task_allocation", 
                        choices=['train_task_allocation'])
    args = parser.parse_args()
    
    print("🚀 LangGraph SMART-LLM Pipeline Starting...")
    print("=" * 50)
    
    # Load API key (identical to multi_llm_pipeline.py)
    try:
        api_key = Path(f"{args.openai_api_key_file}.txt").read_text().strip()
    except FileNotFoundError:
        print(f"❌ Error: API key file '{args.openai_api_key_file}.txt' not found")
        return
    
    # Initialize LangGraph pipeline
    pipeline = create_langgraph_pipeline(
        api_key, args.decomp_model, args.alloc_model, args.codegen_model
    )
    
    # Load test tasks (identical to multi_llm_pipeline.py)
    tasks = []
    robots_list = []
    ground_truth_list = []
    trans_list = []
    max_trans_list = []
    
    test_file = f"./data/{args.test_set}/FloorPlan{args.floor_plan}.json"
    try:
        with open(test_file, "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                tasks.append(list(data.values())[0])
                robot_ids = list(data.values())[1]
                ground_truth_list.append(list(data.values())[2])
                trans_list.append(list(data.values())[3])
                max_trans_list.append(list(data.values())[4])
                
                # Convert robot IDs to robot configs (identical to multi_llm_pipeline.py)
                task_robots = []
                for i, r_id in enumerate(robot_ids):
                    rob = robots.robots[r_id-1].copy()
                    rob['name'] = f'robot{i+1}'
                    task_robots.append(rob)
                robots_list.append(task_robots)
    except FileNotFoundError:
        print(f"❌ Error: Test file '{test_file}' not found")
        return
    
    print(f"📋 Loaded {len(tasks)} tasks from FloorPlan{args.floor_plan}")
    
    # Load training examples (identical to multi_llm_pipeline.py)
    examples = load_training_examples(args.prompt_decompse_set, args.prompt_allocation_set)
    
    # Get objects for compatibility (identical to multi_llm_pipeline.py)
    objects = get_ai2_thor_objects(args.floor_plan)
    print(f"🎯 Loaded {len(objects)} objects from environment")
    
    # Process tasks through LangGraph
    results = []
    successful_tasks = 0
    
    for i, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"📋 Task {i+1}/{len(tasks)}: {task}")
        print(f"🤖 Robots: {len(robots_list[i])}")
        
        # Process task with objects included
        final_state = pipeline.process_single_task(
            task, robots_list[i], objects, ground_truth_list[i], 
            trans_list[i], max_trans_list[i], args.floor_plan, examples
        )
        
        # Track success
        if final_state.get('is_successful', False):
            successful_tasks += 1
        
        results.append({
            'task': task,
            'decomposed_plan': final_state.get('decomposed_plan', ''),
            'allocation_plan': final_state.get('allocation_plan', ''), 
            'code_plan': final_state.get('generated_code', ''),
            'robots': robots_list[i],
            'errors': final_state.get('errors', []),
            'final_state': final_state
        })
    
    # Save results in IDENTICAL format to multi_llm_pipeline.py
    save_results(results, args.floor_plan, objects, ground_truth_list, trans_list, max_trans_list)
    
    # Print summary
    success_rate = (successful_tasks / len(tasks)) * 100 if tasks else 0
    print(f"\n🎉 LangGraph Pipeline Complete!")
    print(f"📊 Results: {successful_tasks}/{len(tasks)} tasks successful ({success_rate:.1f}%)")
    print("📁 Results saved to logs/ directory")
    print("🚀 Ready for execution with execute_plan.py!")


def save_results(results, floor_plan, objects, ground_truth_list, trans_list, max_trans_list):
    """Save results in IDENTICAL format to multi_llm_pipeline.py for compatibility"""
    
    if not os.path.isdir("./logs/"):
        os.makedirs("./logs/")
        
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    
    for i, result in enumerate(results):
        task_name = "_".join(result['task'].split(' ')).replace('\n', '')
        folder_name = f"{task_name}_langgraph_{date_time}"
        os.mkdir(f"./logs/{folder_name}")
        
        # Save each stage output (identical structure)
        with open(f"./logs/{folder_name}/decomposed_plan.py", 'w') as f:
            f.write(result['decomposed_plan'])
            
        with open(f"./logs/{folder_name}/allocated_plan.py", 'w') as f:
            f.write(result['allocation_plan'])
            
        with open(f"./logs/{folder_name}/code_plan.py", 'w') as f:
            f.write(result['code_plan'])
        
        # Save metadata in EXACT format expected by execute_plan.py
        with open(f"./logs/{folder_name}/log.txt", 'w') as f:
            f.write(result['task'])                                    # Line 0: task
            f.write(f"\n")                                             # Line 1: empty
            f.write(f"\nLangGraph Multi-Agent Pipeline")               # Line 2: method info
            f.write(f"\n")                                             # Line 3: empty
            f.write(f"\nFloor Plan: {floor_plan}")                     # Line 4: floor plan (execute_plan reads this)
            f.write(f"\n")                                             # Line 5: empty
            f.write(f"\n")                                             # Line 6: empty  
            f.write(f"\nobjects = {objects}")                          # Line 7: objects
            f.write(f"\nrobots = {result['robots']}")                  # Line 8: robots (execute_plan reads this)
            f.write(f"\nground_truth = {ground_truth_list[i]}")        # Line 9: ground truth (execute_plan reads this)
            f.write(f"\ntrans = {trans_list[i]}")                      # Line 10: trans (execute_plan reads this)
            f.write(f"\nmax_trans = {max_trans_list[i]}")              # Line 11: max_trans (execute_plan reads this)


if __name__ == "__main__":
    main()
