#!/usr/bin/env python3
"""
LangGraph SMART-LLM Main Entry Point using Claude API
Drop-in replacement for main.py but with Claude instead of OpenAI.

Usage:
    python scripts/langgraph_pipeline/main_claude.py --floor-plan 6 --test-set final_test --claude-api-key-file claude_api_key.txt
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
from scripts.langgraph_pipeline.graph_builder_claude import create_langgraph_pipeline_claude
from scripts.langgraph_pipeline.utils import load_training_examples, get_ai2_thor_objects


def main():
    """Main entry point with Claude API support."""
    parser = argparse.ArgumentParser(description="LangGraph SMART-LLM Pipeline with Claude API")
    parser.add_argument("--floor-plan", type=int, required=True, help="Floor plan number")
    parser.add_argument("--claude-api-key-file", type=str, default="claude_api_key.txt", 
                       help="File containing Claude API key")
    parser.add_argument("--decomp-model", type=str, default="claude-sonnet-4-20250514",
                       help="Claude model for decomposition")
    parser.add_argument("--alloc-model", type=str, default="claude-sonnet-4-20250514", 
                       help="Claude model for allocation")
    parser.add_argument("--codegen-model", type=str, default="claude-sonnet-4-20250514",
                       help="Claude model for code generation")
    parser.add_argument("--test-set", type=str, default="final_test",
                       help="Test set directory name")
    parser.add_argument("--prompt-decompose-set", type=str, default="train_task_decompose",
                       choices=["train_task_decompose"], help="Decomposition prompt set")
    parser.add_argument("--prompt-allocation-set", type=str, default="train_task_allocation", 
                       choices=["train_task_allocation"], help="Allocation prompt set")
    
    args = parser.parse_args()
    
    # Load Claude API key
    try:
        api_key_path = Path(args.claude_api_key_file)
        if not api_key_path.exists():
            # Try in the main directory
            api_key_path = Path(f"./{args.claude_api_key_file}")
        
        claude_api_key = api_key_path.read_text().strip()
        print(f"✅ Loaded Claude API key from {api_key_path}")
    except Exception as e:
        print(f"❌ Error loading Claude API key: {e}")
        print("💡 Please create a file with your Claude API key (e.g., claude_api_key.txt)")
        return
    
    print("🚀 LangGraph SMART-LLM Pipeline Starting...")
    print("🤖 Using Claude API instead of OpenAI")
    print("=" * 50)
    
    # Initialize pipeline with Claude
    pipeline = create_langgraph_pipeline_claude(
        claude_api_key, args.decomp_model, args.alloc_model, args.codegen_model
    )
    
    # Load tasks (identical to main.py)
    try:
        with open(f"./data/{args.test_set}/FloorPlan{args.floor_plan}.json", "r") as f:
            lines = f.readlines()
        
        tasks = []
        robots_list = []
        ground_truth_list = []
        trans_list = []
        max_trans_list = []
        
        for line in lines:
            data = json.loads(line.strip())
            tasks.append(data["task"])
            robots_list.append([robots.robots[i-1] for i in data["robot list"]])
            ground_truth_list.append(data["object_states"])
            trans_list.append(data["trans"])
            max_trans_list.append(data["max_trans"])
        
        print(f"📋 Loaded {len(tasks)} tasks from FloorPlan{args.floor_plan}")
        
    except FileNotFoundError:
        print(f"❌ Error: Test file './data/{args.test_set}/FloorPlan{args.floor_plan}.json' not found")
        return
    except Exception as e:
        print(f"❌ Error loading tasks: {e}")
        return
    
    # Load training examples (identical to main.py)
    examples = load_training_examples(args.prompt_decompose_set, args.prompt_allocation_set)
    
    # Get objects for compatibility (identical to main.py)
    objects = get_ai2_thor_objects(args.floor_plan)
    print(f"🎯 Loaded {len(objects)} objects from environment")
    
    # Process tasks through LangGraph with Claude
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
        
        # Store result for saving
        result = {
            'task': task,
            'decomposed_plan': final_state.get('decomposed_plan', ''),
            'allocation_plan': final_state.get('allocation_plan', ''),
            'generated_code': final_state.get('generated_code', ''),
            'success': final_state.get('is_successful', False),
            'final_stage': final_state.get('final_stage_reached', 'unknown'),
            'iterations': final_state.get('iteration_count', 0),
            'errors': final_state.get('errors', [])
        }
        results.append(result)
    
    # Print final summary
    success_rate = (successful_tasks / len(tasks)) * 100 if tasks else 0
    print(f"\n🎉 LangGraph Pipeline Complete!")
    print(f"📊 Results: {successful_tasks}/{len(tasks)} tasks successful ({success_rate:.1f}%)")
    print(f"📁 Results saved to logs/ directory")
    print(f"🚀 Ready for execution with execute_plan.py!")
    
    # Save results (identical to main.py)
    save_results(results, args.floor_plan, objects, ground_truth_list, trans_list, max_trans_list, robots_list)


def save_results(results, floor_plan, objects, ground_truth_list, trans_list, max_trans_list, robots_list):
    """Save results in the same format as multi_llm_pipeline.py"""
    
    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    
    for i, result in enumerate(results):
        if result['success']:
            task_name = result['task'].replace(' ', '_').replace(',', '').replace("'", "")
            log_dir = f"logs/{task_name}_claude_{timestamp}"
            os.makedirs(log_dir, exist_ok=True)
            
            # Save decomposition
            with open(f"{log_dir}/decompose_plan.py", "w") as f:
                f.write(result['decomposed_plan'])
            
            # Save allocation  
            with open(f"{log_dir}/allocation_plan.py", "w") as f:
                f.write(result['allocation_plan'])
            
            # Save code
            with open(f"{log_dir}/code_plan.py", "w") as f:
                f.write(result['generated_code'])
            
            # Save metadata
            metadata = {
                'task': result['task'],
                'floor_plan': floor_plan,
                'objects': len(objects),
                'ground_truth': ground_truth_list[i],
                'trans': trans_list[i],
                'max_trans': max_trans_list[i],
                'api': 'claude',
                'timestamp': timestamp,
                'iterations': result['iterations'],
                'final_stage': result['final_stage']
            }
            
            # Save metadata in EXACT format expected by execute_plan.py (matching OpenAI format)
            with open(f"{log_dir}/log.txt", "w") as f:
                f.write(result['task'])                                    # Line 0: task
                f.write(f"\n")                                             # Line 1: empty
                f.write(f"\nLangGraph Multi-Agent Pipeline (Claude)")      # Line 2: method info
                f.write(f"\n")                                             # Line 3: empty
                f.write(f"\nFloor Plan: {floor_plan}")                     # Line 4: floor plan (execute_plan reads this)
                f.write(f"\n")                                             # Line 5: empty
                f.write(f"\n")                                             # Line 6: empty  
                f.write(f"\nobjects = {objects}")                          # Line 7: objects
                f.write(f"\nrobots = {robots_list[i]}")                    # Line 8: robots (execute_plan reads this)
                f.write(f"\nground_truth = {ground_truth_list[i]}")        # Line 9: ground truth (execute_plan reads this)
                f.write(f"\ntrans = {trans_list[i]}")                      # Line 10: trans (execute_plan reads this)
                f.write(f"\nmax_trans = {max_trans_list[i]}")              # Line 11: max_trans (execute_plan reads this)


if __name__ == "__main__":
    main()
