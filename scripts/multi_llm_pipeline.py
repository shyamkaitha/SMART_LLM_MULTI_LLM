#!/usr/bin/env python3
"""
Multi-LLM Pipeline for SMART-LLM Framework
This module implements a 3-stage pipeline using specialized LLMs for:
1. Task Decomposition
2. Robot Allocation  
3. Code Generation
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from openai import OpenAI
import ai2thor.controller
import sys
sys.path.append(".")

import resources.actions as actions
import resources.robots as robots


class LLMClient:
    """Specialized LLM client for different stages of the pipeline"""
    
    def __init__(self, model_name, role, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.role = role
        self.system_prompts = {
            "Task Decomposition Specialist": "You are an expert at breaking down complex household tasks into logical subtasks. Focus on task dependencies, required skills, and parallelization opportunities.",
            "Robot Allocation Expert": "You are a robot task allocation specialist. Your expertise is in matching robot capabilities to task requirements, optimizing for efficiency and constraints.",
            "Code Generation Specialist": "You are a Python code generation expert specializing in robotics. Generate clean, executable code using the provided action APIs."
        }
    
    def call_llm(self, prompt, max_tokens=1000, temperature=0, frequency_penalty=0):
        """Make a call to the specialized LLM"""
        if isinstance(prompt, str):
            # Convert string to messages format
            messages = [
                {"role": "system", "content": self.system_prompts[self.role]},
                {"role": "user", "content": prompt}
            ]
        else:
            # Already in messages format, add system prompt
            messages = [{"role": "system", "content": self.system_prompts[self.role]}] + prompt
            
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            frequency_penalty=frequency_penalty
        )
        return response, response.choices[0].message.content.strip()


class MultiLLMPipeline:
    """Main pipeline orchestrator for multi-LLM task planning"""
    
    def __init__(self, api_key, decomp_model="gpt-4", alloc_model="gpt-4", codegen_model="gpt-3.5-turbo"):
        self.decomposition_llm = LLMClient(decomp_model, "Task Decomposition Specialist", api_key)
        self.allocation_llm = LLMClient(alloc_model, "Robot Allocation Expert", api_key)
        self.codegen_llm = LLMClient(codegen_model, "Code Generation Specialist", api_key)
        
    def load_training_examples(self, decomp_prompt_set="train_task_decompose", alloc_prompt_set="train_task_allocation"):
        """Load training examples for each stage"""
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
    
    def get_ai2_thor_objects(self, floor_plan_id):
        """Get objects from AI2Thor environment"""
        controller = ai2thor.controller.Controller(scene=f"FloorPlan{floor_plan_id}")
        objs = [{"name": obj["objectType"], "mass": obj["mass"]} 
                for obj in controller.last_event.metadata["objects"]]
        controller.stop()
        return objs
    
    def stage1_decompose_task(self, task, objects, examples):
        """Stage 1: Task Decomposition using specialized LLM"""
        print("Stage 1: Task Decomposition...")
        
        # Build prompt with examples and context
        prompt = f"from skills import {actions.ai2thor_actions}\n"
        prompt += "import time\nimport threading\n"
        prompt += f"\nobjects = {objects}\n\n"
        prompt += examples
        prompt += f"\n\n# Task Description: {task}"
        
        _, decomposed_plan = self.decomposition_llm.call_llm(
            prompt, max_tokens=1300, frequency_penalty=0.0
        )
        
        return decomposed_plan
    
    def stage2_allocate_robots(self, decomposed_plan, available_robots, objects, examples):
        """Stage 2: Robot Allocation using specialized LLM"""
        print("Stage 2: Robot Allocation...")
        
        # Use a shorter, more focused prompt for allocation
        prompt = f"from skills import {actions.ai2thor_actions}\n"
        prompt += "import time\nimport threading\n"
        
        # Include only a subset of examples to reduce token count
        example_lines = examples.split('\n')
        # Take first 100 lines of examples to stay within context limits
        short_examples = '\n'.join(example_lines[:100])
        prompt += short_examples + "\n\n"
        
        prompt += decomposed_plan
        prompt += "\n# TASK ALLOCATION\n"
        prompt += f"# Scenario: There are {len(available_robots)} robots available. "
        prompt += "Robots should be assigned to subtasks that match their skills and mass capacity.\n"
        prompt += f"robots = {available_robots}\n"
        
        # Reduce objects list to just names to save tokens
        object_names = [obj['name'] for obj in objects]
        prompt += f"object_types = {object_names}\n"
        prompt += "# SOLUTION\n"
        
        # Use same max_tokens as original framework (400 for GPT-4 allocation)
        _, allocation_plan = self.allocation_llm.call_llm(
            prompt, max_tokens=400, frequency_penalty=0.69
        )
        
        return allocation_plan
    
    def stage3_generate_code(self, decomposed_plan, allocation_plan, available_robots, objects, examples):
        """Stage 3: Code Generation using specialized LLM"""
        print("Stage 3: Code Generation...")
        
        # PRIORITY: Training examples FIRST - LLM learns from patterns
        example_lines = examples.split('\n')
        # Use ALL clean examples (88 lines) - they're already minimal and perfect
        prompt = '\n'.join(example_lines) + "\n\n"
        
        # Add only essential context
        prompt += "# FOLLOW THE PATTERNS ABOVE EXACTLY - NO DEVIATIONS\n"
        prompt += "# Robot parameter FIRST: GoToObject(robot, 'Object')\n"
        prompt += "# Function signature: def task_name(robot):\n"
        prompt += "# Execution: task_name(robots[0])\n\n"
        
        # Add current task context
        prompt += f"# NEW TASK TO SOLVE:\n{decomposed_plan}\n"
        prompt += f"# AVAILABLE ROBOTS: {available_robots}\n"
        prompt += f"# ALLOCATION: {allocation_plan}\n"
        prompt += "# GENERATE CODE FOLLOWING THE EXACT PATTERNS ABOVE:\n"
        
        # Use same max_tokens as original framework (1400 for code generation)
        _, code_plan = self.codegen_llm.call_llm(
            prompt, max_tokens=1000, frequency_penalty=0.8
        )
        
        return code_plan
    
    def process_tasks(self, tasks, robots_list, floor_plan):
        """Process all tasks through the 3-stage pipeline"""
        
        # Load training examples with command line arguments
        decomp_examples, alloc_examples, codegen_examples = self.load_training_examples(
            "train_task_decompose", "train_task_allocation"
        )
        
        # Get environment objects
        objects = self.get_ai2_thor_objects(floor_plan)
        
        # Process each task
        results = []
        for i, task in enumerate(tasks):
            print(f"\nProcessing Task {i+1}: {task}")
            
            # Stage 1: Decomposition
            decomposed_plan = self.stage1_decompose_task(task, objects, decomp_examples)
            
            # Stage 2: Allocation  
            available_robots = robots_list[i]
            allocation_plan = self.stage2_allocate_robots(
                decomposed_plan, available_robots, objects, alloc_examples
            )
            
            # Stage 3: Code Generation
            code_plan = self.stage3_generate_code(
                decomposed_plan, allocation_plan, available_robots, objects, codegen_examples
            )
            
            results.append({
                'task': task,
                'decomposed_plan': decomposed_plan,
                'allocation_plan': allocation_plan, 
                'code_plan': code_plan,
                'robots': available_robots
            })
        
        return results
    
    def save_results(self, results, floor_plan, objects, ground_truth_list, trans_list, max_trans_list):
        """Save results in the exact same format as original pipeline for compatibility"""
        if not os.path.isdir("./logs/"):
            os.makedirs("./logs/")
            
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        
        for i, result in enumerate(results):
            task_name = "_".join(result['task'].split(' ')).replace('\n', '')
            folder_name = f"{task_name}_plans_{date_time}"
            os.mkdir(f"./logs/{folder_name}")
            
            # Save each stage output
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
                f.write(f"\nMulti-LLM Pipeline")                           # Line 2: method info
                f.write(f"\n")                                             # Line 3: empty
                f.write(f"\nFloor Plan: {floor_plan}")                     # Line 4: floor plan (execute_plan reads this)
                f.write(f"\n")                                             # Line 5: empty
                f.write(f"\n")                                             # Line 6: empty  
                f.write(f"\nobjects = {objects}")                          # Line 7: objects
                f.write(f"\nrobots = {result['robots']}")                  # Line 8: robots (execute_plan reads this)
                f.write(f"\nground_truth = {ground_truth_list[i]}")        # Line 9: ground truth (execute_plan reads this)
                f.write(f"\ntrans = {trans_list[i]}")                      # Line 10: trans (execute_plan reads this)
                f.write(f"\nmax_trans = {max_trans_list[i]}")              # Line 11: max_trans (execute_plan reads this)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-plan", type=int, required=True)
    parser.add_argument("--openai-api-key-file", type=str, default="api_key")
    parser.add_argument("--decomp-model", type=str, default="gpt-4")
    parser.add_argument("--alloc-model", type=str, default="gpt-4") 
    parser.add_argument("--codegen-model", type=str, default="gpt-4")
    parser.add_argument("--test-set", type=str, default="final_test")
    parser.add_argument("--prompt-decompse-set", type=str, default="train_task_decompose", 
                        choices=['train_task_decompose'])
    parser.add_argument("--prompt-allocation-set", type=str, default="train_task_allocation", 
                        choices=['train_task_allocation'])
    args = parser.parse_args()
    
    # Load API key
    api_key = Path(f"{args.openai_api_key_file}.txt").read_text().strip()
    
    # Initialize pipeline
    pipeline = MultiLLMPipeline(
        api_key, args.decomp_model, args.alloc_model, args.codegen_model
    )
    
    # Load test tasks
    tasks = []
    robots_list = []
    ground_truth_list = []
    trans_list = []
    max_trans_list = []
    
    with open(f"./data/{args.test_set}/FloorPlan{args.floor_plan}.json", "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tasks.append(list(data.values())[0])
            robot_ids = list(data.values())[1]
            ground_truth_list.append(list(data.values())[2])
            trans_list.append(list(data.values())[3])
            max_trans_list.append(list(data.values())[4])
            
            # Convert robot IDs to robot configs
            task_robots = []
            for i, r_id in enumerate(robot_ids):
                rob = robots.robots[r_id-1].copy()
                rob['name'] = f'robot{i+1}'
                task_robots.append(rob)
            robots_list.append(task_robots)
    
    # Get objects for compatibility
    objects = pipeline.get_ai2_thor_objects(args.floor_plan)
    
    # Process tasks
    results = pipeline.process_tasks(tasks, robots_list, args.floor_plan)
    
    # Save results with all required metadata
    pipeline.save_results(results, args.floor_plan, objects, ground_truth_list, trans_list, max_trans_list)
    
    print(f"\nProcessed {len(tasks)} tasks with multi-LLM pipeline")
    print("Results saved to logs/ directory")


if __name__ == "__main__":
    main()
