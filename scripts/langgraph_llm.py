#!/usr/bin/env python3
"""
LangGraph-based Multi-LLM Pipeline for SMART-LLM Framework
Replaces the sequential LLM calls with a graph-based agent system
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

# LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any


class PipelineState(TypedDict):
    """Shared state between all agents in the pipeline"""
    task: str
    objects: List[Dict]
    available_robots: List[Dict]
    ground_truth: List[Dict]
    trans: int
    max_trans: int
    floor_plan: int
    
    # Examples/context
    decomp_examples: str
    alloc_examples: str
    codegen_examples: str
    
    # Stage outputs
    decomposed_plan: str
    allocation_plan: str
    code_plan: str
    
    # Metadata
    current_stage: str
    errors: List[str]


class TaskDecompositionAgent:
    """Specialized agent for breaking down tasks into subtasks"""
    
    def __init__(self, model_name="gpt-4", api_key=None):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0,
            max_tokens=1300
        )
        
    def __call__(self, state: PipelineState) -> PipelineState:
        """Execute task decomposition"""
        print("🔍 Stage 1: Task Decomposition...")
        
        system_prompt = """You are an expert at breaking down complex household tasks into logical subtasks. 
        Focus on task dependencies, required skills, and parallelization opportunities. 
        Analyze what can be done in parallel vs sequentially."""
        
        # Build prompt with context
        prompt = f"from skills import {actions.ai2thor_actions}\n"
        prompt += "import time\nimport threading\n"
        prompt += f"\nobjects = {state['objects']}\n\n"
        prompt += state['decomp_examples']
        prompt += f"\n\n# Task Description: {state['task']}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            decomposed_plan = response.content.strip()
            
            # Update state
            state['decomposed_plan'] = decomposed_plan
            state['current_stage'] = 'allocation'
            
        except Exception as e:
            state['errors'].append(f"Decomposition error: {str(e)}")
            
        return state


class RobotAllocationAgent:
    """Specialized agent for allocating robots to subtasks"""
    
    def __init__(self, model_name="gpt-4", api_key=None):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0,
            max_tokens=400,
            model_kwargs={"frequency_penalty": 0.69}
        )
        
    def __call__(self, state: PipelineState) -> PipelineState:
        """Execute robot allocation"""
        print("🤖 Stage 2: Robot Allocation...")
        
        system_prompt = """You are a robot task allocation specialist. Your expertise is in matching robot 
        capabilities to task requirements, optimizing for efficiency and constraints. Consider both skills 
        and mass capacity when forming teams."""
        
        # Build focused prompt for allocation
        prompt = f"from skills import {actions.ai2thor_actions}\n"
        prompt += "import time\nimport threading\n"
        
        # Include subset of examples to reduce token count
        example_lines = state['alloc_examples'].split('\n')
        short_examples = '\n'.join(example_lines[:100])
        prompt += short_examples + "\n\n"
        
        prompt += state['decomposed_plan']
        prompt += "\n# TASK ALLOCATION\n"
        prompt += f"# Scenario: There are {len(state['available_robots'])} robots available. "
        prompt += "Robots should be assigned to subtasks that match their skills and mass capacity.\n"
        prompt += f"robots = {state['available_robots']}\n"
        
        # Reduce objects list to save tokens
        object_names = [obj['name'] for obj in state['objects']]
        prompt += f"object_types = {object_names}\n"
        prompt += "# SOLUTION\n"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            allocation_plan = response.content.strip()
            
            # Update state
            state['allocation_plan'] = allocation_plan
            state['current_stage'] = 'code_generation'
            
        except Exception as e:
            state['errors'].append(f"Allocation error: {str(e)}")
            
        return state


class CodeGenerationAgent:
    """Specialized agent for generating executable Python code"""
    
    def __init__(self, model_name="gpt-4", api_key=None):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0,
            max_tokens=1400,
            model_kwargs={"frequency_penalty": 0.4}
        )
        
    def __call__(self, state: PipelineState) -> PipelineState:
        """Execute code generation"""
        print("💻 Stage 3: Code Generation...")
        
        system_prompt = """You are a Python code generation expert specializing in robotics. 
        Generate clean, executable code using the provided action APIs. Focus on proper threading, 
        error handling, and efficient execution."""
        
        prompt = f"from skills import {actions.ai2thor_actions}\n"
        prompt += "import time\nimport threading\n"
        
        # Reduce objects to just names for code generation
        object_names = [obj['name'] for obj in state['objects']]
        prompt += f"object_types = {object_names}\n\n"
        
        # Include subset of examples for code generation
        example_lines = state['codegen_examples'].split('\n')
        short_examples = '\n'.join(example_lines[:150])
        prompt += short_examples + "\n\n"
        
        prompt += state['decomposed_plan']
        prompt += "\n# TASK ALLOCATION\n"
        prompt += f"robots = {state['available_robots']}\n"
        prompt += state['allocation_plan']
        prompt += "\n# CODE Solution\n"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            code_plan = response.content.strip()
            
            # Update state
            state['code_plan'] = code_plan
            state['current_stage'] = 'complete'
            
        except Exception as e:
            state['errors'].append(f"Code generation error: {str(e)}")
            
        return state


class LangGraphSMARTLLM:
    """Main LangGraph-based pipeline orchestrator"""
    
    def __init__(self, api_key, decomp_model="gpt-4", alloc_model="gpt-4", codegen_model="gpt-4"):
        self.api_key = api_key
        
        # Initialize specialized agents
        self.decomp_agent = TaskDecompositionAgent(decomp_model, api_key)
        self.alloc_agent = RobotAllocationAgent(alloc_model, api_key)
        self.codegen_agent = CodeGenerationAgent(codegen_model, api_key)
        
        # Build the graph
        self.workflow = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define the state graph
        workflow = StateGraph(PipelineState)
        
        # Add nodes (agents)
        workflow.add_node("decompose", self.decomp_agent)
        workflow.add_node("allocate", self.alloc_agent)
        workflow.add_node("generate_code", self.codegen_agent)
        
        # Define the flow
        workflow.set_entry_point("decompose")
        workflow.add_edge("decompose", "allocate")
        workflow.add_edge("allocate", "generate_code")
        workflow.add_edge("generate_code", END)
        
        return workflow.compile()
    
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
    
    def process_single_task(self, task, available_robots, ground_truth, trans, max_trans, 
                           floor_plan, examples):
        """Process a single task through the LangGraph pipeline"""
        
        # Initialize state
        objects = self.get_ai2_thor_objects(floor_plan)
        
        initial_state = PipelineState(
            task=task,
            objects=objects,
            available_robots=available_robots,
            ground_truth=ground_truth,
            trans=trans,
            max_trans=max_trans,
            floor_plan=floor_plan,
            decomp_examples=examples[0],
            alloc_examples=examples[1],
            codegen_examples=examples[2],
            decomposed_plan="",
            allocation_plan="",
            code_plan="",
            current_stage="decomposition",
            errors=[]
        )
        
        # Execute the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return final_state
    
    def process_tasks(self, tasks, robots_list, ground_truth_list, trans_list, max_trans_list, floor_plan):
        """Process all tasks through the LangGraph pipeline"""
        
        # Load training examples
        examples = self.load_training_examples("train_task_decompose", "train_task_allocation")
        
        # Process each task
        results = []
        for i, task in enumerate(tasks):
            print(f"\n📋 Processing Task {i+1}: {task}")
            
            final_state = self.process_single_task(
                task, robots_list[i], ground_truth_list[i], 
                trans_list[i], max_trans_list[i], floor_plan, examples
            )
            
            # Check for errors
            if final_state['errors']:
                print(f"⚠️ Errors in task {i+1}: {final_state['errors']}")
            
            results.append({
                'task': task,
                'decomposed_plan': final_state['decomposed_plan'],
                'allocation_plan': final_state['allocation_plan'], 
                'code_plan': final_state['code_plan'],
                'robots': robots_list[i],
                'errors': final_state['errors']
            })
        
        return results
    
    def save_results(self, results, floor_plan, ground_truth_list, trans_list, max_trans_list):
        """Save results in exact format expected by execute_plan.py"""
        if not os.path.isdir("./logs/"):
            os.makedirs("./logs/")
            
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        
        objects = self.get_ai2_thor_objects(floor_plan)
        
        for i, result in enumerate(results):
            task_name = "_".join(result['task'].split(' ')).replace('\n', '')
            folder_name = f"{task_name}_langgraph_{date_time}"
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
                f.write(f"\nLangGraph Multi-Agent Pipeline")               # Line 2: method info
                f.write(f"\n")                                             # Line 3: empty
                f.write(f"\nFloor Plan: {floor_plan}")                     # Line 4: floor plan
                f.write(f"\n")                                             # Line 5: empty
                f.write(f"\n")                                             # Line 6: empty  
                f.write(f"\nobjects = {objects}")                          # Line 7: objects
                f.write(f"\nrobots = {result['robots']}")                  # Line 8: robots
                f.write(f"\nground_truth = {ground_truth_list[i]}")        # Line 9: ground truth
                f.write(f"\ntrans = {trans_list[i]}")                      # Line 10: trans
                f.write(f"\nmax_trans = {max_trans_list[i]}")              # Line 11: max_trans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-plan", type=int, required=True)
    parser.add_argument("--openai-api-key-file", type=str, default="api_key")
    parser.add_argument("--decomp-model", type=str, default="gpt-4")
    parser.add_argument("--alloc-model", type=str, default="gpt-4") 
    parser.add_argument("--codegen-model", type=str, default="gpt-4")
    parser.add_argument("--test-set", type=str, default="final_test")
    args = parser.parse_args()
    
    # Load API key
    api_key = Path(f"{args.openai_api_key_file}.txt").read_text().strip()
    
    # Initialize LangGraph pipeline
    pipeline = LangGraphSMARTLLM(
        api_key, args.decomp_model, args.alloc_model, args.codegen_model
    )
    
    # Load test tasks (same format as original)
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
    
    # Process tasks through LangGraph
    results = pipeline.process_tasks(
        tasks, robots_list, ground_truth_list, trans_list, max_trans_list, args.floor_plan
    )
    
    # Save results
    pipeline.save_results(results, args.floor_plan, ground_truth_list, trans_list, max_trans_list)
    
    print(f"\n✅ Processed {len(tasks)} tasks with LangGraph pipeline")
    print("📁 Results saved to logs/ directory")
    print("🚀 Ready for execution with execute_plan.py!")


if __name__ == "__main__":
    main()