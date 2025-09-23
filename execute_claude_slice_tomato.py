#!/usr/bin/env python3
"""
Manual execution script for Claude-generated 'Slice the tomato' task
Based on demo_execution.py structure
"""

import ai2thor
import ai2thor.controller
import time
import threading
from collections import defaultdict
from data.aithor_connect.aithor_connect import *
from data.aithor_connect.end_thread import *
from data.aithor_connect.imports_aux_fn import *

# Task metadata
task_name = "Slice the tomato"
floor_plan = 6
ground_truth = [{'name': 'Tomato', 'contains': [], 'state': 'SLICED'}]
no_trans_gt = 0
max_trans = 1

# Robot configuration
robots = [{'name': 'robot1', 'skills': ['GoToObject', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'SwitchOn', 'SwitchOff', 'PickupObject', 'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject', 'PullObject'], 'mass': 100}]

# Initialize AI2Thor controller
controller = ai2thor.controller.Controller(scene=f"FloorPlan{floor_plan}")

# Initialize robots
robot_controllers = []
for i, robot in enumerate(robots):
    robot_controller = RobotController(controller, f"robot{i+1}")
    robot_controllers.append(robot_controller)

print(f"🚀 Executing Claude-generated task: {task_name}")
print(f"🏠 Floor Plan: {floor_plan}")
print(f"🤖 Robots: {len(robots)}")

# Action queue and metrics
action_queue = []
action_results = defaultdict(list)
start_time = time.time()

# Define action functions from Claude-generated code
def GoToObject(robot, object_name):
    try:
        robot.GotoObject(object_name)
        action_results['goto'].append(True)
    except Exception as e:
        print(f"Error going to {object_name}: {e}")
        action_results['goto'].append(False)

def PickupObject(robot, object_name):
    try:
        robot.PickupObject(object_name)
        action_results['pickup'].append(True)
    except Exception as e:
        print(f"Error picking up {object_name}: {e}")
        action_results['pickup'].append(False)

def SliceObject(robot, object_name):
    try:
        robot.SliceObject(object_name)
        action_results['slice'].append(True)
    except Exception as e:
        print(f"Error slicing {object_name}: {e}")
        action_results['slice'].append(False)

def PutObject(robot, object_name, target_name):
    try:
        robot.PutObject(object_name, target_name)
        action_results['put'].append(True)
    except Exception as e:
        print(f"Error putting {object_name} on {target_name}: {e}")
        action_results['put'].append(False)

# Claude-generated task implementation
def slice_tomato(robot):
    print("🔪 Starting tomato slicing task...")
    
    # 1: Go to the Knife.
    print("Step 1: Going to Knife")
    GoToObject(robot, 'Knife')
    
    # 2: Pick up the Knife.
    print("Step 2: Picking up Knife")
    PickupObject(robot, 'Knife')
    
    # 3: Go to the Tomato.
    print("Step 3: Going to Tomato")
    GoToObject(robot, 'Tomato')
    
    # 4: Slice the Tomato.
    print("Step 4: Slicing Tomato")
    SliceObject(robot, 'Tomato')
    
    # 5: Go to the CounterTop.
    print("Step 5: Going to CounterTop")
    GoToObject(robot, 'CounterTop')
    
    # 6: Put the Knife back on the CounterTop.
    print("Step 6: Putting Knife on CounterTop")
    PutObject(robot, 'Knife', 'CounterTop')
    
    print("✅ Task completed!")

# Execute the task
try:
    slice_tomato(robot_controllers[0])
    
    # Calculate metrics
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Check ground truth
    objects = controller.last_event.metadata["objects"]
    tomato_objects = [obj for obj in objects if obj["objectType"] == "Tomato"]
    
    success = False
    if tomato_objects:
        # Check if any tomato is sliced
        for tomato in tomato_objects:
            if tomato.get("isSliced", False):
                success = True
                break
    
    print(f"\n🎯 EXECUTION RESULTS:")
    print(f"Task: {task_name}")
    print(f"Success: {'✅' if success else '❌'}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Action Results: {dict(action_results)}")
    
    if success:
        print("🎉 Claude-generated task executed successfully!")
    else:
        print("⚠️  Task execution completed but may not have achieved ground truth")
        
finally:
    controller.stop()
    print("🏁 AI2Thor controller stopped")
