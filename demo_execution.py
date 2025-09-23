#!/usr/bin/env python3
"""
Demo execution of "Put mug in coffee machine" task
This demonstrates the complete SMART-LLM workflow and AI2Thor integration
"""

import math
import re
import shutil
import subprocess
import time
import threading
import cv2
import numpy as np
from ai2thor.controller import Controller
from scipy.spatial import distance
from typing import Tuple
from collections import deque
import random
import os
from glob import glob

def closest_node(node, nodes, no_robot, clost_node_location):
    crps = []
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))
    for i in range(no_robot):
        pos_index = dist_indices[(i * 5) + clost_node_location[i]]
        crps.append (nodes[pos_index])
    return crps

def distance_pts(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
    return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

def generate_video():
    frame_rate = 5
    cur_path = os.path.dirname(__file__) + "/*/"
    for imgs_folder in glob(cur_path, recursive = False):
        view = imgs_folder.split('/')[-2]
        if not os.path.isdir(imgs_folder):
            print("The input path: {} you specified does not exist.".format(imgs_folder))
        else:
            command_set = ['ffmpeg', '-i',
                                '{}/img_%05d.png'.format(imgs_folder), 
                                '-framerate', str(frame_rate),
                                '-pix_fmt', 'yuv420p',
                                '{}/video_{}.mp4'.format(os.path.dirname(__file__), view)]
            subprocess.call(command_set)

# Robot configuration - using robot1 with all skills
robots = [{'name': 'robot1', 'skills': ['GoToObject', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'SwitchOn', 'SwitchOff', 'PickupObject', 'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject', 'PullObject'], 'mass': 100}]

# Floor plan and ground truth
floor_no = 21
ground_truth = [{'name': 'CoffeeMachine', 'contains': ['Mug'], 'state': None}]
no_trans_gt = 0
max_trans = 0

# Execution tracking
total_exec = 0
success_exec = 0

print("🚀 Starting SMART-LLM Demo: Put mug in coffee machine")
print("📍 Floor Plan:", floor_no)
print("🤖 Robots:", len(robots))

# Initialize AI2Thor Controller
c = Controller(height=1000, width=1000)
c.reset("FloorPlan" + str(floor_no)) 
no_robot = len(robots)

print("🏠 Environment initialized")

# Initialize multi-agent scene
multi_agent_event = c.step(dict(action='Initialize', agentMode="default", snapGrid=False, gridSize=0.5, rotateStepDegrees=20, visibilityDistance=100, fieldOfView=90, agentCount=no_robot))

# Add top view camera
event = c.step(action="GetMapViewCameraProperties")
event = c.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

# Get reachable positions
reachable_positions_ = c.step(action="GetReachablePositions").metadata["actionReturn"]
reachable_positions = positions_tuple = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]

print("🗺️  Reachable positions:", len(reachable_positions))

# Randomize positions of the agents
for i in range(no_robot):
    init_pos = random.choice(reachable_positions_)
    c.step(dict(action="Teleport", position=init_pos, agentId=i))
    print(f"🤖 Robot {i+1} spawned at position: ({init_pos['x']:.2f}, {init_pos['y']:.2f}, {init_pos['z']:.2f})")
    
objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
print(f"🎯 Objects in scene: {len(objs)}")

# Look down for better object detection
for i in range(no_robot):
    multi_agent_event = c.step(action="LookDown", degrees=35, agentId=i)

action_queue = []
task_over = False
recp_id = None

def exec_actions():
    global total_exec, success_exec
    # Create folders to save images
    for i in range(no_robot):
        folder_name = "agent_" + str(i+1)
        folder_path = os.path.dirname(__file__) + "/" + folder_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    folder_name = "top_view"
    folder_path = os.path.dirname(__file__) + "/" + folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    img_counter = 0
    
    print("🎬 Starting action execution and image capture...")
    
    while not task_over:
        if len(action_queue) > 0:
            try:
                act = action_queue[0]
                print(f"⚡ Executing: {act['action']} (Agent {act['agent_id']})")
                
                if act['action'] == 'ObjectNavExpertAction':
                    multi_agent_event = c.step(dict(action=act['action'], position=act['position'], agentId=act['agent_id']))
                    next_action = multi_agent_event.metadata['actionReturn']
                    if next_action != None:
                        multi_agent_event = c.step(action=next_action, agentId=act['agent_id'], forceAction=True)
                
                elif act['action'] == 'PickupObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="PickupObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print(f"❌ Error: {multi_agent_event.metadata['errorMessage']}")
                    else:
                        success_exec += 1
                        print(f"✅ Successfully picked up {act['objectId']}")

                elif act['action'] == 'PutObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="PutObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print(f"❌ Error: {multi_agent_event.metadata['errorMessage']}")
                    else:
                        success_exec += 1
                        print(f"✅ Successfully put {act['objectId']} in target")
                
                elif act['action'] == 'RotateLeft':
                    multi_agent_event = c.step(action="RotateLeft", degrees=act['degrees'], agentId=act['agent_id'])
                    
                elif act['action'] == 'RotateRight':
                    multi_agent_event = c.step(action="RotateRight", degrees=act['degrees'], agentId=act['agent_id'])
                
                elif act['action'] == 'Done':
                    multi_agent_event = c.step(action="Done")
                    print("🏁 Task completed!")
                    
            except Exception as e:
                print(f"💥 Exception during execution: {e}")
                
            # Save images
            for i, e in enumerate(multi_agent_event.events):
                f_name = os.path.dirname(__file__) + "/agent_" + str(i+1) + "/img_" + str(img_counter).zfill(5) + ".png"
                cv2.imwrite(f_name, e.cv2img)
            
            top_view_rgb = cv2.cvtColor(c.last_event.events[0].third_party_camera_frames[-1], cv2.COLOR_BGR2RGB)
            f_name = os.path.dirname(__file__) + "/top_view/img_" + str(img_counter).zfill(5) + ".png"
            cv2.imwrite(f_name, top_view_rgb)
            
            img_counter += 1    
            action_queue.pop(0)
        
        time.sleep(0.1)

# Start action execution thread
actions_thread = threading.Thread(target=exec_actions)
actions_thread.start()

def GoToObject(robots, dest_obj):
    print(f"🎯 Going to object: {dest_obj}")
    
    if not isinstance(robots, list):
        robots = [robots]
    no_agents = len(robots)
    
    dist_goals = [10.0] * len(robots)
    prev_dist_goals = [10.0] * len(robots)
    count_since_update = [0] * len(robots)
    clost_node_location = [0] * len(robots)
    
    # Find object in scene
    objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
    objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])
    
    dest_obj_id = None
    dest_obj_center = None
    
    for idx, obj in enumerate(objs):
        if dest_obj.lower() in obj.lower():
            dest_obj_id = obj
            dest_obj_center = objs_center[idx]
            if dest_obj_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                break
    
    if dest_obj_id is None:
        print(f"❌ Object '{dest_obj}' not found in scene")
        return
    
    print(f"🎯 Found target: {dest_obj_id} at {dest_obj_center}")
    
    dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']]
    crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
    goal_thresh = 1.0
    
    while all(d > goal_thresh for d in dist_goals):
        for ia, robot in enumerate(robots):
            robot_name = robot['name']
            agent_id = int(robot_name[-1]) - 1
            
            metadata = c.last_event.events[agent_id].metadata
            location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
                "horizon": metadata["agent"]["cameraHorizon"]}
            
            prev_dist_goals[ia] = dist_goals[ia]
            dist_goals[ia] = distance_pts([location['x'], location['y'], location['z']], crp[ia])
            
            dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
            
            if dist_del < 0.2:
                count_since_update[ia] += 1
            else:
                count_since_update[ia] = 0
                
            if count_since_update[ia] < 8:
                action_queue.append({'action':'ObjectNavExpertAction', 'position':dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]), 'agent_id':agent_id})
            else:
                clost_node_location[ia] += 1
                count_since_update[ia] = 0
                crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
            time.sleep(0.5)
    
    print(f"✅ Reached target: {dest_obj}")

def PickupObject(robot, pick_obj):
    print(f"🤏 Picking up: {pick_obj}")
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    pick_obj_id = None
    for obj in objs:
        if pick_obj.lower() in obj.lower():
            pick_obj_id = obj
            break
    
    if pick_obj_id:
        action_queue.append({'action':'PickupObject', 'objectId':pick_obj_id, 'agent_id':agent_id})
        time.sleep(1)
    else:
        print(f"❌ Could not find {pick_obj} to pick up")

def PutObject(robot, put_obj, recp):
    print(f"📍 Putting {put_obj} in {recp}")
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    recp_obj_id = None
    for obj in objs:
        if recp.lower() in obj.lower():
            recp_obj_id = obj
            break
    
    if recp_obj_id:
        action_queue.append({'action':'PutObject', 'objectId':recp_obj_id, 'agent_id':agent_id})
        time.sleep(1)
    else:
        print(f"❌ Could not find {recp} as receptacle")

# MAIN TASK EXECUTION
print("\n🎬 Starting Task Execution: Put mug in coffee machine")
print("=" * 60)

def put_mug_in_coffee_machine(robot):
    print("📋 Task: Put mug in coffee machine")
    print("🤖 Using robot:", robot['name'])
    
    # 1: Go to the Mug
    print("Step 1: Going to Mug...")
    GoToObject(robot, 'Mug')
    
    # 2: Pick up the Mug
    print("Step 2: Picking up Mug...")
    PickupObject(robot, 'Mug')
    
    # 3: Go to the CoffeeMachine
    print("Step 3: Going to CoffeeMachine...")
    GoToObject(robot, 'CoffeeMachine')
    
    # 4: Put the Mug in the CoffeeMachine
    print("Step 4: Putting Mug in CoffeeMachine...")
    PutObject(robot, 'Mug', 'CoffeeMachine')
    
    print("✅ Task sequence completed!")

# Execute the task
put_mug_in_coffee_machine(robots[0])

# End task
print("\n🏁 Ending task execution...")
for i in range(5):
    action_queue.append({'action':'Done'})
    time.sleep(0.1)

task_over = True
time.sleep(3)

# Calculate metrics
if total_exec == 0:
    exec_rate = 0.0
else:
    exec_rate = float(success_exec) / float(total_exec)

print("\n📊 EXECUTION METRICS")
print("=" * 40)
print(f"Total Executions: {total_exec}")
print(f"Successful Executions: {success_exec}")
print(f"Execution Rate: {exec_rate:.2%}")

# Check ground truth
objs = list([obj for obj in c.last_event.metadata["objects"]])
gcr_tasks = 0.0
gcr_complete = 0.0

for obj_gt in ground_truth:
    obj_name = obj_gt['name']
    contains = obj_gt['contains']
    gcr_tasks += 1
    
    for obj in objs:
        if obj_name.lower() in obj["name"].lower():
            print(f"🔍 Checking {obj['name']} for contained objects...")
            if obj['receptacleObjectIds'] is not None:
                contained_objects = [c.last_event.get_object(obj_id)['objectType'] for obj_id in obj['receptacleObjectIds']]
                print(f"📦 Contains: {contained_objects}")
                
                for req_obj in contains:
                    if any(req_obj.lower() in contained.lower() for contained in contained_objects):
                        gcr_complete += 1
                        print(f"✅ Found {req_obj} in {obj_name}")
            break

if gcr_tasks == 0:
    gcr = 1
else:
    gcr = gcr_complete / gcr_tasks

tc = 1 if gcr == 1.0 else 0

print(f"Ground Truth Completion Rate (GCR): {gcr:.2%}")
print(f"Task Completion (TC): {tc}")

no_trans = 1  # This task has 1 transformation (put mug in coffee machine)
if max_trans == no_trans_gt and no_trans_gt == no_trans:
    ru = 1
elif max_trans == no_trans_gt:
    ru = 0
else:
    ru = (max_trans - no_trans) / (max_trans - no_trans_gt)

sr = 1 if tc == 1 and ru == 1 else 0

print(f"Resource Utilization (RU): {ru}")
print(f"Success Rate (SR): {sr}")

print("\n🎥 Generating video...")
generate_video()

print("\n🎉 Demo completed!")
print("📁 Check the generated images in agent_1/ and top_view/ folders")
print("🎬 Check for generated videos")
