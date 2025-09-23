# CLEAN TRAINING EXAMPLES - CONSISTENT PATTERNS ONLY

# Task: Throw the Spatula in the trash
def throw_spatula_in_trash(robot):
    # 0: SubTask 1: Throw the Spatula in the trash
    # 1: Go to the Spatula using robot.
    GoToObject(robot,'Spatula')
    # 2: Pick up the Spatula using robot.
    PickupObject(robot,'Spatula')
    # 3: Go to the GarbageCan using robot.
    GoToObject(robot,'GarbageCan')
    # 4: Throw the Spatula in the GarbageCan using robot.
    ThrowObject(robot,'Spatula')

# Execute SubTask 1 with robot1
throw_spatula_in_trash(robots[0])

# Task: Slice the tomato
def slice_tomato(robot):
    # Go to Knife
    GoToObject(robot, 'Knife')
    # Pick up Knife
    PickupObject(robot, 'Knife')
    # Go to Tomato
    GoToObject(robot, 'Tomato')
    # Slice Tomato
    SliceObject(robot, 'Tomato')
    # Put Knife back on CounterTop
    PutObject(robot, 'Knife','CounterTop')

# Assign the task to robot1
slice_tomato(robots[0])

# Task: Wash the lettuce and place lettuce on the Countertop
def wash_lettuce_and_handoff(robot):
    # 0: SubTask 1: Wash the Lettuce and prepare for handoff
    # 1: Go to the Lettuce using robot.
    GoToObject(robot, 'Lettuce')
    # 2: Pick up the Lettuce using robot.
    PickupObject(robot, 'Lettuce')
    # 3: Go to the Sink using robot.
    GoToObject(robot, 'Sink')
    # 4: Put the Lettuce inside the sink using robot
    PutObject(robot, 'Lettuce', 'Sink')
    # 5: Switch on the Faucet to clean the Lettuce using robot
    SwitchOn(robot, 'Faucet')
    # 6: Wait for a while to let the Lettuce clean.
    time.sleep(5)
    # 7: Switch off the Faucet using robot
    SwitchOff(robot, 'Faucet')
    # 8: Pick up the clean Lettuce using robot.
    PickupObject(robot, 'Lettuce')
    # 9: Go to the CounterTop for handoff using robot.
    GoToObject(robot, 'CounterTop')
    # 10: Place the Lettuce on the CounterTop using robot
    PutObject(robot, 'Lettuce', 'CounterTop')

# Execute SubTask 1 with Robot1 (handles full task in single robot)
wash_lettuce_and_handoff(robots[0])

# Task: Simple object interaction
def simple_task(robot):
    GoToObject(robot, 'Object')
    PickupObject(robot, 'Object')
    GoToObject(robot, 'Container')
    PutObject(robot, 'Object', 'Container')

# Execute simple task
simple_task(robots[0])

# Task: Switch operations
def switch_task(robot):
    GoToObject(robot, 'Device')
    SwitchOn(robot, 'Device')
    time.sleep(2)
    SwitchOff(robot, 'Device')

# Execute switch task
switch_task(robots[0])

# Task: Break object
def break_task(robot):
    GoToObject(robot, 'Breakable')
    BreakObject(robot, 'Breakable')

# Execute break task
break_task(robots[0])


