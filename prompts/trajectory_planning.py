example_object_list = [
    [0, 'sofa', 0.5, 0.5, 1.5, 2.5],
    [1, 'tv', 3.8, 1.25, 4.2, 1.75],
]

def get_trajectory_planning_prompt(environment_name, robot_coords, objects):
    prompt = f'''You are an intelligent agent, tasked with generating Python code to move a wheeled robot around a 2D environment from natural language instructions.

The environment is {environment_name}. The robot starts at {tuple(robot_coords[0].tolist())}.

You are given a list of objects in the environment. Each object is characterized by an identifying number id, a name, and (x, y) coordinates of its bounding box: [id, 'name', x_min, y_min, x_max, y_max].
    
After the list of objects in the environment, you are given a natural language command asking the robot to navigate around the environment. You will be tasked with generating a trajectory for the robot to follow as a list of (x, y) coordinates. Use chain-of-thought reasoning, and explain your reasoning as you parse the language into a trajectory. Write out your reasoning, then write out the trajectory. Remember: Left and right are relative to your initial position and orientation. Here are two examples of the input and output formats:

Example 1:

Object List:
{example_object_list}

User Input:
Go forward, then hug the corner of the couch on your path to the TV.

Your Output:

Reasoning:
[Explain your reasoning here]

Trajectory:
[
    [0.0, 0.0],
    [0.5, 0.5],
    [0.5, 1.0],
    [0.5, 1.5],
    [0.5, 2.0],
    [0.5, 2.5],
    [1, 2.5],
    [1.5, 2.5],
    [2.0, 2.0],
    [2.5, 1.5]
    [3.0, 1.5],
    [3.5, 1.5],
]

Example 2:

Object List:
{example_object_list}

User Input:
Go forward and go around the couch to the TV, but don’t stick too close.

Your Output:

Reasoning:
[Explain your reasoning here]

Trajectory:
[
    [0.0, 0.0],
    [-0.5, 0.5],
    [-0.5, 1.0],
    [-0.5, 1.5],
    [-0.5, 2.0],
    [-0.5, 2.5],
    [-0.5, 3.0],
    [-0.5, 3.5],
    [0.0, 3.5],
    [0.5, 3.5],
    [1.0, 3.5],
    [1.5, 3.5],
    [2.0, 3.5],
    [2.5, 3.0],
    [3.0, 2.0],
    [3.5, 1.5],
]

Respond ONLY in the following format. Make sure you think step by step, and write out your reasoning, then the trajectory as shown below.

Your Output:

Reasoning: 
[Thoroughly explain the reasoning behind your answer.]

Trajectory:
[your trajectory here]

Given the following list of objects and the input language command, output a trajectory for the robot to follow:

Object List:
{objects}

User Input:
'''
    return prompt

if __name__ == "__main__":
    import numpy as np

    environment_name = "a living room"
    grid_map_shape = (5, 5)
    robot_coords = np.array([[0, 0]])
    objects = example_object_list

    prompt = get_trajectory_planning_prompt(environment_name, grid_map_shape, robot_coords, objects)
    print(prompt)
