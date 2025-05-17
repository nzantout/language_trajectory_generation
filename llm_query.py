from mistralai import Mistral
import numpy as np
import json
import re
import argparse
from prompts.trajectory_planning import get_trajectory_planning_prompt
from collision_avoidance_optimization import optimize_trajectory
import os

def parse_trajectory_response(response: str) -> list:
    """Parse the trajectory from Mistral's response."""
    response_lines = response.split("\n")
    trajectory_index = -1
    for i, line in enumerate(response_lines):
        if line.startswith("Trajectory:"):
            trajectory_index = i
            break
    if trajectory_index == -1:
        raise ValueError("Trajectory not found in the response")
    start_index = trajectory_index + 1
    for i in range(start_index, len(response_lines)):
        if response_lines[start_index].startswith('['):
            start_index = i
            break
    response_recombined = "\n".join(response_lines[start_index:])
    trajectory = eval(response_recombined)
    return trajectory

def plan_trajectory(
    user_input: str,
    objects: list,
    robot_coords: np.ndarray,
    environment_name: str = "a living room",
    api_key: str = os.getenv("MISTRAL_API_KEY", None)
) -> list:
    """
    Plan a trajectory using Mistral based on user input and environment information.
    
    Args:
        user_input (str): Natural language instruction for the robot
        objects (list): List of objects in format [id, name, x_min, y_min, x_max, y_max]
        robot_coords (np.ndarray): Starting coordinates of the robot
        environment_name (str): Name of the environment
        api_key (str): Mistral API key. If None, will look for MISTRAL_API_KEY env variable
        
    Returns:
        list: List of [x, y] coordinates representing the planned trajectory
    """
    if api_key is None:
        import os
        api_key = os.getenv("MISTRAL_API_KEY")
        if api_key is None:
            raise ValueError("Mistral API key not provided and MISTRAL_API_KEY environment variable not set")
    
    # Initialize Mistral client
    client = Mistral(api_key=api_key)
    
    # Get the prompt template
    prompt = get_trajectory_planning_prompt(environment_name, robot_coords, objects)
    
    # Add the user input to the prompt
    full_prompt = prompt + user_input
    
    # Query Mistral
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": full_prompt}]
    )

    print(full_prompt)

    response_content = response.choices[0].message.content
    print(response_content)
    
    # Parse the response to extract the trajectory
    trajectory = parse_trajectory_response(response_content)
    
    return trajectory

if __name__ == "__main__":
    # Example usage
    example_objects = [
        [0, 'sofa', 0.5, 0.5, 1.5, 2.5],
        [1, 'tv', 3.8, 1.25, 4.2, 1.75],
    ]
    robot_start = np.array([[0.0, 0.0, 0.0]])
    user_command = "Go around the left side of the couch on your path to the TV, then back to the couch."
    # user_command = input("Enter a command: ")
    
    trajectory = plan_trajectory(user_command, example_objects, robot_start)
    print("Planned trajectory:", trajectory)
    optimized_trajectory = optimize_trajectory(
        robot_start[0], 
        np.array(trajectory), 
        example_objects, 
        0.1, 
        30.0,
        enforce_full_sparse_trajectory_collision=True)
    print("Optimized trajectory:", optimized_trajectory)

