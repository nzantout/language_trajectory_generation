from collections import deque
from typing import List, Any
import numpy as np
import shapely
from shapely.geometry import Polygon, Point
from matplotlib.patches import Rectangle

def get_rect_intersection_subgraphs(rectangles: List[List[Any]]) -> List[float]:  
    rectangle_adj_list = [[] for _ in range(len(rectangles))]
    for i in range(len(rectangles)):
        for j in range(i+1, len(rectangles)):
            rect1 = rectangles[i]
            rect2 = rectangles[j]
            x_min1, y_min1, x_max1, y_max1 = rect1[2:]
            x_min2, y_min2, x_max2, y_max2 = rect2[2:]

            # Check if rectangles intersect
            if not (x_min1 > x_max2 or x_min2 > x_max1 or y_min1 > y_max2 or y_min2 > y_max1):
                rectangle_adj_list[i].append(j)
                rectangle_adj_list[j].append(i)
    
    visited = [False] * len(rectangles)
    intersection_subgraphs = []
    for i in range(len(rectangles)):
        if not visited[i]:
            stack = deque([i])
            intersection_subgraph = []
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    intersection_subgraph.append(node)
                    for neighbor in rectangle_adj_list[node]:
                        if not visited[neighbor]:
                            stack.append(neighbor)
            intersection_subgraphs.append(intersection_subgraph)
    return intersection_subgraphs

def get_rect_union_polygons(rectangles: List[List[Any]]) -> List[Polygon]:
    """
    Get the union polygons of rectangles.
    
    Args:
        rectangles: List of rectangles, each defined as [id, name, x_min, y_min, x_max, y_max]
        
    Returns:
        List of union polygons
    """
    polygons = []
    for rect in rectangles:
        x_min, y_min, x_max, y_max = rect[2:]
        polygons.append(shapely.box(x_min, y_min, x_max, y_max))
    
    intersection_subgraphs = get_rect_intersection_subgraphs(rectangles)
    union_polygons = [shapely.union_all([polygons[i] for i in subgraph]) for subgraph in intersection_subgraphs]
    
    return union_polygons

def point_segment_distance(point: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray) -> float:
    """
    Calculate the distance from a point to a line segment.
    
    Args:
        point: Array of shape (2,) representing the point [x, y]
        segment_start: Array of shape (2,) representing the start of the segment [x1, y1]
        segment_end: Array of shape (2,) representing the end of the segment [x2, y2]
        
    Returns:
        Distance from the point to the line segment
    """
    segment_vector = segment_end - segment_start
    point_vector = point - segment_start
    segment_length_squared = np.dot(segment_vector, segment_vector)
    
    if segment_length_squared == 0:
        return np.linalg.norm(point_vector)
    
    t = np.clip(np.dot(point_vector, segment_vector) / segment_length_squared, 0, 1)
    projection = segment_start + t * segment_vector
    return np.linalg.norm(point - projection), projection

def enforce_collision_avoidance(point: np.ndarray, rectangles: List[List[Any]], buffer_distance: float = 0.3) -> np.ndarray:
    """
    Enforce collision avoidance for a point with respect to rectangular obstacles.
    
    Args:
        point: Array of shape (2,) representing the point [x, y]
        rectangles: List of rectangles, each defined as [id, name, x_min, y_min, x_max, y_max]
        buffer_distance: Minimum distance to maintain from obstacles
        
    Returns:
        point: Updated point after enforcing collision avoidance
    """
    collision_rects = []
    for rect in rectangles:
        x_min, y_min, x_max, y_max = rect[2:]
        collision_rects.append(
            [rect[0], rect[1], x_min - buffer_distance, y_min - buffer_distance, x_max + buffer_distance, y_max + buffer_distance])

    collision_polygons = get_rect_union_polygons(collision_rects)
    for polygon in collision_polygons:
        polygon_coords = np.array(polygon.exterior.coords.xy).T
        if polygon.contains(Point(point)):
            distances_and_projections = [point_segment_distance(point, polygon_coords[i], polygon_coords[i+1]) for i in range(len(polygon_coords)-1)]
            distances = np.array([d[0] for d in distances_and_projections])
            projections = np.array([d[1] for d in distances_and_projections])
            return projections[np.argmin(distances)]
    return point

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage
    rectangles = [
        [1, "rect1", 0, 0, 2, 2],  # Group 1
        [2, "rect2", 1, 1, 3, 3],  # Group 1
        [3, "rect3", 2, 0, 4, 2],  # Group 1
        [4, "rect4", 5, 5, 7, 7],  # Group 2
        [5, "rect5", 6, 6, 8, 8],  # Group 2
        [6, "rect6", 10, 10, 12, 12],  # No intersection
        [7, "rect7", 13, 13, 15, 15],  # No intersection
        [8, "rect8", 16, 16, 18, 18],  # No intersection
        [9, "rect9", 19, 19, 21, 21],  # No intersection
        [10, "rect10", 22, 22, 24, 24]  # No intersection
    ]
    
    point = np.array([1.47, 1.36])
    
    # Compute and plot union polygons
    union_polygons = get_rect_union_polygons(rectangles)

    point_collision_avoidance = enforce_collision_avoidance(point, rectangles)
    
    fig, ax = plt.subplots()
    for i, polygon in enumerate(union_polygons):
        x, y = polygon.exterior.xy
        ax.fill(x, y, alpha=0.5, label=f"Union Polygon {i + 1}")
    
    # Plot original rectangles for reference
    for i, rect in enumerate(rectangles):
        x_min, y_min, x_max, y_max = rect[2:]
        width = x_max - x_min
        height = y_max - y_min
        ax.add_patch(Rectangle((x_min, y_min), width, height, edgecolor='black', facecolor='none', lw=1, linestyle='--'))
    
    # Plot the original point
    ax.plot(point[0], point[1], 'ro', label="Original Point")
    
    # Plot the collision-avoided point
    ax.plot(point_collision_avoidance[0], point_collision_avoidance[1], 'go', label="Collision Avoided Point")
    
    ax.set_xlim(-1, 25)
    ax.set_ylim(-1, 25)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.title("Union Polygons of Rectangles with Points")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()