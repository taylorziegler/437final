import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, coor, parent=None):
        self.coor = coor  # The coordinates of the node (x, y)
        self.parent = parent  # The parent node
        self.g = 0  # steps takes from the origin to this node
        self.h = 0  # Heuristic distance
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.coor == other.coor

def heuristic(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

def A_star(map, origin, target):
    """
    :param map: A numpy array (2D) with 0 as traversible space and 1 as obstacles.
    :param origin: The coordinates of the origin.
    :param target: The coordinates of the target.
    """
    # Create origin and target nodes
    origin_node = Node(origin)
    target_node = Node(target)

    # Open list (initially with the origin node)
    queue = [origin_node]
    # visited nodes
    visited_set = set()

    # Define possible actions (up, down, left, right, upperleft, upperright, lowerleft, lowerright)
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]# , (-1, -1), (-1, 1), (1, -1), (1, 1)

    while queue:
        # Sort the open list by f value, and pop the node with the lowest f value
        queue.sort(key=lambda node: node.f)
        node = queue.pop(0)
        visited_set.add(node.coor)

        # if we have reached the target, construct and return the path
        if node == target_node:
            path = []
            while node:
                path.append(node.coor)
                node = node.parent
            return path[::-1]  # Return reversed path

        # Generate next states (possible actions)
        for a in actions:
            next_coor = (node.coor[0] + a[0], node.coor[1] + a[1])
            # Check if the next state is within map
            if (0 <= next_coor[0] < map.shape[0]) and (0 <= next_coor[1] < map.shape[1]):
                # Check if it's an obstacle
                if map[next_coor[0], next_coor[1]] == 1:
                    continue

                # If the next state is already visited, skip it
                if next_coor in visited_set:
                    continue

                # Create a new node for the next state and set its parent and costs   
                next_node = Node(next_coor, node)

                # Calculate the costs
                next_node.g = node.g + 1
                next_node.h = heuristic(next_node.coor, target_node.coor)
                next_node.f = next_node.g + next_node.h

                # If the next_node is not in the queue or has a smaller g value, add it
                add = True
                for unvisited_node in queue:
                    if unvisited_node.coor == next_node.coor:
                        add = False
                        unvisited_node.g = min(unvisited_node.g, next_node.g)
                        break
                if add:
                    queue.append(next_node)

    return None 

def visualize_path(map, path):
    map_copy = np.copy(map)
    
    # Plot the map
    plt.imshow(map_copy, cmap='gray')
    x_coor = [coor[1] for coor in path]
    y_coor = [coor[0] for coor in path]

    # Plot the path in red
    plt.plot(x_coor, y_coor, color='red', linewidth=2)

    # Add labels and title
    plt.title("Maps (with path in red, and obstacles in white)")
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Example map as a numpy array: 0 is free space, 1 is an obstacle
    # map = np.array([
    #     [0, 1, 0, 0, 0],
    #     [0, 1, 0, 1, 0],
    #     [0, 0, 0, 1, 0],
    #     [0, 1, 1, 1, 0],
    #     [0, 0, 0, 0, 0]
    # ])
    map = np.load("irregular_matrix_with_obstacles.npy")

    origin = (99, 49)  # origin coor
    target = (99, 49)    # Goal coor
    print(map[target])

    path = A_star(map, origin, target)

    if path:
        print(f"Path found: {path}")
        visualize_path(map, path)
    else:
        print("No path found.")
