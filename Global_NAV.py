import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

growth_size = 1

############################################################################################################################################

def check_matrix(matrix):
    nb_start = np.size(np.argwhere(matrix == 2),0)
    nb_goal = np.size(np.argwhere(matrix == 3),0)
    if nb_start == 1 and nb_goal == 1:
        return True
    else:
        return False

# Matrix conversion to have dimensions, start and goal positions
def conversion(matrix):

    # Put matrix into numpy array (in case not already)
    arr = np.array(matrix)
    
    # Find the indices of start and goal
    start_arr = np.argwhere(arr == 2)
    goal_arr = np.argwhere(arr == 3)
    """ # Verify there is exactly one 2 and one 3 in the matrix
    if np.size(start_arr,0) > 1 or np.size(goal_arr,0) > 1:
        print("Error: There should be exactly one start and one goal on the field.")
        return None """
    
    # Get the height, width, and positions of start and goal
    max_val_x, max_val_y = arr.shape
    start = (start_arr[0][0], start_arr[0][1])
    goal = (goal_arr[0][0], goal_arr[0][1])

    # Replace the positions of 2 and 3 with 0
    arr[start] = 0
    arr[goal] = 0

    return max_val_x, max_val_y, start, goal, arr

############################################################################################################################################

# Creating the occupancy_grid for testing
def create_empty_plot(max_val_x, max_val_y):
    """
    Helper function to create a figure of the desired dimensions & grid
    
    :param max_val_x and max_val_y: dimension of the map along the x and y dimensions
    :return: the fig and ax objects.
    """
    fig, ax = plt.subplots(figsize=(7,7))
    
    major_ticks_x = np.arange(0, max_val_x+1, 5)
    minor_ticks_x = np.arange(0, max_val_x+1, 1)
    major_ticks_y = np.arange(0, max_val_y+1, 5)
    minor_ticks_y = np.arange(0, max_val_y+1, 1)
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.set_ylim([-1,max_val_y])
    ax.set_xlim([-1,max_val_x])
    ax.grid(True)
    
    return fig, ax

############################################################################################################################################

# Grow the obstacles by half the robots radius to avoid collision
def grow_obstacles(matrix, growth_size):
    # Matrix to a numpy array (in case not already)
    arr = np.array(matrix)

    # New matrix with the same shape and filled with zeros
    expanded_matrix = np.zeros_like(arr)

    # Find the indices of obstacles (value = 1) in the original matrix
    obstacle_indices = np.where(arr == 1)

    # Grow obstacles in the expanded matrix
    for i, j in zip(obstacle_indices[0], obstacle_indices[1]):
        # Range for the expanded obstacles
        row_range = slice(max(0, i - growth_size), min(arr.shape[0], i + growth_size + 1))
        col_range = slice(max(0, j - growth_size), min(arr.shape[1], j + growth_size + 1))

        # Setting the corresponding elements to 1 in expanded matrix
        expanded_matrix[row_range, col_range] = 1

    return expanded_matrix

############################################################################################################################################

def reconstruct_path(cameFrom, current):
    """
    Recurrently reconstructs the path from start node to the current node
    :param cameFrom: map (dictionary) containing for each node n the node immediately 
                     preceding it on the cheapest path from start to n 
                     currently known.
    :param current: current node (x, y)
    :return: list of nodes from start to current node
    """
    total_path = [current]
    while current in cameFrom.keys():
        # Add where the current node came from to the start of the list
        total_path.insert(0, cameFrom[current]) 
        current=cameFrom[current]
    return total_path

############################################################################################################################################

# Inputs: max_val_x, max_value_y, occupancy_grid, start, goal
def A_Star(start, goal, h, coords, occupancy_grid, max_val_x = 50, max_val_y = 50):
    """
    A* for 2D occupancy grid. Finds a path from start to goal.
    h is the heuristic function. h(n) estimates the cost to reach goal from node n.
    :param start: start node (x, y)
    :param goal_m: goal node (x, y)
    :param occupancy_grid: the grid map
    :return: a tuple that contains: (the resulting path in meters, the resulting path in data array indices) -> ???
    """
    
    # Check if the start and goal are within the boundaries of the map
    for point in [start, goal]:
        assert point >= (0, 0) and point[0] < max_val_x and point[1] < max_val_y, "start or end goal not contained in the map"
    
    # check if start and goal nodes correspond to free spaces
    if occupancy_grid[start[0], start[1]]:
        raise Exception('Start node is not traversable')

    if occupancy_grid[goal[0], goal[1]]:
        raise Exception('Goal node is not traversable')
    
    # get the possible movements
    s2 = math.sqrt(2)
    movements = [(1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0), 
                (1, 1, s2), (-1, 1, s2), (-1, -1, s2), (1, -1, s2)]
    
    # The set of visited nodes that need to be (re-)expanded, i.e. for which the neighbors need to be explored
    # Initially, only the start node is known.
    openSet = [start]
    
    # The set of visited nodes that no longer need to be expanded.
    closedSet = []

    # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start to n currently known.
    cameFrom = dict()

    # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
    gScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    gScore[start] = 0

    # For node n, fScore[n] := gScore[n] + h(n). map with default value of Infinity
    fScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    fScore[start] = h[start]

    # while there are still elements to investigate
    while openSet != []:
        
        #the node in openSet having the lowest fScore[] value
        fScore_openSet = {key:val for (key,val) in fScore.items() if key in openSet}
        current = min(fScore_openSet, key=fScore_openSet.get)
        del fScore_openSet
        
        #If the goal is reached, reconstruct and return the obtained path
        if current == goal:
            return reconstruct_path(cameFrom, current), closedSet

        openSet.remove(current)
        closedSet.append(current)
        
        #for each neighbor of current:
        for dx, dy, deltacost in movements:
            
            neighbor = (current[0]+dx, current[1]+dy)
            
            # if the node is not in the map, skip
            if (neighbor[0] >= occupancy_grid.shape[0]) or (neighbor[1] >= occupancy_grid.shape[1]) or (neighbor[0] < 0) or (neighbor[1] < 0):
                continue
            
            # if the node is occupied or has already been visited, skip
            if (occupancy_grid[neighbor[0], neighbor[1]]) or (neighbor in closedSet): 
                continue
                
            # d(current,neighbor) is the weight of the edge from current to neighbor
            # tentative_gScore is the distance from start to the neighbor through current
            tentative_gScore = gScore[current] + deltacost
            
            if neighbor not in openSet:
                openSet.append(neighbor)
                
            if tentative_gScore < gScore[neighbor]:
                # This path to neighbor is better than any previous one. Record it!
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + h[neighbor]

    # Open set is empty but goal was never reached
    print("No path found to goal")
    return [], closedSet

############################################################################################################################################

def heuristics(max_val_x, max_val_y, goal):
    # List of all coordinates in the grid
    x,y = np.mgrid[0:max_val_x:1, 0:max_val_y:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
    coords = list([(int(x[0]), int(x[1])) for x in pos])

    # Define the heuristic, here = distance to goal ignoring obstacles (Euclidean)
    h = np.linalg.norm(pos - goal, axis=-1)
    h = dict(zip(coords, h))
    return h, coords

############################################################################################################################################

# Overall call with input: matrix and output: path, without displaying
def global_path(matrix):
    max_val_x, max_val_y, start, end, original_grid = conversion(matrix)

    # Grow the obstacles in the matrix
    growth_size = 0 # size of robot radius (in grid dimension) (would be 5.5cm)
    occupancy_grid = grow_obstacles(original_grid, growth_size)

    # Calling A*
    h, coords = heuristics(max_val_x, max_val_y, end)
    path, visitedNodes = A_Star(start, end, h, coords, occupancy_grid)
    path = np.array(path).reshape(-1, 2).transpose()
    
    return path

############################################################################################################################################