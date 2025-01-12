import numpy as np
import matplotlib.pyplot as plt
import random
from heapq import heappush, heappop


# Step 1: Create a Topographic surface like MARS
def generate_mars_surface(size=100):
    np.random.seed(42)
    surface = np.random.normal(0, 1, (size, size))
    surface = np.cumsum(surface, axis=0) + np.cumsum(surface, axis=1)
    surface = (surface - surface.min()) / (surface.max() - surface.min()) * 255
    return surface

# Step 2: Pathfinding - Stochastic A* Algorithm
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def stochastic_a_star(surface, start, goal, randomness=0.2):
    rows, cols = surface.shape
    open_set = []
    heappush(open_set, (0,start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        neighbors = [
            (current[0] + 1, current[1]),
            (current[0] - 1, current[1]),
            (current[0], current[1] + 1),
            (current[0], current[1] - 1)
        ]

        random.shuffle(neighbors) # order the neighbors randomly

        for neighbor in neighbors:
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                height_diff = abs(surface[neighbor[0], neighbor[1]] - surface[current[0], current[1]])
                stochastic_cost = random.uniform(0, randomness)  # Rastgele maliyet ekle
                tentative_g_score = g_score[current] + 1 + height_diff + stochastic_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
                    came_from[neighbor] = current

    return []

# Step 3: Measure the path cost for improvements
def calculate_path_cost(surface, path):
    if not path:
        return float('inf')
    cost = 0
    for i in range(len(path) - 1):
        current = path[i]
        next_point = path[i + 1]
        height_diff = abs(surface[next_point[0], next_point[1]] - surface[current[0], current[1]])
        cost += 1 + height_diff
    return cost

# Step 4: Find the Path and Visualization
def plot_path(surface, path, start, goal, iteration, cost):
    plt.figure(figsize=(12, 12))
    cmap = plt.cm.get_cmap("YlOrBr")
    plt.imshow(surface, cmap=cmap, origin='lower')

    plt.scatter(start[1], start[0], c='blue', label='Start', s=150, edgecolors='black')
    plt.scatter(goal[1], goal[0], c='green', label='Goal', s=150, edgecolors='black')

    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], c='black', linewidth=2, label=f'Finding Path (Iteration {iteration}, Cost: {cost:.2f})')

    plt.colorbar(label='Height', fraction=0.046, pad=0.04)
    plt.title('MARS Rover Simulation', fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
    plt.grid(False)
    plt.show()


# Step 5: Simulation and Improvement Cycle
mars_surface = generate_mars_surface()
start, goal = (10, 10), (90, 90)
best_cost = float('inf')
best_path = None

for iteration in range(1, 11):  # 10 iteration
    path = stochastic_a_star(mars_surface, start, goal, randomness=0.2)
    cost = calculate_path_cost(mars_surface, path)

    if cost < best_cost:
        best_cost = cost
        best_path = path

    plot_path(mars_surface, path, start, goal, iteration, cost)
    print(f"Iteration {iteration}: Path Cost = {cost:.2f}")

print(f"Best Path Cost: {best_cost:.2f}")