from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import csv
import math

# === Color definitions ===
BLUE   = (0, 0, 255)     # ArUco ID 10 (robot position)
YELLOW = (255, 255, 0)   # ArUco ID 15 (goal)
BLACK  = (0, 0, 0)
PINK   = (203, 192, 255)
GRAY = (127,127,127)

# === Load and prepare the image ===
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)

# === Utility functions ===
def find_color_pixel(image, target_color):
    matches = np.where(np.all(image == target_color, axis=-1))
    if len(matches[0]) > 0:
        y, x = matches[0][0], matches[1][0]
        return (x, y)
    return None
def extract_path_from_image(image, path_color=(203, 192, 255)):
    tolerance = 10  # Puedes probar con 10 si sigue sin detectar
    mask = np.all(np.abs(image - path_color) <= tolerance, axis=-1)
    coords = np.argwhere(mask)

    if coords.shape[0] == 0:
        print("⚠️ No path pixels found in the image.")
        return []

    # coords is a list of (y, x), we convert to (x, y)
    points = [(int(x), int(y)) for y, x in coords]

    # Sort with greedy nearest neighbor
    path = [points[0]]
    used = set([0])

    for _ in range(1, len(points)):
        last_point = path[-1]
        nearest_index = None
        min_dist = float('inf')

        for i, pt in enumerate(points):
            if i in used:
                continue
            dist = math.hypot(pt[0] - last_point[0], pt[1] - last_point[1])
            if dist < min_dist:
                min_dist = dist
                nearest_index = i

        if nearest_index is not None:
            path.append(points[nearest_index])
            used.add(nearest_index)
        else:
            break

    return path


def is_cell_black(image, x, y, cell_size, ignore_pos=None):
    cell = image[y:y+cell_size, x:x+cell_size]
    unique_colors = np.unique(cell.reshape(-1, 3), axis=0)
    
    near_ignore = False
    if ignore_pos:
        ignore_x, ignore_y = ignore_pos
        if abs(x - ignore_x) <= cell_size and abs(y - ignore_y) <= cell_size:
            near_ignore = True

    for color in unique_colors:
        if near_ignore:
            if not (np.array_equal(color, BLACK) or 
                    np.array_equal(color, BLUE) or 
                    np.array_equal(color, YELLOW) or
                    np.array_equal(color, GRAY)):
                return False
        else:
            if not (np.array_equal(color, BLACK) or 
                    np.array_equal(color, BLUE) or 
                    np.array_equal(color, YELLOW)):
                return False
    return True

    

def debug_print_colors(image):
    unique_colors = np.unique(image.reshape(-1, 3), axis=0)
    print(f"Found {len(unique_colors)} unique colors in the image.")
    for color in unique_colors:
        print(color)

def get_cell_center(x, y, cell_size):
    return (x + cell_size // 2, y + cell_size // 2)

def a_star(image, start_px, goal_px, cell_size):
    height, width, _ = image.shape
    grid_w = width // cell_size
    grid_h = height // cell_size

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def neighbors(cell):
        x, y = cell
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_w and 0 <= ny < grid_h:
                world_x, world_y = nx * cell_size, ny * cell_size
                if is_cell_black(image, world_x, world_y, cell_size, ignore_pos=start_px):
                    yield (nx, ny)

    start = (start_px[0] // cell_size, start_px[1] // cell_size)
    goal  = (goal_px[0] // cell_size, goal_px[1] // cell_size)

    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not frontier.empty():
        _, current = frontier.get()
        if current == goal:
            break
        for next_cell in neighbors(current):
            new_cost = cost_so_far[current] + 1
            if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                cost_so_far[next_cell] = new_cost
                priority = new_cost + heuristic(goal, next_cell)
                frontier.put((priority, next_cell))
                came_from[next_cell] = current

    # Reconstruct path
    current = goal
    path = []
    while current and current != start:
        cx, cy = current
        path.append(get_cell_center(cx * cell_size, cy * cell_size, cell_size))
        current = came_from.get(current)
    path.reverse()
    return path

# --- New function to save CSV commands ---
def save_commands_to_csv(path, resolution=10, turn_threshold_deg=10):
    commands = []
    prev_angle = None

    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        angle = math.degrees(math.atan2(dy, dx))
        angle = (angle + 180) % 360 - 180  # Normalize to [-180, 180]

        if prev_angle is not None:
            turn_angle = angle - prev_angle
            turn_angle = (turn_angle + 180) % 360 - 180
            if abs(turn_angle) > turn_threshold_deg:
                if turn_angle > 0:
                    commands.append(('TURN', turn_angle + 35))
                else:
                    commands.append(('TURN', turn_angle - 35))

        commands.append(('MOVE', resolution))
        prev_angle = angle

    return commands  # <== RETURN instead of writing here


def optimize_move_commands(commands):
    """
    Combines consecutive MOVE commands into one if there is no TURN between them.
    Returns a new optimized list.
    """
    optimized = []
    move_accum = 0.0

    for action, value in commands:
        if action == 'MOVE':
            move_accum += value
        else:
            # Before any TURN, push accumulated MOVE if any
            if move_accum > 0:
                optimized.append(('MOVE', move_accum-2))
                move_accum = 0.0
            optimized.append((action, value))

    # Handle trailing MOVE
    if move_accum > 0:
        optimized.append(('MOVE', move_accum))

    return optimized

# === Main ===
def main():
    image_path = "/home/pi/ros2_ws/occupancy_map_with_markers.png"
    output_path = "/home/pi/ros2_ws/occupancy_map_with_grid_and_path.png"
    #Made cell size bigger from previous 10
    cell_size = 10


    image = load_image(image_path)
    height, width = image.shape[:2]

    debug_print_colors(image)

    start_px = find_color_pixel(image, BLUE)
    goal_px  = find_color_pixel(image, YELLOW)

    if not start_px or not goal_px:
        print("❌ Start (blue) or goal (yellow) not found.")
        return

    path = a_star(image, start_px, goal_px, cell_size)
    #path = extract_path_from_image(image_path, PINK)


    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.xticks(np.arange(0, width, cell_size))
    plt.yticks(np.arange(0, height, cell_size))
    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.title("Occupancy Map with Grid Overlay and A* Path")

    if path:
        
        px, py = zip(*path)

        # Save raw path points to TXT
        with open("/home/pi/ros2_ws/a_star_path.txt", "w") as f:
            for point in path:
                f.write(f"{point[0]},{point[1]}\n")
        print("✅ Path points saved to: a_star_path.txt")

        # Generate and optimize movement commands from path
        raw_commands = save_commands_to_csv(path)  # Modify this function to return commands
        #optimized_commands = optimize_move_commands(raw_commands)

        optimized_commands = raw_commands
        # Save optimized commands to CSV
        csv_path = "/home/pi/ros2_ws/a_star_commands.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['action', 'value'])
            for cmd in optimized_commands:
                writer.writerow(cmd)
        print(f"✅ Commands optimized and saved to CSV: {csv_path}")

        # Save simplified waypoints to CSV
        waypoints_csv = "/home/pi/ros2_ws/critical_waypoints.csv"
        with open(waypoints_csv, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y'])
            for point in path:
                writer.writerow(point)
        print(f"✅ Critical waypoints saved to CSV: {waypoints_csv}")

        # Plot path
        plt.plot(px, py, color=np.array(PINK)/255.0, linewidth=2)
        print(f"✅ Drew A* path with {len(path)} steps.")
    else:
        print("⚠️ No path found.")

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Output saved to: {output_path}")

if __name__ == "__main__":
    main()


"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

# === Color definitions ===
BLUE   = (0, 0, 255)     # ArUco ID 10 (robot position)
YELLOW = (255, 255, 0)   # ArUco ID 15 (goal)
BLACK  = (0, 0, 0)
PINK   = (203, 192, 255)

# === Load and prepare the image ===
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)

# === Utility functions ===
def find_color_pixel(image, target_color):
    matches = np.where(np.all(image == target_color, axis=-1))
    if len(matches[0]) > 0:
        y, x = matches[0][0], matches[1][0]
        return (x, y)
    return None

def is_cell_black(image, x, y, cell_size):
    cell = image[y:y+cell_size, x:x+cell_size]
    unique_colors = np.unique(cell.reshape(-1, 3), axis=0)
    for color in unique_colors:
        if not (np.array_equal(color, BLACK) or 
                np.array_equal(color, BLUE) or 
                np.array_equal(color, YELLOW)):
            return False
    return True

def debug_print_colors(image):
    unique_colors = np.unique(image.reshape(-1, 3), axis=0)
    print(f"Found {len(unique_colors)} unique colors in the image.")
    for color in unique_colors:
        print(color)

def get_cell_center(x, y, cell_size):
    return (x + cell_size // 2, y + cell_size // 2)

def a_star(image, start_px, goal_px, cell_size):
    height, width, _ = image.shape
    grid_w = width // cell_size
    grid_h = height // cell_size

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def neighbors(cell):
        x, y = cell
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_w and 0 <= ny < grid_h:
                if is_cell_black(image, nx * cell_size, ny * cell_size, cell_size):
                    yield (nx, ny)

    start = (start_px[0] // cell_size, start_px[1] // cell_size)
    goal  = (goal_px[0] // cell_size, goal_px[1] // cell_size)

    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not frontier.empty():
        _, current = frontier.get()
        if current == goal:
            break
        for next_cell in neighbors(current):
            new_cost = cost_so_far[current] + 1
            if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                cost_so_far[next_cell] = new_cost
                priority = new_cost + heuristic(goal, next_cell)
                frontier.put((priority, next_cell))
                came_from[next_cell] = current

    current = goal
    path = []
    while current and current != start:
        cx, cy = current
        path.append(get_cell_center(cx * cell_size, cy * cell_size, cell_size))
        current = came_from.get(current)
    path.reverse()
    return path

# === Simplify path by keeping only orientation changes ===
def simplify_path(points):
    if len(points) < 2:
        return points

    simplified = [points[0]]
    prev_dx, prev_dy = None, None

    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        dx = x1 - x0
        dy = y1 - y0
        norm = (0 if dx == 0 else dx // abs(dx), 0 if dy == 0 else dy // abs(dy))

        if norm != (prev_dx, prev_dy):
            simplified.append(points[i - 1])
            prev_dx, prev_dy = norm

    simplified.append(points[-1])
    return simplified

# === Main ===
def main():
    image_path = "/home/pi/ros2_ws/occupancy_map_with_markers.png"
    output_path = "/home/pi/ros2_ws/occupancy_map_with_grid_and_path.png"
    cell_size = 10

    image = load_image(image_path)
    height, width = image.shape[:2]

    debug_print_colors(image)
    start_px = find_color_pixel(image, BLUE)
    goal_px  = find_color_pixel(image, YELLOW)

    if not start_px or not goal_px:
        print("❌ Start (blue) or goal (yellow) not found.")
        return

    path = a_star(image, start_px, goal_px, cell_size)

    if not path:
        print("⚠️ No path found.")
        return

    simplified_path = simplify_path(path)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.xticks(np.arange(0, width, cell_size))
    plt.yticks(np.arange(0, height, cell_size))
    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.title("Occupancy Map with Grid Overlay and Simplified Path")

    # Draw simplified path in pink
    px, py = zip(*simplified_path)
    plt.plot(px, py, color=np.array(PINK)/255.0, linewidth=2)
    print(f"✅ Simplified path has {len(simplified_path)} key points.")

    # Save simplified path to file
    with open("/home/pi/ros2_ws/a_star_path.txt", "w") as f:
        for point in simplified_path:
            f.write(f"{point[0]},{point[1]}\n")
    print("✅ Simplified path saved to: a_star_path.txt")

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Output image saved to: {output_path}")

if __name__ == "__main__":
    main()
"""