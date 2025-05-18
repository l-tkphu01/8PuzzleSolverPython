import time
import tracemalloc
import pygame
import sys
from collections import deque
import heapq
import random
import itertools
import math
import matplotlib.pyplot as plt
import os
import imageio
import numpy as np
import seaborn

pygame.init()
WIDTH = 650
HEIGHT = 950
TILE_SIZE = WIDTH // 3
FPS = 60
FONT = pygame.font.SysFont("Times New Roman", 40, bold=True)
SMALL_FONT = pygame.font.SysFont("Times New Roman", 20, bold=True)
# Màu nền và giao diện mới
BACKGROUND = (245, 245, 250)  # Màu nền trắng xám nhạt
TILE_COLOR = (100, 180, 220)  # Màu ô vuông chính (xanh pastel)
TILE_HOVER = (120, 200, 240)  # Màu khi hover (xanh sáng hơn)
BORDER_COLOR = (70, 130, 180) # Màu viền (xanh đậm)
BUTTON_COLOR = (150, 210, 240) # Màu nút (xanh nhạt)
BUTTON_ACTIVE = (80, 160, 220) # Màu nút active (xanh đậm hơn)
BUTTON_HOVER = (170, 220, 250) # Màu nút hover (xanh rất nhạt)
RESET_COLOR = (255, 140, 100)  # Màu nút reset (cam san hô)
TEXT_COLOR = (30, 60, 90)      # Màu chữ (xanh đen nhạt)
SHADOW_COLOR = (200, 200, 210, 100) # Màu bóng (xám nhẹ)
WARNING_BG = (255, 245, 225)   # Màu nền cảnh báo (vàng nhạt)
BLACK = (40, 40, 40)           # Màu đen nhẹ
PAUSE_COLOR = (180, 120, 240)  # Màu nút pause (tím pastel)
HIGHLIGHT_COLOR = (255, 225, 120) # Màu highlight (vàng chanh)
ACCENT_COLOR = (255, 105, 120) # Màu nhấn (đỏ cam)
SECONDARY_COLOR = (120, 200, 180) # Màu phụ (xanh lá ngọc)


is_running = False
paused = False
algorithm_state = {"current_step": 0, "solution": None}
button_scroll_y = 0
button_max_scroll = 0
button_dragging = False

# Danh sách lưu hiệu suất thuật toán
algorithm_performance = []

def safe_filename(name):
    """Loại bỏ hoặc thay thế các ký tự không hợp lệ trong tên file"""
    name = name.replace("*", "_star")
    invalid_chars = '<>:"/\\|?'
    for char in invalid_chars:
        name = name.replace(char, "_")
    return name

def create_gif(filenames, output_filename="puzzle_solution.gif", algorithm_name="", duration=0.5):
    """Tạo GIF từ danh sách các file PNG đã chụp"""
    if not filenames:
        print("No images to create GIF")
        return
    try:
        with imageio.get_writer(output_filename, mode='I', duration=duration, loop=0) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        print(f"GIF created: {os.path.abspath(output_filename)}")
    except OSError as e:
        print(f"Error creating GIF: {e}")

def plot_performance_comparison():
    """Tạo và lưu biểu đồ cột thẳng đứng so sánh hiệu suất các thuật toán"""
    if not algorithm_performance:
        print("No performance data to plot")
        return
    
    algorithms = [entry["algorithm"] for entry in algorithm_performance]
    steps = [entry["steps"] for entry in algorithm_performance]
    times = [entry["time"] for entry in algorithm_performance]
    memory = [entry["memory"] / 1024 for entry in algorithm_performance]  # Chuyển sang KB
    
    # Thiết lập biểu đồ
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        plt.style.use('ggplot')  # Sử dụng style tích hợp sẵn
    except ValueError:
        print("ggplot style not found, using 'default' instead")
        plt.style.use('default')  # Style dự phòng
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    x = np.arange(len(algorithms))
    width = 0.25
    
    # Biểu đồ số bước
    ax1.bar(x, steps, width, label='Steps', color='#1f77b4')
    ax1.set_ylabel('Steps', fontsize=12)
    ax1.set_title('Performance Comparison of Algorithms', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Biểu đồ thời gian
    ax2.bar(x, times, width, label='Time (seconds)', color='#ff7f0e')
    ax2.set_ylabel('Time (s)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Biểu đồ bộ nhớ
    ax3.bar(x, memory, width, label='Memory (KB)', color='#2ca02c')
    ax3.set_ylabel('Memory (KB)', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(pad=2.0)
    
    # Lưu biểu đồ
    import os
    charts_dir = "charts"
    os.makedirs(charts_dir, exist_ok=True)
    # Lưu file với tên thuật toán vừa chạy
    latest_algorithm = algorithm_performance[-1]["algorithm"]
    safe_name = safe_filename(latest_algorithm.lower())
    output_file = os.path.join(charts_dir, f"{safe_name}_performance.png")
    try:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Performance comparison chart for {latest_algorithm} saved: {os.path.abspath(output_file)}")
    except OSError as e:
        print(f"Error saving performance chart {output_file}: {e}")
    plt.close()
    
def run_algorithm_with_stats(algorithm, start, goal, algorithm_name):
    """Chạy thuật toán và đo lường"""
    tracemalloc.start()
    start_time = time.time()
    
    solution = algorithm(start, goal)
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    time_taken = end_time - start_time
    memory_used = peak
    steps = len(solution) - 1 if solution else 0
    
    # Lưu dữ liệu hiệu suất
    if solution:
        algorithm_performance.append({
            "algorithm": algorithm_name,
            "steps": steps,
            "time": time_taken,
            "memory": memory_used
        })
        plot_performance_comparison()
    
    return solution, time_taken, memory_used

def show_algorithm_stats(screen, clock, steps, time_taken, memory_usage):
    """Hiển thị thông tin cực nhanh - chỉ 1 lần vẽ duy nhất"""
    stats_rect = pygame.Rect(WIDTH//4, HEIGHT//4, WIDTH//2, HEIGHT//3)
    pygame.draw.rect(screen, WARNING_BG, stats_rect, border_radius=10)
    pygame.draw.rect(screen, BORDER_COLOR, stats_rect, 3, border_radius=10)
    title = SMALL_FONT.render("Algorithm Statistics", True, TEXT_COLOR)
    screen.blit(title, (stats_rect.centerx - title.get_width()//2, stats_rect.y + 20))
    y_offset = stats_rect.y + 60
    stats = [
        f"Steps: {steps}",
        f"Time: {time_taken:.4f} sec",
        f"Memory: {memory_usage/1024:.2f} KB"
    ]
    for stat in stats:
        text = SMALL_FONT.render(stat, True, TEXT_COLOR)
        screen.blit(text, (stats_rect.centerx - text.get_width()//2, y_offset))
        y_offset += 30
    ok_rect = pygame.Rect(stats_rect.centerx - 50, stats_rect.bottom - 60, 100, 40)
    pygame.draw.rect(screen, RESET_COLOR, ok_rect, border_radius=5)
    pygame.draw.rect(screen, BORDER_COLOR, ok_rect, 2, border_radius=5)
    ok_text = SMALL_FONT.render("OK", True, TEXT_COLOR)
    screen.blit(ok_text, (ok_rect.centerx - ok_text.get_width()//2, ok_rect.centery - ok_text.get_height()//2))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if ok_rect.collidepoint(event.pos):
                    waiting = False
        clock.tick(30)

# Helper Functions
def find_empty(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return (i, j)
    return None

def get_neighbors(state):
    neighbors = []
    empty_i, empty_j = find_empty(state)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_i, new_j = empty_i + di, empty_j + dj
        if 0 <= new_i < 3 and 0 <= new_j < 3:
            new_state = [list(row) for row in state]
            new_state[empty_i][empty_j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[empty_i][empty_j]
            neighbors.append(tuple(map(tuple, new_state)))
    return neighbors

def get_nondeterministic_neighbors(state):
    possible_states = set()
    empty_i, empty_j = find_empty(state)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_i, new_j = empty_i + di, empty_j + dj
        if 0 <= new_i < 3 and 0 <= new_j < 3:
            new_state = [list(row) for row in state]
            new_state[empty_i][empty_j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[empty_i][empty_j]
            possible_states.add(tuple(map(tuple, new_state)))
    return possible_states

def get_belief_state(state, observation):
    empty_pos, tile_00 = observation
    belief_states = set()
    base_state = [list(row) for row in state]
    empty_i, empty_j = empty_pos
    if base_state[empty_i][empty_j] != 0:
        return frozenset()  # Invalid observation
    base_state[0][0] = tile_00
    belief_states.add(tuple(map(tuple, base_state)))
    return frozenset(belief_states)

def is_solvable(start, goal):
    start_flat = [tile for row in start for tile in row if tile != 0]
    goal_flat = [tile for row in goal for tile in row if tile != 0]
    inversions = 0
    for i in range(len(start_flat)):
        for j in range(i + 1, len(start_flat)):
            if start_flat[i] > start_flat[j]:
                inversions += 1
    return inversions % 2 == 0

def manhattan_distance(state, goal):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                value = state[i][j]
                for gi in range(3):
                    for gj in range(3):
                        if goal[gi][gj] == value:
                            distance += abs(i - gi) + abs(j - gj)
    return distance

def reconstruct_path(parent, current):
    path = []
    while current is not None:
        path.append(current)
        current = parent.get(current)
    return path[::-1]

# Search Algorithms
def bfs(start, goal):
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    while queue and is_running:
        current = queue.popleft()
        if current == goal:
            return reconstruct_path(parent, current)
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                parent[neighbor] = current
    return None

def dfs(start, goal):
    if not is_solvable(start, goal):
        return None
    
    max_depth = 50  
    stack = [(start, 0, 0)]  
    visited = set()
    parent = {start: None}
    g_scores = {start: 0} 
    while stack and is_running:
        current, depth, g_score = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return reconstruct_path(parent, current)
        if depth >= max_depth:
            continue
        neighbors = get_neighbors(current)
        new_neighbors = []
        for neighbor in neighbors:
            if neighbor not in visited:
                new_g_score = g_score + 1  
                if neighbor not in g_scores or new_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = new_g_score
                    new_neighbors.append((neighbor, depth + 1, new_g_score))
                    parent[neighbor] = current
        new_neighbors.sort(key=lambda x: x[2], reverse=True)
        stack.extend(new_neighbors)
    return None

def ucs(start, goal):
    queue = [(0, start)]
    visited = {start}
    parent = {start: None}
    while queue and is_running:
        cost, current = heapq.heappop(queue)
        if current == goal:
            return reconstruct_path(parent, current)
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                heapq.heappush(queue, (cost + 1, neighbor))
                parent[neighbor] = current
    return None

def ids(start, goal):
    def dls(current, depth, visited, parent):
        if depth < 0 or not is_running:
            return None
        if current == goal:
            return reconstruct_path(parent, current)
        visited.add(current)
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                parent[neighbor] = current
                result = dls(neighbor, depth - 1, visited, parent)
                if result:
                    return result
        return None
    
    depth = 0
    while is_running:
        visited = {start}
        parent = {start: None}
        result = dls(start, depth, visited, parent)
        if result:
            return result
        depth += 1
        if depth > 50:  # Prevent infinite loop
            return None
    return None

def gss(start, goal):
    queue = [(manhattan_distance(start, goal), start)]
    visited = {start}
    parent = {start: None}
    while queue and is_running:
        _, current = heapq.heappop(queue)
        if current == goal:
            return reconstruct_path(parent, current)
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                heapq.heappush(queue, (manhattan_distance(neighbor, goal), neighbor))
                parent[neighbor] = current
    return None

def a_star(start, goal):
    queue = [(0 + manhattan_distance(start, goal), 0, start)]
    visited = {start}
    parent = {start: None}
    while queue and is_running:
        _, g_score, current = heapq.heappop(queue)
        if current == goal:
            return reconstruct_path(parent, current)
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                new_g_score = g_score + 1
                f_score = new_g_score + manhattan_distance(neighbor, goal)
                heapq.heappush(queue, (f_score, new_g_score, neighbor))
                parent[neighbor] = current
    return None

def ida_star(start, goal):
    def search(current, g, threshold, visited, parent):
        if not is_running:
            return None, float('inf')
        f = g + manhattan_distance(current, goal)
        if f > threshold:
            return None, f
        if current == goal:
            return reconstruct_path(parent, current), f
        visited.add(current)
        min_cost = float('inf')
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                parent[neighbor] = current
                result, cost = search(neighbor, g + 1, threshold, visited, parent)
                if result:
                    return result, cost
                min_cost = min(min_cost, cost)
        return None, min_cost

    threshold = manhattan_distance(start, goal)
    iteration = 0
    while is_running and iteration < 1000:
        visited = {start}
        parent = {start: None}
        result, new_threshold = search(start, 0, threshold, visited, parent)
        if result:
            return result
        if new_threshold == float('inf'):
            return None
        threshold = new_threshold
        iteration += 1
    return None

def simple_hill_climbing(start, goal):
    current = start
    parent = {start: None}
    visited = set()
    while is_running:
        if current == goal:
            return reconstruct_path(parent, current)
        visited.add(current)
        neighbors = get_neighbors(current)
        next_state = None
        min_h = float('inf')
        for neighbor in neighbors:
            if neighbor not in visited:
                h = manhattan_distance(neighbor, goal)
                if h < min_h:
                    min_h = h
                    next_state = neighbor
        if next_state is None or min_h >= manhattan_distance(current, goal):
            return None
        parent[next_state] = current
        current = next_state
    return None

def steepest_ascent_hill_climbing(start, goal):
    current = start
    parent = {start: None}
    visited = set()
    while is_running:
        if current == goal:
            return reconstruct_path(parent, current)
        visited.add(current)
        neighbors = get_neighbors(current)
        best_neighbor = None
        best_h = manhattan_distance(current, goal)
        for neighbor in neighbors:
            if neighbor not in visited:
                h = manhattan_distance(neighbor, goal)
                if h < best_h:
                    best_h = h
                    best_neighbor = neighbor
        if best_neighbor is None:
            return None
        parent[best_neighbor] = current
        current = best_neighbor
    return None

def stochastic_hill_climbing(start, goal):
    current = start
    parent = {start: None}
    visited = set()
    iteration = 0
    while is_running and iteration < 1000:
        if current == goal:
            return reconstruct_path(parent, current)
        visited.add(current)
        neighbors = get_neighbors(current)
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        if not unvisited_neighbors:
            return None
        next_state = random.choice(unvisited_neighbors)
        if manhattan_distance(next_state, goal) >= manhattan_distance(current, goal):
            iteration += 1
            continue
        parent[next_state] = current
        current = next_state
        iteration = 0
    return None

def nondeterministic_search(start, goal):
    if not is_solvable(start, goal):
        return None
    
    initial_belief = frozenset([start])
    queue = deque([(initial_belief, [start])])
    visited = {initial_belief}
    max_iterations = 1000
    iteration = 0

    while queue and is_running and iteration < max_iterations:
        iteration += 1
        current_belief, path = queue.popleft()
        
        if all(state == goal for state in current_belief):
            return path
        
        empty_positions = set(find_empty(state) for state in current_belief)
        
        for empty_pos in empty_positions:
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_i, new_j = empty_pos[0] + di, empty_pos[1] + dj
                if 0 <= new_i < 3 and 0 <= new_j < 3:
                    new_states = set()
                    for state in current_belief:
                        if find_empty(state) == empty_pos:
                            possible_states = get_nondeterministic_neighbors(state)
                            new_states.update(possible_states)
                    
                    new_belief = frozenset(new_states)
                    if new_belief and new_belief not in visited:
                        visited.add(new_belief)
                        rep_state = min(new_states, key=lambda s: manhattan_distance(s, goal))
                        new_path = path + [rep_state]
                        queue.append((new_belief, new_path))
    
    return None

# Cache để lưu trữ kết quả manhattan_distance và find_empty
_manhattan_cache = {}
_empty_cache = {}

def manhattan_distance(state, goal):
    state_tuple = tuple(map(tuple, state))
    if state_tuple not in _manhattan_cache:
        distance = 0
        for i in range(3):
            for j in range(3):
                value = state[i][j]
                if value != 0:
                    goal_pos = next((gi, gj) for gi in range(3) for gj in range(3) if goal[gi][gj] == value)
                    distance += abs(i - goal_pos[0]) + abs(j - goal_pos[1])
        _manhattan_cache[state_tuple] = distance
    return _manhattan_cache[state_tuple]

def find_empty(state):
    state_tuple = tuple(map(tuple, state))
    if state_tuple not in _empty_cache:
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    _empty_cache[state_tuple] = (i, j)
                    break
    return _empty_cache[state_tuple]

# Thuật toán Partial Observation Search tối ưu
# Cache để lưu trữ kết quả manhattan_distance và find_empty
_manhattan_cache = {}
_empty_cache = {}

def manhattan_distance(state, goal):
    state_tuple = tuple(map(tuple, state))
    if state_tuple not in _manhattan_cache:
        distance = 0
        for i in range(3):
            for j in range(3):
                value = state[i][j]
                if value != 0:
                    goal_pos = next((gi, gj) for gi in range(3) for gj in range(3) if goal[gi][gj] == value)
                    distance += abs(i - goal_pos[0]) + abs(j - goal_pos[1])
        _manhattan_cache[state_tuple] = distance
    return _manhattan_cache[state_tuple]

def find_empty(state):
    state_tuple = tuple(map(tuple, state))
    if state_tuple not in _empty_cache:
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    _empty_cache[state_tuple] = (i, j)
                    break
    return _empty_cache[state_tuple]

# Thuật toán Partial Observation Search tối ưu
def partial_observation_search(start, goal):
    if not is_solvable(start, goal):
        return None
    
    initial_empty = find_empty(start)
    initial_tile_00 = start[0][0]
    initial_observation = (initial_empty, initial_tile_00)
    initial_belief = frozenset(get_belief_state(start, initial_observation))
    
    # Sử dụng hàng đợi ưu tiên thay vì deque
    queue = [(manhattan_distance(start, goal), initial_belief, [start])]
    heapq.heapify(queue)
    visited = {initial_belief}
    max_iterations = 10000  # Tăng giới hạn sau khi tối ưu
    
    is_running = True
    
    while queue and is_running and max_iterations > 0:
        _, current_belief, path = heapq.heappop(queue)
        rep_state = path[-1]  # Trạng thái đại diện
        
        # Kiểm tra nhanh với trạng thái đại diện trước
        if rep_state == goal and all(state == goal for state in current_belief):
            return path
        
        empty_pos = find_empty(rep_state)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_i, new_j = empty_pos[0] + di, empty_pos[1] + dj
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_observation = ((new_i, new_j), initial_tile_00)
                new_states = set()
                for state in current_belief:
                    neighbors = get_neighbors(state)
                    for neighbor in neighbors:
                        if find_empty(neighbor) == (new_i, new_j):
                            new_states.add(neighbor)
                new_belief = frozenset(new_states)
                if new_belief and new_belief not in visited:
                    visited.add(new_belief)
                    rep_state = min(new_states, key=lambda s: manhattan_distance(s, goal))
                    new_path = path + [rep_state]
                    # Ưu tiên dựa trên manhattan_distance
                    priority = manhattan_distance(rep_state, goal)
                    heapq.heappush(queue, (priority, new_belief, new_path))
        
        max_iterations -= 1
    
    return None

def simulated_annealing(start, goal):
    if not is_solvable(start, goal):
        return None
    
    current = start
    parent = {start: None}
    temperature = 1000.0
    cooling_rate = 0.99  # Slower cooling
    max_iterations = 5000  # Increased limit
    while is_running and max_iterations > 0:
        if current == goal:
            return reconstruct_path(parent, current)
        
        temperature *= cooling_rate
        if temperature < 0.01:
            break
            
        neighbors = get_neighbors(current)
        if not neighbors:
            break
            
        next_state = random.choice(neighbors)
        current_h = manhattan_distance(current, goal)
        next_h = manhattan_distance(next_state, goal)
        delta_h = next_h - current_h
        
        try:
            if delta_h <= 0 or random.random() < math.exp(-delta_h / max(temperature, 1e-10)):
                parent[next_state] = current
                current = next_state
        except OverflowError:
            pass
            
        max_iterations -= 1
    
    return None

def beam_search(start, goal, beam_width=3):
    if not is_solvable(start, goal):
        return None
    
    beam = [(manhattan_distance(start, goal), start)]
    parent = {start: None}
    visited = set()
    
    while beam and is_running:
        new_beam = []
        for _, current in beam:
            if current == goal:
                return reconstruct_path(parent, current)
                
            if current in visited:
                continue
                
            visited.add(current)
            neighbors = get_neighbors(current)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    h = manhattan_distance(neighbor, goal)
                    new_beam.append((h, neighbor))
                    parent[neighbor] = current
        
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
    
    return None

def genetic_algorithm(start, goal):
    if not is_solvable(start, goal):
        return None
    
    def state_to_flat(state):
        return [tile for row in state for tile in row]
    
    def flat_to_state(flat):
        return tuple(tuple(flat[i * 3:(i + 1) * 3]) for i in range(3))
    
    def fitness(state):
        return -manhattan_distance(state, goal)
    
    def crossover(parent1, parent2):
        flat1, flat2 = state_to_flat(parent1), state_to_flat(parent2)
        point = random.randint(1, 7)
        child = flat1[:point] + flat2[point:]
        # Repair child to ensure all tiles 0-8 are present
        required = set(range(9))
        current = set(child)
        if current != required:
            missing = required - current
            duplicates = [x for x in child if child.count(x) > 1]
            for i, d in enumerate(duplicates[:len(missing)]):
                child[child.index(d)] = missing.pop()
        return flat_to_state(child) if set(child) == required else None
    
    def mutate(state):
        flat = state_to_flat(state)
        i, j = random.sample(range(9), 2)
        flat[i], flat[j] = flat[j], flat[i]
        return flat_to_state(flat)
    
    population_size = 100  # Increased population
    max_generations = 200
    mutation_rate = 0.2  # Increased mutation
    
    population = [start]
    for _ in range(population_size - 1):
        flat = state_to_flat(start)
        random.shuffle(flat)
        new_state = flat_to_state(flat)
        if is_solvable(new_state, goal):
            population.append(new_state)
    
    generation = 0
    parent = {start: None}
    
    while is_running and generation < max_generations:
        fitness_scores = [(fitness(state), state) for state in population]
        fitness_scores.sort(reverse=True)
        
        if fitness_scores[0][0] == 0:
            solution = fitness_scores[0][1]
            return reconstruct_path(parent, solution)
        
        new_population = [s for _, s in fitness_scores[:population_size // 4]]
        
        while len(new_population) < population_size:
            parent1, parent2 = random.choices([s for _, s in fitness_scores[:population_size // 2]], k=2)
            child = crossover(parent1, parent2)
            if child:
                if random.random() < mutation_rate:
                    child = mutate(child)
                if is_solvable(child, goal):
                    new_population.append(child)
                    parent[child] = parent1
        
        population = new_population
        generation += 1
    
    return None

def no_observation_search(start, goal):
    if not is_solvable(start, goal):
        return None
    
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    
    while queue and is_running:
        current = queue.popleft()
        if current == goal:
            return reconstruct_path(parent, current)
        
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                parent[neighbor] = current
    
    return None

def csp_backtracking(start, goal, canvas=None):
    if not is_solvable(start, goal):
        return None
    
    # A* Search
    queue = [(0 + manhattan_distance(start, goal), 0, start)]  # (f_score, g_score, state)
    visited = set()
    parent = {start: None}
    g_scores = {start: 0}
    visited_counts = [1]  # Theo dõi số trạng thái đã thăm
    
    while queue and is_running:
        queue.sort(key=lambda x: x[0])  # Sắp xếp theo f_score (thay vì dùng heapq để đơn giản)
        f_score, g_score, current = queue.pop(0)
        
        if current in visited:
            continue
            
        visited.add(current)
        visited_counts.append(len(visited))
        
        if current == goal:
            if canvas:
                canvas.clear()
                x_data = list(range(len(visited_counts)))
                canvas.plot(x_data, visited_counts, "Visited States (CSP Backtracking - A*)", "Iteration", "Visited Count", color="purple")
            return reconstruct_path(parent, current)
        
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                new_g_score = g_score + 1
                if neighbor not in g_scores or new_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = new_g_score
                    f_score = new_g_score + manhattan_distance(neighbor, goal)
                    queue.append((f_score, new_g_score, neighbor))
                    parent[neighbor] = current
    
    return None

def csp_forward_checking(start, goal, canvas=None):
    if not is_solvable(start, goal):
        return None
    
    # Greedy Best-First Search
    queue = [(manhattan_distance(start, goal), start)]  # (h_score, state)
    visited = set()
    parent = {start: None}
    visited_counts = [1]  # Theo dõi số trạng thái đã thăm
    
    while queue and is_running:
        queue.sort(key=lambda x: x[0])  # Sắp xếp theo h_score
        h_score, current = queue.pop(0)
        
        if current in visited:
            continue
            
        visited.add(current)
        visited_counts.append(len(visited))
        
        if current == goal:
            if canvas:
                canvas.clear()
                x_data = list(range(len(visited_counts)))
                canvas.plot(x_data, visited_counts, "Visited States (CSP Forward Checking - Greedy)", "Iteration", "Visited Count", color="cyan")
            return reconstruct_path(parent, current)
        
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                h_score = manhattan_distance(neighbor, goal)
                queue.append((h_score, neighbor))
                parent[neighbor] = current
    
    return None

def csp_min_conflicts(start, goal):
    if not is_solvable(start, goal):
        return None
    
    # Simulated Annealing
    current = start
    parent = {start: None}
    temperature = 1000.0
    cooling_rate = 0.995
    max_steps = 5000
    steps = 0
    
    while steps < max_steps and is_running:
        if current == goal:
            return reconstruct_path(parent, current)
        
        temperature *= cooling_rate
        if temperature < 0.01:
            break
        
        neighbors = get_neighbors(current)
        if not neighbors:
            break
        
        next_state = random.choice(neighbors)
        current_h = manhattan_distance(current, goal)
        next_h = manhattan_distance(next_state, goal)
        delta_h = next_h - current_h
        
        try:
            if delta_h <= 0 or random.random() < math.exp(-delta_h / max(temperature, 1e-10)):
                parent[next_state] = current
                current = next_state
        except OverflowError:
            pass
        
        steps += 1
    
    return None

def q_learning(start, goal):
    global is_running
    if not is_solvable(start, goal):
        return None
    
    # Use a dictionary with LRU-like behavior to limit memory
    q_table = {}
    max_q_table_size = 100000  # Maximum number of states to store
    state_order = []  # Track order of states for LRU eviction
    learning_rate = 0.3  # Increased for faster learning
    discount_factor = 0.85  # Slightly reduced to favor short-term rewards
    exploration_rate = 1.0
    min_exploration_rate = 0.05  # Higher minimum to maintain some exploration
    exploration_decay = 0.997  # Slightly faster decay
    max_episodes = 5000  # Reduced episodes with better initialization
    goal_reached_count = 0  # Track how often goal is reached
    
    def state_key(state):
        return tuple(tuple(row) for row in state)
    
    def manage_q_table(state_str):
        """Limit Q-table size by removing least recently used states"""
        if len(q_table) >= max_q_table_size:
            oldest_state = state_order.pop(0)
            if oldest_state in q_table:
                del q_table[oldest_state]
        if state_str not in state_order:
            state_order.append(state_str)
    
    def get_action(state):
        neighbors = get_neighbors(state)
        if not neighbors:
            return state
        state_str = state_key(state)
        if random.random() < exploration_rate:
            return random.choice(neighbors)
        else:
            if state_str not in q_table:
                q_table[state_str] = {state_key(n): -manhattan_distance(n, goal) / 10.0 for n in neighbors}
                manage_q_table(state_str)
            return max(q_table[state_str].items(), key=lambda x: x[1])[0]
    
    def reward(state):
        if state == goal:
            return 50  # Reduced reward to balance with Manhattan-based penalty
        return -manhattan_distance(state, goal) / 5.0  # Normalized Manhattan distance
    
    parent = {start: None}
    
    # Training phase
    for episode in range(max_episodes):
        current = start
        steps = 0
        max_steps = 500  # Reduced max steps per episode
        
        while steps < max_steps:
            # Check for UI events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    global paused
                    if paused:
                        paused = False
            
            state_str = state_key(current)
            neighbors = get_neighbors(current)
            if not neighbors:
                break
            if state_str not in q_table:
                q_table[state_str] = {state_key(n): -manhattan_distance(n, goal) / 10.0 for n in neighbors}
                manage_q_table(state_str)
            
            action = get_action(current)
            next_state = action
            
            next_state_str = state_key(next_state)
            next_neighbors = get_neighbors(next_state)
            if next_state_str not in q_table:
                q_table[next_state_str] = {state_key(n): -manhattan_distance(n, goal) / 10.0 for n in next_neighbors}
                manage_q_table(next_state_str)
            
            r = reward(next_state)
            
            current_q = q_table[state_str][state_key(next_state)]
            next_max_q = max(q_table[next_state_str].values()) if q_table[next_state_str] else 0
            q_table[state_str][state_key(next_state)] = (1 - learning_rate) * current_q + learning_rate * (r + discount_factor * next_max_q)
            
            parent[next_state] = current
            current = next_state
            steps += 1
            
            if current == goal:
                goal_reached_count += 1
                # Adapt exploration rate based on success
                exploration_rate = max(min_exploration_rate, exploration_rate * 0.95)
                return reconstruct_path(parent, current)
        
        # Update exploration rate
        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)
        
        # Early stopping if goal is reached frequently
        if goal_reached_count > 10:
            break
    
    if not is_running:
        return None
    
    # Path construction phase
    current = start
    path = [start]
    steps = 0
    max_steps = 500
    
    while current != goal and steps < max_steps and is_running:
        state_str = state_key(current)
        neighbors = get_neighbors(current)
        if state_str not in q_table:
            q_table[state_str] = {state_key(n): -manhattan_distance(n, goal) / 10.0 for n in neighbors}
            manage_q_table(state_str)
        try:
            action = max(q_table[state_str].items(), key=lambda x: x[1])[0]
        except ValueError:
            return None
        next_state = action
        path.append(next_state)
        parent[next_state] = current
        current = next_state
        steps += 1
    
    if current == goal:
        return path
    return None

# UI Functions
def draw_gradient(surface, rect, color1, color2, vertical=True):
    x, y, w, h = rect
    screen_rect = surface.get_rect()
    clipped_rect = pygame.Rect(max(0, x), max(0, y), min(w, screen_rect.width - x), min(h, screen_rect.height - y))
    if clipped_rect.width <= 0 or clipped_rect.height <= 0:
        return
    if vertical:
        for i in range(clipped_rect.height):
            alpha = i / clipped_rect.height
            pygame.draw.line(surface, pygame.Color(*color2).lerp(color1, alpha),
                             (clipped_rect.x, clipped_rect.y + i),
                             (clipped_rect.x + clipped_rect.width, clipped_rect.y + i))
    else:
        for i in range(clipped_rect.width):
            alpha = i / clipped_rect.width
            pygame.draw.line(surface, pygame.Color(*color2).lerp(color1, alpha),
                             (clipped_rect.x + i, clipped_rect.y),
                             (clipped_rect.x + i, clipped_rect.y + clipped_rect.height))

def draw_button(screen, rect, text, is_hovered, is_active, is_pressed, base_color, hover_color, active_color, text_color, border_color):
    color = active_color if is_active else (hover_color if is_hovered else base_color)
    draw_gradient(screen, rect, color, (255, 255, 255), False)
    pygame.draw.rect(screen, border_color, rect, 2, border_radius=5)
    text_surface = SMALL_FONT.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

def draw_board(screen, state, prev_state=None):
    screen.fill(BACKGROUND)
    shadow_rect = pygame.Rect(5, 5, WIDTH - 10, WIDTH - 10)
    pygame.draw.rect(screen, SHADOW_COLOR, shadow_rect, border_radius=15)
    pygame.draw.rect(screen, BORDER_COLOR, (0, 0, WIDTH, WIDTH), 5, border_radius=10)
    moved_tile = None
    if prev_state:
        for i in range(3):
            for j in range(3):
                if state[i][j] != prev_state[i][j] and state[i][j] != 0:
                    moved_tile = (i, j)
                    break
            if moved_tile:
                break

    for i in range(3):
        for j in range(3):
            value = state[i][j]
            if value != 0:
                tile = pygame.Rect(j * TILE_SIZE + 10, i * TILE_SIZE + 10, TILE_SIZE - 20, TILE_SIZE - 20)
                if (i, j) == moved_tile:
                    draw_gradient(screen, tile, HIGHLIGHT_COLOR, (255, 255, 224))
                else:
                    draw_gradient(screen, tile, TILE_COLOR, TILE_HOVER)
                pygame.draw.rect(screen, BORDER_COLOR, tile, 2, border_radius=5)
                text = FONT.render(str(value), True, TEXT_COLOR)
                text_rect = text.get_rect(center=tile.center)
                screen.blit(text, text_rect)

def draw_buttons(screen, selected_method=None):
    global paused, button_scroll_y, button_max_scroll
    methods = ["BFS", "DFS", "UCS", "IDS", "GSS", "A*", "IDA*", "SHC", "SAHC", "STHC", 
               "NDS", "POS", "SA", "BS", "GA", "NOS", "CSP-BT", "CSP-FC", "CSP-MC", "QL"]
    mouse_pos = pygame.mouse.get_pos()
    buttons = []

    button_frame = pygame.Rect(50, WIDTH + 60, WIDTH - 100, 180)
    content_rect = pygame.Rect(button_frame.x, button_frame.y, button_frame.width - 20, button_frame.height)
    pygame.draw.rect(screen, SHADOW_COLOR, button_frame.move(5, 5), border_radius=10)
    pygame.draw.rect(screen, BLACK, button_frame, 3, border_radius=10)

    btn_width = (content_rect.width - 50) // 4
    btn_height = 40
    btn_spacing = 15

    total_rows = (len(methods) + 3) // 4
    content_height = total_rows * (btn_height + btn_spacing) + 20
    button_max_scroll = max(0, content_height - button_frame.height)
    button_scroll_y = max(0, min(button_scroll_y, button_max_scroll))

    screen.set_clip(content_rect)

    for i, method in enumerate(methods):
        row = i // 4
        col = i % 4
        btn_x = content_rect.x + 10 + col * (btn_width + btn_spacing)
        btn_y = content_rect.y + 10 + row * (btn_height + btn_spacing) - button_scroll_y

        if content_rect.y <= btn_y <= content_rect.y + content_rect.height - btn_height:
            btn_rect = pygame.Rect(btn_x, btn_y, btn_width, btn_height)
            btn_rect.clamp_ip(screen.get_rect())
            buttons.append((btn_rect, method))

            is_hovered = btn_rect.collidepoint(mouse_pos)
            is_active = method == selected_method
            draw_button(screen, btn_rect, method, is_hovered, is_active, False,
                        BUTTON_COLOR, BUTTON_HOVER, BUTTON_ACTIVE, TEXT_COLOR, BORDER_COLOR)

    screen.set_clip(None)

    scrollbar_rect = pygame.Rect(button_frame.right - 20, button_frame.y, 20, button_frame.height)
    pygame.draw.rect(screen, (200, 200, 200), scrollbar_rect, border_radius=10)
    if button_max_scroll > 0:
        thumb_height = max(20, button_frame.height * (button_frame.height / content_height))
        thumb_y = button_frame.y + (button_scroll_y / button_max_scroll) * (button_frame.height - thumb_height)
        thumb_rect = pygame.Rect(button_frame.right - 20, thumb_y, 20, thumb_height)
        is_hovered = thumb_rect.collidepoint(mouse_pos)
        thumb_color = BUTTON_HOVER if is_hovered else BUTTON_COLOR
        pygame.draw.rect(screen, thumb_color, thumb_rect, border_radius=10)
    else:
        thumb_rect = pygame.Rect(0, 0, 0, 0)

    step_forward_rect = pygame.Rect((WIDTH - 140) // 2 - 110, WIDTH + 4, 140, 25)
    is_hovered = step_forward_rect.collidepoint(mouse_pos)
    draw_button(screen, step_forward_rect, "Step Forward", is_hovered, False, False,
                PAUSE_COLOR, (167, 132, 239), PAUSE_COLOR, TEXT_COLOR, BORDER_COLOR)
    buttons.append((step_forward_rect, "Step Forward"))

    step_backward_rect = pygame.Rect((WIDTH - 140) // 2 - 110, WIDTH + 31, 140, 25)
    is_hovered = step_backward_rect.collidepoint(mouse_pos)
    draw_button(screen, step_backward_rect, "Step Back", is_hovered, False, False,
                PAUSE_COLOR, (167, 132, 239), PAUSE_COLOR, TEXT_COLOR, BORDER_COLOR)
    buttons.append((step_backward_rect, "Step Back"))

    reset_rect = pygame.Rect((WIDTH - 140) // 2 + 40, WIDTH + 10, 100, 40)
    is_hovered = reset_rect.collidepoint(mouse_pos)
    draw_button(screen, reset_rect, "Reset", is_hovered, False, False,
                RESET_COLOR, (255, 185, 20), RESET_COLOR, TEXT_COLOR, BORDER_COLOR)
    buttons.append((reset_rect, "Reset"))

    pause_rect = pygame.Rect((WIDTH - 140) // 2 + 150, WIDTH + 10, 100, 40)
    is_hovered = pause_rect.collidepoint(mouse_pos)
    pause_text = "Pause" if not paused else "Continue"
    draw_button(screen, pause_rect, pause_text, is_hovered, False, False,
                PAUSE_COLOR, (167, 132, 239), PAUSE_COLOR, TEXT_COLOR, BORDER_COLOR)
    buttons.append((pause_rect, "Pause"))

    back_rect = pygame.Rect((WIDTH - 140) // 2 + 260, WIDTH + 10, 100, 40)
    is_hovered = back_rect.collidepoint(mouse_pos)
    draw_button(screen, back_rect, "Back", is_hovered, False, False,
                RESET_COLOR, (255, 185, 20), RESET_COLOR, TEXT_COLOR, BORDER_COLOR)
    buttons.append((back_rect, "Back"))

    return buttons, scrollbar_rect, thumb_rect

def draw_status(screen, steps):
    # Vẽ nút hiển thị số bước, căn chỉnh với các nút điều khiển khác
    step_rect = pygame.Rect(10, WIDTH + 10, 100, 40)  # Tăng chiều rộng và căn chỉnh vị trí
    mouse_pos = pygame.mouse.get_pos()
    is_hovered = step_rect.collidepoint(mouse_pos)
    draw_button(screen, step_rect, f"Steps: {steps}", is_hovered, False, False,
                BUTTON_COLOR, BUTTON_HOVER, BUTTON_COLOR, TEXT_COLOR, BORDER_COLOR)

def show_warning(screen, clock):
    global paused, is_running
    paused = True
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    screen.blit(overlay, (0, 0))
    warning_rect = pygame.Rect(WIDTH//4, HEIGHT//4, WIDTH//2, HEIGHT//2)
    draw_gradient(screen, warning_rect, WARNING_BG, (255, 245, 238))
    pygame.draw.rect(screen, BORDER_COLOR, warning_rect, 3, border_radius=10)
    text = SMALL_FONT.render("Algorithm running...", True, TEXT_COLOR)
    screen.blit(text, (warning_rect.centerx - text.get_width()//2, warning_rect.y + 20))
    stop_rect = pygame.Rect(warning_rect.x + 50, warning_rect.bottom - 80, 100, 40)
    cont_rect = pygame.Rect(warning_rect.right - 150, warning_rect.bottom - 80, 100, 40)
    draw_button(screen, stop_rect, "Stop", False, False, False,
                RESET_COLOR, RESET_COLOR, RESET_COLOR, TEXT_COLOR, BORDER_COLOR)
    draw_button(screen, cont_rect, "Continue", False, False, False,
                BUTTON_ACTIVE, BUTTON_ACTIVE, BUTTON_ACTIVE, TEXT_COLOR, BORDER_COLOR)
    pygame.display.flip()
    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if stop_rect.collidepoint(event.pos):
                    is_running = False
                    paused = False
                    return "stop"
                if cont_rect.collidepoint(event.pos):
                    paused = False
                    return "continue"
        clock.tick(FPS)

def show_no_solution_warning(screen, clock):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    screen.blit(overlay, (0, 0))
    warning_rect = pygame.Rect(WIDTH//4, HEIGHT//4, WIDTH//2, HEIGHT//2)
    draw_gradient(screen, warning_rect, WARNING_BG, (255, 245, 238))
    pygame.draw.rect(screen, BORDER_COLOR, warning_rect, 3, border_radius=10)
    text = SMALL_FONT.render("No solution found", True, TEXT_COLOR)
    screen.blit(text, (warning_rect.centerx - text.get_width()//2, warning_rect.y + 20))
    ok_rect = pygame.Rect(warning_rect.centerx - 50, warning_rect.bottom - 80, 100, 40)
    draw_button(screen, ok_rect, "OK", False, False, False,
                RESET_COLOR, RESET_COLOR, RESET_COLOR, TEXT_COLOR, BORDER_COLOR)
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if ok_rect.collidepoint(event.pos):
                    waiting = False
        clock.tick(FPS)

# Screen Classes
class CentralScreen:
    def __init__(self, screen):
        self.screen = screen
        self.methods = ["BFS", "DFS", "UCS", "IDS", "GSS", "A*", "IDA*", "SHC", "SAHC", "STHC", "LS", "NDS", "POS",
                        "SA", "BS", "GA", "NOS", "CSP-BT", "CSP-FC", "CSP-MC", "QL", "Play All"]

    def draw(self):
        self.screen.fill(BACKGROUND)
        title = FONT.render("8-Puzzle Solver", True, TEXT_COLOR)
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))

        btn_width = 120
        btn_height = 50
        spacing = 20
        total_width = 3 * btn_width + 2 * spacing
        start_x = (WIDTH - total_width) // 2
        start_y = 150

        buttons = []
        for i, method in enumerate(self.methods):
            row = i // 3
            col = i % 3
            btn_x = start_x + col * (btn_width + spacing)
            btn_y = start_y + row * (btn_height + spacing)
            btn_rect = pygame.Rect(btn_x, btn_y, btn_width, btn_height)
            buttons.append((btn_rect, method))

            mouse_pos = pygame.mouse.get_pos()
            is_hovered = btn_rect.collidepoint(mouse_pos)
            draw_button(self.screen, btn_rect, method, is_hovered, False, False,
                        BUTTON_COLOR, BUTTON_HOVER, BUTTON_ACTIVE, TEXT_COLOR, BORDER_COLOR)

        return buttons

class GuideScreen:
    def __init__(self, screen, method):
        self.screen = screen
        self.method = method
        self.scroll_y = 0
        self.max_scroll = 0
        self.dragging = False
        self.descriptions = {
        "BFS": (
                "Breadth-First Search",
                "Tìm kiếm theo từng mức, đảm bảo đường đi ngắn nhất.",
                "Description: Tìm kiếm theo chiều rộng khám phá tất cả \n"
                "các nút ở độ sâu \n"
                "hiện tại trước khi chuyển sang độ sâu tiếp theo, đảm bảo \n"
                "tìm đường ngắn nhất trong đồ thị không trọng số.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), \n"
                "(4, 3, 1)).\n"
                "2. Ô trống (0) tại (1,0). Các bước di chuyển: Xuống → \n"
                "((2, 6, 5), \n"
                "(4, 8, 7), (0, 3, 1)), Phải → ((2, 6, 5), (8, 0, 7),\n"
                "(4, 3, 1)).\n"
                "3. Thêm các trạng thái con vào hàng đợi, xử lý theo thứ \n"
                "tự FIFO.\n"
                "4. Lặp lại: Lấy trạng thái ra khỏi hàng đợi, tạo các \n"
                "trạng thái con,\n"
                "kiểm tra mục tiêu.\n"
                "5. Dừng khi tìm thấy ((1, 2, 3), (4, 5, 6), (7, 8, 0)), \n"
                "thường mất 20-30 bước. \n"
                "Ưu điểm: [ \n"
                    "- Đảm bảo tìm được đường đi ngắn nhất trong đồ thị \n"
                    "- không trọng số. \n"
                    "- Hoàn chỉnh, luôn tìm được giải pháp nếu tồn tại. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Tiêu tốn bộ nhớ lớn, cần lưu trữ tất cả trạng \n"
                    "- thái đã thăm. \n"
                    "- Thời gian tìm kiếm lâu, có thể mất nhiều bước \n"
                    "- để tìm ra giải pháp. \n"
                    "- Không hiệu quả với các bài toán lớn. \n"
                    "] \n"
            ),
            "DFS": (
                "Depth-First Search",
                "Khám phá một nhánh sâu trước khi quay lại.",
                "Description: Tìm kiếm theo chiều sâu khám phá một nhánh \n"
                "đến mức sâu nhất \n"
                "trước khi quay lại thử nhánh khác, phù hợp cho không gian \n"
                "trạng thái sâu.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), (4, 3, 1)).\n"
                "2. Ô trống (0) tại (1,0). Chọn nhánh đầu tiên: Xuống → \n"
                "((2, 6, 5), \n"
                "(4, 8, 7), (0, 3, 1)).\n"
                "3. Tiếp tục sâu hơn: Ô trống (0) tại (2,0), di chuyển \n"
                "Phải → ((2, 6, 5), \n"
                "(4, 8, 7), (3, 0, 1)).\n"
                "4. Nếu gặp ngõ cụt, quay lại trạng thái trước và thử \n"
                "nhánh khác.\n"
                "5. Dừng khi tìm thấy ((1, 2, 3), (4, 5, 6), (7, 8, 0)), \n"
                "có thể mất 50+ bước. \n"
                "Ưu điểm: [ \n"
                    "- Tốn ít bộ nhớ, chỉ lưu trữ đường đi hiện tại. \n"
                    "- Hiệu quả cho không gian trạng thái sâu. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Có thể bị kẹt trong vòng lặp vô hạn nếu không \n"
                    "- kiểm tra chu kỳ. \n"
                    "- Không đảm bảo đường đi ngắn nhất. \n"
                    "- Không hiệu quả với các bài toán có không gian \n"
                    "- trạng thái rộng. \n"
                    "] \n",
            ),
            "UCS": (
                "Uniform Cost Search",
                "Ưu tiên đường đi có chi phí thấp nhất.",
                "Description: Tìm kiếm chi phí đồng đều ưu tiên trạng thái \n"
                "có tổng chi phí \n"
                "đường đi thấp nhất, đảm bảo giải pháp tối ưu trong đồ thị \n"
                "có trọng số.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), (4, 3, 1)), chi phí = 0.\n"
                "2. Ô trống (0) tại (1,0). Các bước di chuyển: Xuống → \n"
                "((2, 6, 5), \n"
                "(4, 8, 7), (0, 3, 1)) (chi phí 1), Phải → ((2, 6, 5), \n"
                "(8, 0, 7), (4, 3, 1)) (chi phí 1).\n"
                "3. Thêm vào hàng đợi ưu tiên, sắp xếp theo chi phí \n"
                "tăng dần.\n"
                "4. Lấy trạng thái có chi phí thấp nhất, tạo các trạng \n"
                "thái con, tiếp tục.\n"
                "5. Dừng khi tìm thấy ((1, 2, 3), (4, 5, 6), (7, 8, 0)), \n"
                "thường mất 20-30 bước. \n"
                "Ưu điểm: [ \n"
                    "- Đảm bảo đường đi ngắn nhất trong đồ thị có \n"
                    "- trọng số. \n"
                    "- Hoàn chỉnh nếu chi phí luôn dương. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Tốn nhiều bộ nhớ tương tự BFS. \n"
                    "- Chậm nếu không gian trạng thái lớn. \n"
                    "- Không sử dụng heuristic để tối ưu hóa. \n"
                    "] \n",
            ),
            "IDS": (
                "Iterative Deepening Search",
                "Kết hợp BFS và DFS, tăng độ sâu từng bước.",
                "Description: Tìm kiếm lặp sâu dần kết hợp ưu điểm của \n"
                "BFS và DFS, \n"
                "tăng giới hạn độ sâu từng bước để tìm giải pháp tối ưu \n"
                "với ít bộ nhớ hơn.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), (4, 3, 1)), độ sâu 0.\n"
                "2. Độ sâu 1: Ô trống (0) tại (1,0). Các bước di chuyển: \n"
                "Xuống → ((2, 6, 5), \n"
                "(4, 8, 7), (0, 3, 1)), Phải → ((2, 6, 5), (8, 0, 7), \n"
                "(4, 3, 1)).\n"
                "3. Nếu không tìm thấy mục tiêu, tăng độ sâu lên 2, \n"
                "lặp lại DFS.\n"
                "4. Tiếp tục tăng độ sâu (3, 4, ...) cho đến khi tìm \n"
                "thấy mục tiêu.\n"
                "5. Dừng khi tìm thấy ((1, 2, 3), (4, 5, 6), (7, 8, 0)), \n"
                "thường mất 20-30 bước. \n"
                "Ưu điểm: [ \n"
                    "- Kết hợp ưu điểm của BFS (đảm bảo đường đi \n"
                    "- ngắn nhất) và DFS (ít bộ nhớ). \n"
                    "- Hoàn chỉnh và tối ưu cho đồ thị không trọng số. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Lặp lại các trạng thái ở các độ sâu thấp, gây \n"
                    "- lãng phí thời gian. \n"
                    "- Không hiệu quả nếu không gian trạng thái rất sâu. \n"
                    "- Không sử dụng heuristic để tối ưu hóa. \n"
                    "] \n",
            ),
            "GSS": (
                "Greedy Search",
                "Ưu tiên trạng thái gần mục tiêu nhất theo heuristic.",
                "Description: Tìm kiếm tham lam chọn trạng thái có giá trị \n"
                "heuristic thấp \n"
                "nhất (như khoảng cách Manhattan) để tiến gần mục tiêu \n"
                "nhanh nhất.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), (4, 3, 1)), heuristic (Manhattan) = 14.\n"
                "2. Ô trống (0) tại (1,0). Các bước di chuyển: Xuống → \n"
                "((2, 6, 5), \n"
                "(4, 8, 7), (0, 3, 1)) (h=14), Phải → ((2, 6, 5), \n"
                "(8, 0, 7), (4, 3, 1)) (h=14).\n"
                "3. Chọn trạng thái có h thấp nhất, tiếp tục.\n"
                "4. Lặp lại, luôn ưu tiên heuristic thấp nhất.\n"
                "5. Dừng khi tìm thấy ((1, 2, 3), (4, 5, 6), (7, 8, 0)), \n"
                "thường mất 10-20 bước nhưng không tối ưu. \n"
                "Ưu điểm: [ \n"
                    "- Nhanh chóng khi heuristic tốt. \n"
                    "- Tốn ít bộ nhớ hơn BFS hoặc UCS. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Không đảm bảo đường đi ngắn nhất. \n"
                    "- Có thể bị kẹt ở cực đại cục bộ. \n"
                    "- Hiệu quả phụ thuộc vào chất lượng heuristic. \n"
                    "] \n",
            ),
            "A*": (
                "A* Search",
                "Kết hợp chi phí và heuristic để tìm đường đi tối ưu.",
                "Description: Tìm kiếm A* kết hợp chi phí đường đi (g) \n"
                "và heuristic (h) \n"
                "để ưu tiên trạng thái có tổng chi phí dự đoán (f = g + h) \n"
                "thấp nhất.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), (4, 3, 1)), g = 0, h (Manhattan) = 14, f = 14.\n"
                "2. Ô trống (0) tại (1,0). Các bước di chuyển: Xuống → \n"
                "((2, 6, 5), \n"
                "(4, 8, 7), (0, 3, 1)) (g=1, h=14, f=15), Phải → \n"
                "((2, 6, 5), (8, 0, 7), (4, 3, 1)) (g=1, h=14, f=15).\n"
                "3. Thêm vào hàng đợi ưu tiên, sắp xếp theo f tăng dần.\n"
                "4. Lấy trạng thái có f thấp nhất, tạo các trạng thái \n"
                "con, tiếp tục.\n"
                "5. Dừng khi tìm thấy ((1, 2, 3), (4, 5, 6), (7, 8, 0)), \n"
                "thường mất 20-30 bước. \n"
                "Ưu điểm: [ \n"
                    "- Tối ưu nếu heuristic hợp lệ (như Manhattan). \n"
                    "- Hiệu quả hơn BFS và UCS. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Tốn bộ nhớ để lưu trữ hàng đợi ưu tiên. \n"
                    "- Chậm nếu heuristic không chính xác. \n"
                    "- Không hiệu quả với không gian trạng thái rất lớn. \n"
                    "] \n",
            ),
            "IDA*": (
                "Iterative Deepening A*",
                "A* kết hợp lặp sâu dần, tiết kiệm bộ nhớ.",
                "Description: Tìm kiếm A* lặp sâu dần sử dụng ngưỡng \n"
                "dựa trên f (g + h) \n"
                "và tăng dần để tìm giải pháp tối ưu với ít bộ nhớ hơn A*.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), (4, 3, 1)), ngưỡng = h = 14.\n"
                "2. Ô trống (0) tại (1,0). Các bước di chuyển: Xuống → \n"
                "((2, 6, 5), \n"
                "(4, 8, 7), (0, 3, 1)) (f=15 > 14, cắt), Phải → \n"
                "((2, 6, 5), (8, 0, 7), (4, 3, 1)) (f=15 > 14, cắt).\n"
                "3. Tăng ngưỡng lên 15, lặp lại.\n"
                "4. Tiếp tục tăng ngưỡng cho đến khi tìm thấy mục tiêu.\n"
                "5. Dừng khi tìm thấy ((1, 2, 3), (4, 5, 6), (7, 8, 0)), \n"
                "thường mất 20-30 bước. \n"
                "Ưu điểm: [ \n"
                    "- Tốn ít bộ nhớ hơn A*. \n"
                    "- Vẫn đảm bảo tối ưu với heuristic hợp lệ. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Lặp lại trạng thái, gây lãng phí thời gian. \n"
                    "- Chậm hơn A* nếu không gian trạng thái lớn. \n"
                    "- Phụ thuộc vào chất lượng heuristic. \n"
                    "] \n"
            ),
            "SHC": (
                "Simple Hill Climbing",
                "Chọn láng giềng đầu tiên tốt hơn.",
                "Description: Tìm kiếm leo đồi đơn giản chọn láng giềng \n"
                "đầu tiên có \n"
                "heuristic tốt hơn trạng thái hiện tại để tiến tới mục tiêu.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), (4, 3, 1)), h = 14.\n"
                "2. Ô trống (0) tại (1,0). Các bước di chuyển: Xuống → \n"
                "((2, 6, 5), \n"
                "(4, 8, 7), (0, 3, 1)) (h=14), Phải → ((2, 6, 5), \n"
                "(8, 0, 7), (4, 3, 1)) (h=14).\n"
                "3. Chọn láng giềng đầu tiên có h thấp hơn.\n"
                "4. Lặp lại cho đến khi không cải thiện.\n"
                "5. Có thể không đạt ((1, 2, 3), (4, 5, 6), (7, 8, 0)) \n"
                "nếu kẹt ở cực đại cục bộ. \n"
                "Ưu điểm: [ \n"
                    "- Đơn giản và nhanh cho các bài toán nhỏ. \n"
                    "- Tốn ít bộ nhớ. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Dễ kẹt ở cực đại cục bộ. \n"
                    "- Không đảm bảo tìm được giải pháp. \n"
                    "- Hiệu quả thấp với không gian trạng thái phức tạp. \n"
                    "] \n"
            ),
            "SAHC": (
                "Steepest-Ascent Hill Climbing",
                "Chọn láng giềng tốt nhất.",
                "Description: Tìm kiếm leo đồi dốc nhất chọn láng giềng \n"
                "có heuristic \n"
                "tốt nhất (h thấp nhất) trong tất cả láng giềng của trạng \n"
                "thái hiện tại.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), (4, 3, 1)), h = 14.\n"
                "2. Ô trống (0) tại (1,0). Các bước di chuyển: Xuống → \n"
                "((2, 6, 5), \n"
                "(4, 8, 7), (0, 3, 1)) (h=14), Phải → ((2, 6, 5), \n"
                "(8, 0, 7), (4, 3, 1)) (h=14).\n"
                "3. So sánh tất cả láng giềng, chọn h thấp nhất.\n"
                "4. Lặp lại cho đến khi không cải thiện.\n"
                "5. Có thể không đạt mục tiêu nếu kẹt ở cực đại cục bộ. \n"
                "Ưu điểm: [ \n"
                    "- Tìm giải pháp tốt hơn Simple Hill Climbing. \n"
                    "- Vẫn đơn giản và ít tốn bộ nhớ. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Vẫn có thể kẹt ở cực đại cục bộ. \n"
                    "- Không đảm bảo giải pháp tối ưu. \n"
                    "- Yêu cầu kiểm tra tất cả láng giềng, tốn thời gian hơn SHC. \n"
                    "] \n"
            ),
            "STHC": (
                "Stochastic Hill Climbing",
                "Chọn ngẫu nhiên láng giềng tốt hơn.",
                "Description: Tìm kiếm leo đồi ngẫu nhiên chọn ngẫu nhiên \n"
                "một láng giềng \n"
                "có heuristic tốt hơn trạng thái hiện tại, tăng khả năng \n"
                "thoát cực đại cục bộ.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), (4, 3, 1)), h = 14.\n"
                "2. Ô trống (0) tại (1,0). Các bước di chuyển: Xuống → \n"
                "((2, 6, 5), \n"
                "(4, 8, 7), (0, 3, 1)) (h=14), Phải → ((2, 6, 5), \n"
                "(8, 0, 7), (4, 3, 1)) (h=14).\n"
                "3. Chọn ngẫu nhiên láng giềng, di chuyển nếu h giảm.\n"
                "4. Lặp lại cho đến khi không cải thiện.\n"
                "5. Có thể thoát cực đại cục bộ nhưng không đảm bảo. \n"
                "Ưu điểm: [ \n"
                    "- Có khả năng thoát cực đại cục bộ nhờ tính ngẫu nhiên. \n"
                    "- Đơn giản và ít tốn bộ nhớ. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Không đảm bảo giải pháp tối ưu. \n"
                    "- Hiệu quả phụ thuộc vào yếu tố ngẫu nhiên. \n"
                    "- Có thể chậm nếu chọn nhầm láng giềng. \n"
                    "] \n",
            ),
            "LS": (
                "Local Search",
                "Cải thiện cục bộ với bước di chuyển tốt hơn đầu tiên.",
                "Description: Tìm kiếm cục bộ chọn láng giềng đầu tiên \n"
                "cải thiện heuristic \n"
                "so với trạng thái hiện tại, tương tự Simple Hill Climbing.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), (4, 3, 1)), h = 14.\n"
                "2. Ô trống (0) tại (1,0). Các bước di chuyển: Xuống → \n"
                "((2, 6, 5), \n"
                "(4, 8, 7), (0, 3, 1)) (h=14), Phải → ((2, 6, 5), \n"
                "(8, 0, 7), (4, 3, 1)) (h=14).\n"
                "3. Chọn láng giềng cải thiện đầu tiên.\n"
                "4. Lặp lại cho đến khi không cải thiện.\n"
                "5. Có thể không đạt mục tiêu nếu kẹt ở cực đại cục bộ. \n"
                "Ưu điểm: [ \n"
                    "- Đơn giản và nhanh cho các bài toán nhỏ. \n"
                    "- Tốn ít bộ nhớ. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Dễ kẹt ở cực đại cục bộ. \n"
                    "- Không đảm bảo tìm được giải pháp. \n"
                    "- Hiệu quả thấp với không gian trạng thái phức tạp. \n"
                    "] \n",
            ),
            "NDS": (
                "Nondeterministic Search",
                "Khám phá tất cả kết quả có thể của hành động.",
                "Description: Tìm kiếm không xác định khám phá tất cả \n"
                "kết quả có thể \n"
                "của mỗi hành động, duy trì tập niềm tin về trạng thái.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((1, 2, 3), (4, 5, 6), (7, 0, 8)), tập niềm tin = {trạng thái ban đầu}.\n"
                "2. Ô trống (0) tại (2,1). Thử Phải: Kết quả có thể bao \n"
                "gồm ((1, 2, 3), \n"
                "(4, 5, 6), (7, 8, 0)).\n"
                "3. Cập nhật tập niềm tin với tất cả trạng thái có thể, \n"
                "thêm vào hàng đợi BFS.\n"
                "4. Lặp lại, kiểm tra nếu tất cả trạng thái niềm tin đạt \n"
                "((1, 2, 3), (4, 5, 6), (7, 8, 0)).\n"
                "5. Dừng khi đạt mục tiêu hoặc không có giải pháp. \n"
                "Ưu điểm: [ \n"
                    "- Xử lý tốt các bài toán không xác định. \n"
                    "- Hoàn chỉnh nếu có giải pháp. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Tốn nhiều bộ nhớ để lưu tập niềm tin. \n"
                    "- Chậm khi không gian trạng thái lớn. \n"
                    "- Phức tạp hơn các thuật toán thông thường. \n"
                    "] \n",
            ),
            "POS": (
                "Partial Observation Search",
                "Tìm kiếm với thông tin quan sát hạn chế.",
                "Description: Tìm kiếm quan sát một phần sử dụng thông \n"
                "tin hạn chế \n"
                "(như vị trí ô trống) để duy trì tập niềm tin và tìm mục tiêu.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((1, 2, 3), (4, 5, 6), (7, 0, 8)), quan sát: ô trống (2,1), (0,0)=1.\n"
                "2. Tạo tập niềm tin: tất cả trạng thái có ô trống tại \n"
                "(2,1) và (0,0)=1.\n"
                "3. Di chuyển ô trống Phải: Ô trống mới tại (2,2), \n"
                "(0,0)=1. Cập nhật niềm tin.\n"
                "4. Lặp lại, kiểm tra nếu tất cả trạng thái niềm tin đạt \n"
                "((1, 2, 3), (4, 5, 6), (7, 8, 0)).\n"
                "5. Dừng khi đạt mục tiêu hoặc không có giải pháp. \n"
                "Ưu điểm: [ \n"
                    "- Thích hợp cho các bài toán quan sát hạn chế. \n"
                    "- Có thể tối ưu hóa bằng heuristic. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Tốn bộ nhớ để lưu trữ tập niềm tin. \n"
                    "- Phức tạp hơn các thuật toán thông thường. \n"
                    "- Chậm nếu tập niềm tin lớn. \n"
                    "] \n",
            ),
            "SA": (
                "Simulated Annealing",
                "Tìm kiếm xác suất, chấp nhận giải pháp tệ hơn.",
                "Description: Tìm kiếm ủ nhiệt chấp nhận các trạng thái \n"
                "tệ hơn với xác \n"
                "suất dựa trên nhiệt độ để thoát cực đại cục bộ và tìm \n"
                "giải pháp tốt.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), (4, 3, 1)), nhiệt độ cao (1000).\n"
                "2. Ô trống (0) tại (1,0). Di chuyển ngẫu nhiên: Phải → \n"
                "((2, 6, 5), \n"
                "(8, 0, 7), (4, 3, 1)).\n"
                "3. So sánh heuristic: Di chuyển nếu tốt hơn hoặc xác \n"
                "suất (exp(-Δh/T)) cao.\n"
                "4. Giảm nhiệt độ (T = T * 0.995), lặp lại.\n"
                "5. Dừng khi tìm thấy ((1, 2, 3), (4, 5, 6), (7, 8, 0)) \n"
                "hoặc T quá thấp. \n"
                "Ưu điểm: [ \n"
                    "- Có thể thoát cực đại cục bộ. \n"
                    "- Hiệu quả cho bài toán phức tạp. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Không đảm bảo giải pháp tối ưu. \n"
                    "- Phụ thuộc vào tham số nhiệt độ. \n"
                    "- Có thể chậm nếu nhiệt độ giảm không phù hợp. \n"
                    "] \n",
            ),
            "BS": (
                "Beam Search",
                "Giữ số lượng cố định các trạng thái tốt nhất.",
                "Description: Tìm kiếm chùm giữ lại một số lượng cố định \n"
                "(độ rộng chùm) \n"
                "các trạng thái tốt nhất dựa trên heuristic ở mỗi bước.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), (4, 3, 1)), độ rộng chùm = 3, h = 14.\n"
                "2. Ô trống (0) tại (1,0). Các bước di chuyển: Xuống → \n"
                "((2, 6, 5), \n"
                "(4, 8, 7), (0, 3, 1)), Phải → ((2, 6, 5), (8, 0, 7), \n"
                "(4, 3, 1)).\n"
                "3. Tạo tất cả trạng thái con, giữ 3 trạng thái có h \n"
                "thấp nhất.\n"
                "4. Lặp lại, mở rộng chùm, giữ các trạng thái tốt nhất.\n"
                "5. Dừng khi tìm thấy ((1, 2, 3), (4, 5, 6), (7, 8, 0)). \n"
                "Ưu điểm: [ \n"
                    "- Tốn ít bộ nhớ hơn BFS hoặc A*. \n"
                    "- Nhanh khi độ rộng chùm nhỏ. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Không đảm bảo giải pháp tối ưu. \n"
                    "- Hiệu quả phụ thuộc vào độ rộng chùm. \n"
                    "- Có thể bỏ sót giải pháp nếu chùm quá hẹp. \n"
                    "] \n",
            ),
            "GA": (
                "Genetic Algorithm",
                "Tìm kiếm tiến hóa với quần thể trạng thái.",
                "Description: Thuật toán di truyền sử dụng quần thể trạng \n"
                "thái, lai ghép \n"
                "và đột biến để tiến hóa dần tới giải pháp tối ưu.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), (4, 3, 1)), tạo 50 trạng thái ngẫu nhiên.\n"
                "2. Đánh giá độ thích nghi (nghịch đảo khoảng cách \n"
                "Manhattan).\n"
                "3. Lai ghép: Kết hợp hai trạng thái cha tại điểm \n"
                "ngẫu nhiên.\n"
                "4. Đột biến: Hoán đổi hai ô ngẫu nhiên, xác suất 10%.\n"
                "5. Lặp lại, chọn các trạng thái tốt nhất, dừng khi tìm \n"
                "thấy ((1, 2, 3), (4, 5, 6), (7, 8, 0)). \n"
                "Ưu điểm: [ \n"
                    "- Hiệu quả cho bài toán phức tạp, không gian lớn. \n"
                    "- Có thể tìm giải pháp toàn cục. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Phụ thuộc vào tham số (kích thước quần thể, đột biến). \n"
                    "- Không đảm bảo giải pháp tối ưu. \n"
                    "- Tốn thời gian để hội tụ. \n"
                    "] \n",
            ),
            "NOS": (
                "No Observation Search",
                "Tìm kiếm mù, không có thông tin trạng thái.",
                "Description: Tìm kiếm không quan sát thực hiện tìm kiếm \n"
                "mù, không sử dụng \n"
                "heuristic, tương tự BFS nhưng không cần thông tin trạng thái.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), (4, 3, 1)).\n"
                "2. Ô trống (0) tại (1,0). Các bước di chuyển: Xuống, \n"
                "Phải, không dùng heuristic.\n"
                "3. Sử dụng BFS, thêm trạng thái con vào hàng đợi.\n"
                "4. Lặp lại cho đến khi tìm thấy ((1, 2, 3), (4, 5, 6), (7, 8, 0)).\n"
                "5. Không tối ưu, tương tự BFS. \n"
                "Ưu điểm: [ \n"
                    "- Đơn giản, không cần heuristic. \n"
                    "- Hoàn chỉnh nếu có giải pháp. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Tốn nhiều bộ nhớ như BFS. \n"
                    "- Chậm do không có thông tin hướng dẫn. \n"
                    "- Không hiệu quả với không gian trạng thái lớn. \n"
                    "] \n",
            ),
            "CSP-BT": (
                "CSP Backtracking",
                "Tìm kiếm dựa trên ràng buộc với quay lui.",
                "Description: Tìm kiếm quay lui CSP gán giá trị cho các \n"
                "ô dựa trên ràng \n"
                "buộc, quay lui nếu vi phạm để tìm giải pháp hợp lệ.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7),\n"
                " (4, 3, 1))"
                "2. Ô trống (0) tại (1,0). Thử các bước di chuyển: \n"
                "Xuống, Phải.\n"
                "3. Kiểm tra tính hợp lệ (9 ô duy nhất, bao gồm 0), \n"
                "tiếp tục.\n"
                "4. Quay lui nếu không tiến triển.\n"
                "5. Dừng khi tìm thấy ((1, 2, 3), (4, 5, 6), (7, 8, 0)). \n"
                "Ưu điểm: [ \n"
                    "- Hiệu quả cho bài toán ràng buộc. \n"
                    "- Có thể mở rộng với các ràng buộc phức tạp. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Chậm nếu không gian trạng thái lớn. \n"
                    "- Yêu cầu kiểm tra ràng buộc phức tạp. \n"
                    "- Có thể lặp lại nhiều trạng thái không cần thiết. \n"
                    "] \n",
            ),
            "CSP-FC": (
                "CSP Forward Checking",
                "CSP với kiểm tra phía trước để cắt tỉa miền.",
                "Description: Tìm kiếm CSP với kiểm tra phía trước cắt \n"
                "tỉa miền giá trị \n"
                "của các biến sau mỗi bước gán để giảm không gian tìm kiếm.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), \n"
                "(4, 3, 1)).\n"
                "2. Ô trống (0) tại (1,0). Lấy miền: Xuống, Phải.\n"
                "3. Kiểm tra phía trước: Giữ trạng thái không tăng \n"
                "heuristic quá nhiều.\n"
                "4. Tiếp tục quay lui, giảm số trạng thái cần kiểm tra.\n"
                "5. Dừng khi tìm thấy ((1, 2, 3), (4, 5, 6), (7, 8, 0)). \n"
                "Ưu điểm: [ \n"
                    "- Giảm không gian tìm kiếm nhờ kiểm tra phía trước. \n"
                    "- Hiệu quả hơn backtracking đơn thuần. \n"
                "] \n"
            "Nhược điểm: [ \n"
                    "- Vẫn chậm nếu miền lớn. \n"
                    "- Phụ thuộc vào heuristic. \n"
                    "- Phức tạp hơn backtracking thông thường. \n"
                "] \n",
            ),
            "CSP-MC": (
                "CSP Min-Conflicts",
                "CSP giảm thiểu xung đột.",
                "Description: Tìm kiếm CSP tối thiểu xung đột chọn trạng \n"
                "thái giảm số \n"
                "ràng buộc bị vi phạm nhất, sử dụng tính ngẫu nhiên để cải thiện.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), \n"
                "(4, 3, 1)).\n"
                "2. Ô trống (0) tại (1,0). Lấy láng giềng: Xuống, Phải.\n"
                "3. Chọn ngẫu nhiên láng giềng có ít xung đột nhất \n"
                "(heuristic).\n"
                "4. Lặp lại, di chuyển đến trạng thái giảm xung đột.\n"
                "5. Dừng khi tìm thấy ((1, 2, 3), (4, 5, 6), (7, 8, 0)) \n"
                "hoặc đạt số bước tối đa. \n"
                "Ưu điểm: [ \n"
                    "- Nhanh cho bài toán CSP lớn. \n"
                    "- Có khả năng thoát cực đại cục bộ. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Không đảm bảo giải pháp tối ưu. \n"
                    "- Hiệu quả phụ thuộc vào xung đột. \n"
                    "- Có thể không tìm được giải pháp nếu giới hạn bước. \n"
                    "] \n",
            ),
            "QL": (
                "Q-Learning",
                "Học tăng cường, học chính sách tối ưu.",
                "Description: Q-Learning là thuật toán học tăng cường học \n"
                "chính sách tối \n"
                "ưu bằng cách cập nhật bảng Q dựa trên phần thưởng và trạng \n"
                "thái tiếp theo.\n"
                "1. Bắt đầu từ trạng thái ban đầu: ((2, 6, 5), (0, 8, 7), \n"
                "(4, 3, 1)), khởi tạo bảng Q.\n"
                "2. Ô trống (0) tại (1,0). Chọn hành động (ngẫu nhiên \n"
                "hoặc giá trị Q): Xuống, Phải.\n"
                "3. Cập nhật giá trị Q: Q(s,a) = (1-α)Q(s,a) + α(R + \n"
                "γ max Q(s',a')).\n"
                "4. Lặp lại qua các tập, giảm tỷ lệ khám phá.\n"
                "5. Dừng khi tìm thấy ((1, 2, 3), (4, 5, 6), (7, 8, 0)). \n"
                "Ưu điểm: [ \n"
                    "- Học được chính sách tối ưu qua thời gian. \n"
                    "- Thích hợp cho bài toán động. \n"
                    "] \n"
                "Nhược điểm: [ \n"
                    "- Chậm trong giai đoạn học. \n"
                    "- Yêu cầu nhiều tài nguyên để lưu bảng Q. \n"
                    "- Phụ thuộc vào tham số học (α, γ). \n"
                    "] \n",
            )
        }

    def draw(self):
        self.screen.fill(BACKGROUND)
        guide_rect = pygame.Rect(50, 50, WIDTH - 100, HEIGHT - 150)
        content_rect = pygame.Rect(guide_rect.x, guide_rect.y, guide_rect.width - 20, guide_rect.height)
        pygame.draw.rect(self.screen, SHADOW_COLOR, guide_rect.move(5, 5), border_radius=10)
        pygame.draw.rect(self.screen, WARNING_BG, guide_rect, border_radius=10)
        pygame.draw.rect(self.screen, BORDER_COLOR, guide_rect, 3, border_radius=10)

        self.screen.set_clip(content_rect)
        title, desc, steps = self.descriptions[self.method]
        title_text = FONT.render(title, True, TEXT_COLOR)
        desc_text = SMALL_FONT.render(desc, True, TEXT_COLOR)
        steps_lines = steps.split('\n')
        steps_texts = [SMALL_FONT.render(line, True, TEXT_COLOR) for line in steps_lines]

        title_height = title_text.get_height()
        desc_height = desc_text.get_height()
        steps_height = sum(text.get_height() for text in steps_texts) + len(steps_texts) * 30
        content_height = title_height + desc_height + steps_height + 120
        self.max_scroll = max(0, content_height - content_rect.height)

        self.scroll_y = max(0, min(self.scroll_y, self.max_scroll))

        y_offset = content_rect.y - self.scroll_y
        self.screen.blit(title_text, (content_rect.centerx - title_text.get_width()//2, y_offset + 20))
        y_offset += 80
        self.screen.blit(desc_text, (content_rect.centerx - desc_text.get_width()//2, y_offset))
        y_offset += 40
        for step_text in steps_texts:
            self.screen.blit(step_text, (content_rect.x + 20, y_offset))
            y_offset += 30

        self.screen.set_clip(None)

        scrollbar_rect = pygame.Rect(guide_rect.right - 20, guide_rect.y, 20, guide_rect.height)
        pygame.draw.rect(self.screen, (200, 200, 200), scrollbar_rect, border_radius=10)
        if self.max_scroll > 0:
            thumb_height = max(20, guide_rect.height * (guide_rect.height / content_height))
            thumb_y = guide_rect.y + (self.scroll_y / self.max_scroll) * (guide_rect.height - thumb_height)
            thumb_rect = pygame.Rect(guide_rect.right - 20, thumb_y, 20, thumb_height)
            mouse_pos = pygame.mouse.get_pos()
            is_hovered = thumb_rect.collidepoint(mouse_pos)
            thumb_color = BUTTON_HOVER if is_hovered else BUTTON_COLOR
            pygame.draw.rect(self.screen, thumb_color, thumb_rect, border_radius=10)
        else:
            thumb_rect = pygame.Rect(0, 0, 0, 0)

        back_rect = pygame.Rect(guide_rect.x + 50, guide_rect.bottom + 20, 100, 40)
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = back_rect.collidepoint(mouse_pos)
        draw_button(self.screen, back_rect, "Back", is_hovered, False, False,
                    RESET_COLOR, (255, 185, 20), RESET_COLOR, TEXT_COLOR, BORDER_COLOR)

        return back_rect, scrollbar_rect, thumb_rect

    def handle_event(self, event):
        back_rect, scrollbar_rect, thumb_rect = self.draw()

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            if back_rect.collidepoint(mouse_pos):
                return "back"
            if thumb_rect.collidepoint(mouse_pos):
                self.dragging = True
            elif scrollbar_rect.collidepoint(mouse_pos):
                relative_y = mouse_pos[1] - scrollbar_rect.y
                scroll_ratio = relative_y / scrollbar_rect.height
                self.scroll_y = scroll_ratio * self.max_scroll

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_y = event.pos[1]
            scrollbar_height = scrollbar_rect.height
            thumb_height = thumb_rect.height if self.max_scroll > 0 else 0
            scrollable_track = scrollbar_height - thumb_height
            if scrollable_track > 0:
                track_y = (mouse_y - scrollbar_rect.y - thumb_height / 2) / scrollable_track
                track_y = max(0, min(1, track_y))
                self.scroll_y = track_y * self.max_scroll

        elif event.type == pygame.MOUSEWHEEL:
            if scrollbar_rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll_y -= event.y * 20
                self.scroll_y = max(0, min(self.scroll_y, self.max_scroll))

        return None

def main():
    global is_running, paused, algorithm_state, button_scroll_y, button_max_scroll, button_dragging
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("8-Puzzle Solver")
    clock = pygame.time.Clock()
    initial_states = [
        ((2, 6, 5), (8, 0, 7), (4, 3, 1)),
    ]
    current_state_index = 0
    initial_state = initial_states[current_state_index]
    goal_state = ((1, 2, 3), (4, 5, 6), (7, 8, 0))
    current_state = initial_state
    prev_state = None
    methods = {
        "BFS": bfs, "DFS": dfs, "UCS": ucs, "IDS": ids, "GSS": gss, "A*": a_star, "IDA*": ida_star,
        "SHC": simple_hill_climbing, "SAHC": steepest_ascent_hill_climbing,
        "STHC": stochastic_hill_climbing, "NDS": nondeterministic_search,
        "POS": partial_observation_search, "SA": simulated_annealing,
        "BS": beam_search, "GA": genetic_algorithm,
        "NOS": no_observation_search, "CSP-BT": csp_backtracking,
        "CSP-FC": csp_forward_checking, "CSP-MC": csp_min_conflicts,
        "QL": q_learning
    }
    selected_method = None
    current_screen = "central"
    central_screen = CentralScreen(screen)
    guide_screen = None
    last_update = pygame.time.get_ticks()
    step_interval = 500

    # Tạo thư mục để lưu ảnh tạm, GIF và biểu đồ
    temp_dir = "temp_puzzle_frames"
    charts_dir = "charts"
    try:
        # Xóa thư mục temp cũ nếu tồn tại
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Created temp directory: {os.path.abspath(temp_dir)}")
        # Kiểm tra quyền ghi
        test_file = os.path.join(temp_dir, "test_write.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"Write permission confirmed for {temp_dir}")
        # Tạo thư mục charts
        os.makedirs(charts_dir, exist_ok=True)
        test_file = os.path.join(charts_dir, "test_write.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"Write permission confirmed for {charts_dir}")
    except OSError as e:
        print(f"Error setting up directories {temp_dir} or {charts_dir}: {e}")
        return
    os.makedirs("gifs", exist_ok=True)

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    run = False
            if current_screen == "guide" and guide_screen:
                action = guide_screen.handle_event(event)
                if action == "back":
                    current_screen = "central"
                    guide_screen = None
                continue
            if current_screen == "game":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if button_thumb_rect.collidepoint(mouse_pos):
                        button_dragging = True
                    elif button_scrollbar_rect.collidepoint(mouse_pos):
                        relative_y = mouse_pos[1] - button_scrollbar_rect.y
                        scroll_ratio = relative_y / button_scrollbar_rect.height
                        button_scroll_y = scroll_ratio * button_max_scroll
                elif event.type == pygame.MOUSEBUTTONUP:
                    button_dragging = False
                elif event.type == pygame.MOUSEMOTION and button_dragging:
                    mouse_y = event.pos[1]
                    scrollbar_height = button_scrollbar_rect.height
                    thumb_height = button_thumb_rect.height if button_max_scroll > 0 else 0
                    scrollable_track = scrollbar_height - thumb_height
                    if scrollable_track > 0:
                        track_y = (mouse_y - button_scrollbar_rect.y - thumb_height / 2) / scrollable_track
                        track_y = max(0, min(1, track_y))
                        button_scroll_y = track_y * button_max_scroll
                elif event.type == pygame.MOUSEWHEEL:
                    if button_scrollbar_rect.collidepoint(pygame.mouse.get_pos()):
                        button_scroll_y -= event.y * 20
                        button_scroll_y = max(0, min(button_scroll_y, button_max_scroll))
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    for btn_rect, action in buttons:
                        if btn_rect.collidepoint(x, y):
                            if action == "Reset":
                                is_running = False
                                paused = False
                                algorithm_state = {"current_step": 0, "solution": None, "screenshots": []}
                                current_state = initial_state
                                prev_state = None
                                selected_method = None
                            elif action == "Pause":
                                if is_running:
                                    paused = not paused
                            elif action == "Step Back" and algorithm_state["solution"]:
                                if algorithm_state["current_step"] > 0:
                                    algorithm_state["current_step"] -= 1
                                    current_state = algorithm_state["solution"][algorithm_state["current_step"]]
                                    prev_state = algorithm_state["solution"][algorithm_state["current_step"] - 1] if algorithm_state["current_step"] > 0 else None
                            elif action == "Step Forward" and algorithm_state["solution"]:
                                if algorithm_state["current_step"] < len(algorithm_state["solution"]) - 1:
                                    prev_state = current_state
                                    algorithm_state["current_step"] += 1
                                    current_state = algorithm_state["solution"][algorithm_state["current_step"]]
                            elif action == "Back":
                                current_screen = "central"
                                is_running = False
                                paused = False
                                algorithm_state = {"current_step": 0, "solution": None, "screenshots": []}
                                current_state = initial_state
                                prev_state = None
                                selected_method = None
                            elif action in methods:
                                if not is_running:
                                    is_running = True
                                    selected_method = action
                                    algorithm_state["screenshots"] = []
                                    try:
                                        solution, time_taken, memory_usage = run_algorithm_with_stats(
                                            methods[action], initial_state, goal_state, action)
                                        if solution:
                                            algorithm_state["solution"] = solution
                                            algorithm_state["current_step"] = 0
                                            current_state = solution[0]
                                            prev_state = None
                                            steps = len(solution) - 1
                                            show_algorithm_stats(screen, clock, steps, time_taken, memory_usage)
                                        else:
                                            is_running = False
                                            selected_method = None
                                            show_no_solution_warning(screen, clock)
                                    except Exception as e:
                                        print(f"Error in {action}: {e}")
                                        is_running = False
                                        selected_method = None
                                        show_no_solution_warning(screen, clock)
                                else:
                                    action_result = show_warning(screen, clock)
                                    if action_result == "stop":
                                        is_running = False
                                        paused = False
                                        algorithm_state = {"current_step": 0, "solution": None, "screenshots": []}
                                        current_state = initial_state
                                        prev_state = None
                                        selected_method = None
            if current_screen == "central":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    buttons = central_screen.draw()
                    for btn_rect, method in buttons:
                        if btn_rect.collidepoint(x, y):
                            if method == "Play All":
                                current_screen = "game"
                                is_running = False
                                paused = False
                                algorithm_state = {"current_step": 0, "solution": None, "screenshots": []}
                                current_state = initial_state
                                prev_state = None
                                selected_method = None
                            else:
                                current_screen = "guide"
                                guide_screen = GuideScreen(screen, method)
        if current_screen == "central":
            central_screen.draw()
        elif current_screen == "guide" and guide_screen:
            guide_screen.draw()
        elif current_screen == "game":
            current_time = pygame.time.get_ticks()
            if is_running and not paused and algorithm_state["solution"] and current_time - last_update >= step_interval:
                if algorithm_state["current_step"] < len(algorithm_state["solution"]) - 1:
                    prev_state = current_state
                    algorithm_state["current_step"] += 1
                    current_state = algorithm_state["solution"][algorithm_state["current_step"]]
                    last_update = current_time
                    screenshot_file = os.path.join(temp_dir, f"step_{algorithm_state['current_step']}.png")
                    try:
                        pygame.image.save(screen, screenshot_file)
                        algorithm_state["screenshots"].append(screenshot_file)
                        print(f"Saved screenshot: {screenshot_file}")
                    except pygame.error as e:
                        print(f"Error saving screenshot {screenshot_file}: {e}")
                else:
                    is_running = False
                    if algorithm_state["screenshots"]:
                        safe_method_name = safe_filename(selected_method)
                        create_gif(algorithm_state["screenshots"], f"gifs/{safe_method_name}_solution.gif",
                                   algorithm_name=selected_method, duration=step_interval / 1000.0)
                    for screenshot in algorithm_state["screenshots"]:
                        if os.path.exists(screenshot):
                            try:
                                os.remove(screenshot)
                            except OSError as e:
                                print(f"Error removing screenshot {screenshot}: {e}")
                    if os.path.exists(temp_dir):
                        try:
                            os.rmdir(temp_dir)
                        except OSError as e:
                            print(f"Error removing temp directory {temp_dir}: {e}")
                    algorithm_state["screenshots"] = []
            draw_board(screen, current_state, prev_state)
            buttons, button_scrollbar_rect, button_thumb_rect = draw_buttons(screen, selected_method)
            steps = max(0, algorithm_state["current_step"] - 1) if algorithm_state["solution"] else 0
            draw_status(screen, steps)
        pygame.display.flip()
        clock.tick(FPS)
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except OSError as e:
                print(f"Error removing file {file}: {e}")
        try:
            os.rmdir(temp_dir)
        except OSError as e:
            print(f"Error removing temp directory {temp_dir}: {e}")

if __name__ == "__main__":
    main()