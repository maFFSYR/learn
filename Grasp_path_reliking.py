import math
import random
import matplotlib.pyplot as plt

# 示例城市坐标
"""city_coordinates = [
    (1304, 3639), (3488, 3326), (3238, 4196), (4312, 4386), (3007, 2562),
    (2788, 2381), (1332, 3715), (3918, 4061), (3780, 3676), (4029, 4263),
    (3429, 3507), (3394, 3439), (2935, 3140), (2545, 2778), (2370, 2312),
    (1315, 2244), (1399, 1535), (1556, 1229), (1004, 790), (570, 1970),
    (1756, 1491), (1676, 695), (1678, 2179), (2370, 2212), (2578, 2838),
    (2931, 1908), (2367, 2643), (3201, 3240), (3550, 2357), (2826, 2975)
]"""
city_coordinates = [
    (1304, 3639), (3488, 3326), (3238, 4196), (4312, 4386), (3007, 2562),
    (2788, 2381), (1332, 3715), (3918, 4061), (3780, 3676), (4029, 4263),
    (3429, 3507), (3394, 3439), (2935, 3140), (2545, 2778), (2370, 2312),
    (1315, 2244), (1399, 1535), (1556, 1229), (1004, 790), (570, 1970),
    (1756, 1491), (1676, 695), (1678, 2179), (2370, 2212), (2578, 2838),
    (2931, 1908), (2367, 2643), (3201, 3240), (3550, 2357), (2826, 2975),
    (4021, 4123), (3789, 3423), (3055, 2907), (2784, 2317), (1456, 2434),
    (1901, 1542), (2112, 1284), (700, 1800), (900, 2500), (2900, 4000),
    (3456, 3300), (3200, 2500), (3700, 2100), (1000, 500), (4100, 4600),
    (500, 3200), (2400, 1800), (3300, 2700), (2100, 3100), (1500, 3700),
    (2800, 3900), (3000, 3400), (3500, 4000), (3700, 3700), (3800, 4500),
    (3600, 3000), (2400, 2600), (2200, 2300), (2500, 2900), (4000, 3000)
]

# 计算距离矩阵
def calculate_distance_matrix(city_coordinates):
    num_cities = len(city_coordinates)
    matrix = [[0] * num_cities for _ in range(num_cities)]
    for i in range(num_cities):
        for j in range(num_cities):
            x1, y1 = city_coordinates[i]
            x2, y2 = city_coordinates[j]
            matrix[i][j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return matrix


# 计算路径总距离
def calculate_total_distance(solution, distance_matrix):
    total_distance = 0
    for i in range(len(solution)):
        total_distance += distance_matrix[solution[i]][solution[(i + 1) % len(solution)]]
    return total_distance


# 贪婪随机构造解
def construct_greedy_random_solution(distance_matrix, alpha=0.3):
    num_cities = len(distance_matrix)
    unvisited = list(range(num_cities))
    solution = [unvisited.pop(random.randint(0, len(unvisited) - 1))]

    while unvisited:
        candidates = [(city, distance_matrix[solution[-1]][city]) for city in unvisited]
        candidates.sort(key=lambda x: x[1])
        max_index = int(alpha * len(candidates))
        restricted_list = candidates[:max_index + 1]
        next_city = random.choice(restricted_list)[0]
        solution.append(next_city)
        unvisited.remove(next_city)

    return solution


# 局部搜索：两点交换
def two_opt(solution, distance_matrix):
    best = solution[:]
    best_distance = calculate_total_distance(best, distance_matrix)
    for i in range(1, len(solution) - 1):
        for j in range(i + 1, len(solution)):
            new_solution = solution[:]
            new_solution[i:j] = reversed(solution[i:j])
            new_distance = calculate_total_distance(new_solution, distance_matrix)
            if new_distance < best_distance:
                best, best_distance = new_solution, new_distance
    return best


# 路径重连
def path_relinking(global_best_solution, current_solution, distance_matrix):
    best_solution = global_best_solution[:]
    best_distance = calculate_total_distance(best_solution, distance_matrix)
    differences = [(i, j) for i, j in zip(current_solution, global_best_solution) if i != j]
    for i in range(len(differences)):
        new_solution = current_solution[:]
        swap_index = current_solution.index(global_best_solution[i])
        current_index = current_solution.index(current_solution[i])
        new_solution[current_index], new_solution[swap_index] = new_solution[swap_index], new_solution[current_index]
        new_distance = calculate_total_distance(new_solution, distance_matrix)
        if new_distance < best_distance:
            best_solution, best_distance = new_solution, new_distance
    return best_solution, best_distance


# 超启发式算法：动态可视化（仅显示最终结果）
def hyper_grasp_with_path_relinking_visualized(city_coordinates, iterations=100, alpha=0.3):
    distance_matrix = calculate_distance_matrix(city_coordinates)
    global_best_solution = None
    global_best_distance = float('inf')

    # 进行迭代优化
    for iteration in range(1, iterations + 1):
        current_solution = construct_greedy_random_solution(distance_matrix, alpha)
        current_solution = two_opt(current_solution, distance_matrix)
        current_distance = calculate_total_distance(current_solution, distance_matrix)

        # 更新全局最优解
        if current_distance < global_best_distance:
            global_best_solution, global_best_distance = current_solution, current_distance

        # 每5次运行后触发Path-Relinking
        if iteration % 5 == 0:
            print(f"Iteration {iteration}: Running Path-Relinking...")
            relinked_solution, relinked_distance = path_relinking(global_best_solution, current_solution,
                                                                  distance_matrix)
            if relinked_distance < global_best_distance:
                global_best_solution, global_best_distance = relinked_solution, relinked_distance
                print(f"New Global Best Distance: {global_best_distance}")

    return global_best_solution, global_best_distance


def plot_tsp_path(city_coordinates, solution):
    # 创建绘图区域并调整大小
    plt.figure(figsize=(12, 7))

    # 提取所有城市的坐标
    x_coords, y_coords = zip(*city_coordinates)

    # 动态设置边距比例 (例如，10%的外边距)
    x_margin = (max(x_coords) - min(x_coords)) * 0.1
    y_margin = (max(y_coords) - min(y_coords)) * 0.1

    # 设置统一的坐标范围和边距
    plt.xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
    plt.ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)

    # 绘制路径
    for i in range(len(solution)):
        x1, y1 = city_coordinates[solution[i]]
        x2, y2 = city_coordinates[solution[(i + 1) % len(solution)]]
        plt.plot([x1, x2], [y1, y2], 'b-o', markersize=6, linewidth=1.5)

    # 标注城市编号
    for idx, (x, y) in enumerate(city_coordinates):
        plt.text(x, y, f"{idx}", fontsize=8, color='red', ha='center', va='center')

    # 确保坐标轴比例一致
    plt.gca().set_aspect('equal', adjustable='datalim')

    # 添加图形标题和标签
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.title("TSP Path Visualization", fontsize=16)
    plt.text(0.05, 0.95, f"Total Distance: {best_distance:.2f}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')
    plt.grid()
    plt.tight_layout()  # 自动调整布局
    plt.show()

# 主函数
if __name__ == "__main__":
    # 执行带有Path-Relinking的优化
    best_solution, best_distance = hyper_grasp_with_path_relinking_visualized(city_coordinates, iterations=2000,
                                                                              alpha=0.1)
    print("最佳路径:", best_solution)
    print("最短距离:", best_distance)
    plot_tsp_path(city_coordinates, best_solution)
    distance_matrix=calculate_distance_matrix(city_coordinates)
    a=calculate_total_distance( [0, 6, 15, 22, 20, 28, 25, 5, 4, 2, 3, 9, 7, 8, 1, 10, 11, 27, 12, 29, 24, 13, 26, 14, 23, 21, 18, 17, 16, 19],distance_matrix)
    print(a)