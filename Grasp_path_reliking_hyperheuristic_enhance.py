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
# 算子
# 1. 两点交换算子（2-opt）
def two_opt_operator(solution):
    """两点交换算子"""
    if len(solution) < 2:
        return solution  # 如果解的长度小于2，直接返回原解
    i = random.randint(0, len(solution) - 2)  # 确保i的范围有效
    j = random.randint(i + 1, len(solution) - 1)  # 确保j的范围有效
    # 执行两点交换操作
    solution[i:j + 1] = reversed(solution[i:j + 1])
    return solution

#9.两点交换顺序
def two_opt_swap(solution):
    # 随机选择两个不同的城市i和j
    i, j = random.sample(range(len(solution)), 2)
    # 确保i < j，方便交换
    if i > j:
        i, j = j, i
    # 交换路径中城市i和j的位置
    solution = solution[:i] + solution[i:j+1][::-1] + solution[j+1:]
    return solution

# 2. 三点交换算子（3-opt）
def three_opt_operator(solution):
    """3-opt 算子：对路径执行三点断开并重新连接"""
    if len(solution) < 3:  # 如果路径长度不足 3，直接返回原路径
        return solution[:]
    # 随机选择三个断点，确保断点有效
    i = random.randint(0, len(solution) - 3)
    j = random.randint(i + 1, len(solution) - 2)
    k = random.randint(j + 1, len(solution) - 1)
    # 执行三段反转的方式，选择一种新的排列方式
    new_solution = solution[:i] + solution[i:j][::-1] + solution[j:k][::-1] + solution[k:]
    return new_solution

# 3. 插入算子（Insertion Operator）
def insertion_operator(solution):
    if len(solution) < 2:
        return solution
    start_index = random.randint(0, len(solution) - 2)
    end_index = random.randint(start_index + 1, len(solution) - 1)
    sub_path = solution[start_index:end_index]
    del solution[start_index:end_index]
    insert_index = random.randint(0, len(solution))
    solution[insert_index:insert_index] = sub_path
    return solution


# 4. 相邻交换算子（Adjacent Swap Operator）
def adjacent_swap_operator(solution):
    """相邻交换算子"""
    # 如果解的长度小于2，直接返回原解
    if len(solution) < 2:
        return solution  # 或者根据需要返回一个适当的处理方式
    index = random.randint(0, len(solution) - 2)
    # 执行相邻交换操作
    solution[index], solution[index + 1] = solution[index + 1], solution[index]
    return solution

# 5. 块交换算子（Block Swap Operator）
def block_swap_operator(solution):
    """在路径中随机选择两个块并交换位置"""
    if len(solution) < 4:  # 如果路径长度不足，直接返回原路径
        return solution[:]
    block_size = random.randint(1, max(1, len(solution) // 4))  # 确保 block_size 有效
    start1 = random.randint(0, len(solution) - block_size)
    start2 = random.randint(0, len(solution) - block_size)
    # 确保两个块不重叠
    while abs(start1 - start2) < block_size:
        start2 = random.randint(0, len(solution) - block_size)
    new_solution = solution[:]
    # 交换两个块
    new_solution[start1:start1 + block_size], new_solution[start2:start2 + block_size] = \
        new_solution[start2:start2 + block_size], new_solution[start1:start1 + block_size]
    return new_solution

# 6. 反转块插入算子（Reversed Block Insertion Operator）
def reversed_block_insertion_operator(solution):
    """反向块插入算子"""
    if len(solution) < 2:
        return solution  # 如果解的长度小于2，直接返回原解
    start_index = random.randint(0, len(solution) - 2)  # 确保索引范围有效
    end_index = random.randint(start_index + 1, len(solution) - 1)  # 确保end_index有效
    # 选择并反向插入块
    block = solution[start_index:end_index + 1]
    solution = solution[:start_index] + list(reversed(block)) + solution[end_index + 1:]
    return solution


# 7. 交叉交换算子（Cross Exchange Operator）
def cross_exchange_operator(solution):
    if len(solution) < 3:
        return solution  # 如果路径长度小于3，无法进行交换，直接返回原路径
    # 随机选择第一个区间
    start_index_1 = random.randint(0, len(solution) - 3)
    end_index_1 = random.randint(start_index_1 + 2, len(solution) - 1)
    # 随机选择第二个区间，确保区间不重叠
    start_index_2 = random.randint(start_index_1, end_index_1 - 1)
    end_index_2 = random.randint(start_index_2 + 1, len(solution))
    # 提取原路径的四个部分：区间1外的部分、区间2内的部分、区间1内的部分、区间2外的部分
    part_1 = solution[:start_index_1]  # 区间1前面的部分
    part_2 = solution[start_index_2:end_index_2]  # 区间2部分
    part_3 = solution[start_index_1:end_index_1]  # 区间1部分
    part_4 = solution[end_index_2:]  # 区间2后面的部分
    # 记录已经添加的元素，用于去重
    seen = set()
    new_solution = []
    # 添加第一个部分（没有重复问题）
    new_solution.extend(part_1)
    # 添加第二个区间部分，确保没有重复元素
    for x in part_2:
        if x not in seen:
            new_solution.append(x)
            seen.add(x)
    # 添加第一个区间部分，确保没有重复元素
    for x in part_3:
        if x not in seen:
            new_solution.append(x)
            seen.add(x)
    # 添加第二个部分（没有重复问题）
    new_solution.extend(part_4)
    return new_solution

# 8. 随机打乱算子（Random Shuffle Operator）
def random_shuffle_operator(solution):
    """随机打乱算子"""
    # 如果解的长度小于2，直接返回原解
    if len(solution) < 2:
        return solution  # 或者根据需要返回一个适当的处理方式

    start_index = random.randint(0, len(solution) - 2)
    end_index = random.randint(start_index + 1, len(solution) - 1)

    # 执行打乱操作
    solution[start_index:end_index + 1] = random.sample(solution[start_index:end_index + 1],
                                                        len(solution[start_index:end_index + 1]))
    return solution


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
def calculate_distance(solution, distance_matrix):
    distance = 0
    for i in range(len(solution)):
        distance += distance_matrix[solution[i]][solution[(i + 1) % len(solution)]]
    return distance


# 生成一个初始路径解
def generate_initial_path_solution(distance_matrix):
    """
    使用最小路径法生成初始解：
    从第一个城市开始，选择最近的城市作为下一个访问城市，直到访问完所有城市。
    """
    num_cities = len(distance_matrix)
    visited = [False] * num_cities  # 记录哪些城市已经访问
    path_solution = [0]  # 从城市 0 开始
    visited[0] = True  # 标记第一个城市已访问
    current_city = 0  # 从城市 0 开始
    for _ in range(1, num_cities):
        # 找到距离当前城市最近的未访问城市
        min_distance = float('inf')
        next_city = -1
        for city in range(num_cities):
            if not visited[city] and distance_matrix[current_city][city] < min_distance:
                min_distance = distance_matrix[current_city][city]
                next_city = city
        # 将找到的城市加入路径，并标记为已访问
        path_solution.append(next_city)
        visited[next_city] = True
        current_city = next_city
    return path_solution

# 模拟一个简单的算子应用函数（这里根据算子名调用对应的算子函数应用到路径上）
def apply_operator(operator_name, solution):
    new_solution = solution[:]
    if operator_name == "two_opt_operator":
        new_solution = two_opt_operator(solution)
    elif operator_name == "three_opt_operator":
        new_solution = three_opt_operator(solution)
    elif operator_name == "insertion_operator":
        new_solution = insertion_operator(solution)
    elif operator_name == "adjacent_swap_operator":
        new_solution = adjacent_swap_operator(solution)
    elif operator_name == "block_swap_operator":
        new_solution = block_swap_operator(solution)
    elif operator_name == "reversed_block_insertion_operator":
        new_solution = reversed_block_insertion_operator(solution)
    elif operator_name == "cross_exchange_operator":
        new_solution = cross_exchange_operator(solution)
    elif operator_name == "random_shuffle_operator":
        new_solution = random_shuffle_operator(solution)
    elif operator_name == "two_opt_swap":
        new_solution = two_opt_swap(solution)
    return new_solution


def grasp(path_solution, best_solution, distance_matrix, alpha,long ,sequence:[], global_best_solutions_list:[],global_best_solutions_sequence_list:[]):
    # 定义候选算子列表
    candidate_operators = [
        "two_opt_operator", "two_opt_swap", "three_opt_operator", "insertion_operator",
        "adjacent_swap_operator", "block_swap_operator", "reversed_block_insertion_operator",
        "cross_exchange_operator", "random_shuffle_operator"
    ]

    # 存储优化后的解和对应的距离

    RCL_results = []
    #print("当前最好路径",best_solution)
    grasp_best_solution=best_solution[:]
    best_distance=calculate_distance(grasp_best_solution, distance_matrix)

    #print("第一个的距离",best_distance,grasp_best_solution)

    # 用每个算子优化当前解
    for operator in candidate_operators:
        a = path_solution[:]
        new_path_solution = apply_operator(operator, a)
        new_distance = calculate_distance(new_path_solution, distance_matrix)
        RCL_results.append((operator, new_distance, new_path_solution))
       # print("rcl_results",RCL_results)
    # 按距离升序排序，获取最优化的解
    RCL_results.sort(key=lambda x: x[1])

    # 构建 RCL（受限制候选列表），选择距离较好的解
    max_index = max(1, int(alpha * len(RCL_results)))  # 确保最大索引至少为 1
    RCL= RCL_results[:max_index + 1]


    # 随机选择一个算子并更新解
    selected_operator, selected_distance, selected_path_solution = random.choice(RCL)
    #print("执行hou长", selected_distance,grasp_best_solution,best_solution)
    # 存储更新后的解
    sequence.append(selected_operator)

    # 更新全局最优解表
    if selected_distance < best_distance:
        grasp_best_solution = selected_path_solution

       # print("原长",best_distance,selected_operator)
        #print("执行更新长",selected_distance,selected_path_solution)
        best_distance=selected_distance


    # 更新全局最优解列表（最多存储 10 个解）

    if (selected_path_solution, selected_distance) not in global_best_solutions_list:
        global_best_solutions_list.append((selected_path_solution, selected_distance))
    # 如果全局最优解列表超过 10 个，移除最差的解
    if len(global_best_solutions_list) > 10:
        global_best_solutions_list.sort(key=lambda x: x[1])  # 按距离升序排序
        global_best_solutions_list.pop()  # 移除最差的解
    if (grasp_best_solution, best_distance) not in global_best_solutions_list:
        global_best_solutions_list.append((grasp_best_solution, best_distance))
    if len(global_best_solutions_list) > 10:
        global_best_solutions_list.sort(key=lambda x: x[1])  # 按距离升序排序
        global_best_solutions_list.pop()  # 移除最差的解
    # 递归调用，继续优化，直到满足停止条件
    if len(sequence) < long:  # 或根据其他条件控制停止
        #print("下次调用前距离",best_distance,grasp_best_solution)
        return grasp(selected_path_solution, grasp_best_solution, distance_matrix, alpha,long, sequence, global_best_solutions_list, global_best_solutions_sequence_list)

    # 进行grasp的局部优化
    sequence, solution, distance,global_best_solutions_list,global_best_solutions_sequence_list = grasp_search(selected_path_solution, sequence,distance_matrix,global_best_solutions_list,global_best_solutions_sequence_list)
    #判断一下局部优化后有没有产生新的解
    if distance < best_distance:
        grasp_best_solution = solution[:]
        # 返回最终的解和最优解
    return sequence, solution,  grasp_best_solution,distance,global_best_solutions_list,global_best_solutions_sequence_list


def grasp_search(path_solution,sequence, distance_matrix,global_best_solutions_list,global_best_solutions_sequence_list):


    # 所有可用的算子
    all_operators = [
        "two_opt_operator","two_opt_swap", "three_opt_operator", "insertion_operator",
        "adjacent_swap_operator", "block_swap_operator",
        "reversed_block_insertion_operator", "cross_exchange_operator", "random_shuffle_operator"
    ]

    # 初始化当前最佳距离和算子序列
    best_current_distance = calculate_distance(path_solution, distance_matrix)
    best_sequence = sequence.copy()  # 当前的最优序列
    best_path_solution = path_solution.copy()  # 当前的最优路径

    # 用于存储每个替换操作后的结果
    operator_results = []

    # 遍历算子序列中的每个算子
    for replace_index in range(len(sequence)):
        # 遍历每个算子可以替换的ji种可能的其他算子
        for operator in all_operators:
            if operator != sequence[replace_index]:  # 确保不替换为同一个算子
                # 创建一个新的算子序列
                new_sequence = sequence.copy()
                new_sequence[replace_index] = operator  # 替换当前位置的算子

                # 复制原始路径解，准备进行算子应用
                new_path_solution = path_solution.copy()

                # 逐步应用新序列中的每个算子
                for op_name in new_sequence:
                    new_path_solution = apply_operator(op_name, new_path_solution)
                    new_distance = calculate_distance(new_path_solution, distance_matrix)
                    #加入判断，让局部搜索过程中优秀的解直接进入优秀表中
                    if (new_path_solution, new_distance) not in global_best_solutions_list:
                        global_best_solutions_list.append((new_path_solution, new_distance))

                    # 如果全局最优解列表超过 10 个，移除最差的解
                    if len(global_best_solutions_list) > 10:
                        global_best_solutions_list.sort(key=lambda x: x[1])  # 按距离升序排序
                        global_best_solutions_list.pop()  # 移除最差的解

                # 如果新路径的距离比当前最优距离更小，则更新最优解
                if new_distance < best_current_distance:
                    best_current_distance = new_distance
                    best_sequence = new_sequence
                    best_path_solution = new_path_solution
    #用于将要path——reliking的存入，是带算子序列的和 global_best_solutions_list不一样
    if (best_path_solution,best_sequence, new_distance) not in global_best_solutions_sequence_list:
        global_best_solutions_sequence_list.append((best_path_solution,best_sequence, new_distance))

    # 如果全局最优解列表超过 10 个，移除最差的解
    if len(global_best_solutions_sequence_list) > 10:
        global_best_solutions_sequence_list.sort(key=lambda x: x[2])  # 按距离升序排序
        global_best_solutions_sequence_list.pop()  # 移除最差的解
    # 返回最优的算子序列、路径解和路径距离
    return best_sequence, best_path_solution, best_current_distance, global_best_solutions_list,global_best_solutions_sequence_list


# 路径-重连方法 (Path Relinking)
def path_relinking(solution, sequence, distance_matrix, global_best_solutions_list,
                   global_best_solutions_sequence_list):
    # 随机从global_best_solutions_sequence_list中选择一个目标算子序列作为头

    random_index = random.randint(0, len(global_best_solutions_sequence_list) - 1)
    head_solution, head_sequence, head_distance = global_best_solutions_sequence_list[random_index]

    #目标从上次的解里找
    target_sequence = sequence
    head_solution =solution


    # 初始解和目标解的路径和序列
    current_solution = head_solution.copy()
    current_sequence = head_sequence.copy()

    # 逐步优化，尝试通过路径重连接近目标解
    while current_sequence != target_sequence:
        # 随机选择一个目标序列中的算子位置，替换为当前路径中的算子
        for replace_index in range(len(current_sequence)):
            if current_sequence[replace_index] != target_sequence[replace_index]:
                # 替换当前序列中的算子，接近目标序列
                current_sequence[replace_index] = target_sequence[replace_index]

                # 应用更新后的算子序列
                new_solution = current_solution.copy()
                for op_name in current_sequence:
                    new_solution = apply_operator(op_name, new_solution)

                    # 计算新的路径距离
                    new_distance = calculate_distance(new_solution, distance_matrix)

                    # 判断是否需要更新
                    if new_distance < calculate_distance(current_solution, distance_matrix):
                        current_solution = new_solution
                        # 更新全局最优解表
                        if (current_solution, new_distance) not in global_best_solutions_list:
                            global_best_solutions_list.append((current_solution, new_distance))

                        # 如果全局最优解列表超过10个，移除最差的解
                        if len(global_best_solutions_list) > 10:
                            global_best_solutions_list.sort(key=lambda x: x[1])
                            global_best_solutions_list.pop()

    # 更新global_best_solutions_sequence_list
        current_distance = calculate_distance(current_solution, distance_matrix)
        if (current_solution, current_sequence, current_distance) not in global_best_solutions_sequence_list:
            global_best_solutions_sequence_list.append((current_solution, current_sequence, current_distance))

        # 如果global_best_solutions_sequence_list超过10个，移除最差的解
        if len(global_best_solutions_sequence_list) > 10:
            global_best_solutions_sequence_list.sort(key=lambda x: x[2])
            global_best_solutions_sequence_list.pop()

    # 返回最终的最优解
    return current_solution, current_sequence,  global_best_solutions_list, global_best_solutions_sequence_list

#比较两个列表中最小的那一个
def compare_best_solutions(global_best_solutions_list, global_best_solutions_sequence_list):
    # 找到第一个表不带序列的最小的距离及其路径



    best_grasp_solution=global_best_solutions_list[0][0]
    best_grasp_distance=global_best_solutions_list[0][1]

    # 计算路径重连最优解的距离

    best_grasp_sequence_solution=global_best_solutions_sequence_list[0][1]
    best_grasp_sequence_distance=global_best_solutions_sequence_list[0][2]


    # 比较两个最小距离
    if best_grasp_distance < best_grasp_sequence_distance:

        max_distance = best_grasp_distance
        max_solution = best_grasp_solution

    else:

        max_distance = best_grasp_sequence_distance
        max_solution = best_grasp_sequence_solution
    return max_distance, max_solution



def plot_best_solution(city_coordinates, best_solution):
    # 从城市坐标中获取 x 和 y
    x_coords = [city_coordinates[i][0] for i in best_solution]
    y_coords = [city_coordinates[i][1] for i in best_solution]

    # 添加第一个城市到最后一个城市，以闭合路径
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])

    # 创建图形
    plt.figure(figsize=(12, 8))
    plt.plot(x_coords, y_coords, 'bo-', markerfacecolor='red', label='Best Path')

    # 标注城市
    for i, city in enumerate(best_solution):
        plt.text(city_coordinates[city][0] + 20, city_coordinates[city][1] + 20, str(city),
                 fontsize=9, ha='center', va='center', color='black')

    plt.title("Best Solution Path")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # 自动调整布局使得图像自适应
    plt.tight_layout()
    plt.grid(True)
    plt.show()


# 在总体方法中调用此函数
def overall_method(initial_path_solution, iterations, alpha, long, distance_matrix):
    global_best_solutions_list = []
    global_best_solutions_sequence_list = []

    # 初始路径解

    a=initial_path_solution
    b=initial_path_solution
    current_solution = a[:]
    best_solution=b[:]
    best_sequence = []

    # 执行GRASP迭代
    for i in range(1,iterations+1):
        print(f"Iteration {i }/{iterations}")
        sequence, solution,best_solution, distance, global_best_solutions_list, global_best_solutions_sequence_list = grasp(
            current_solution, best_solution, distance_matrix, alpha, long, [], global_best_solutions_list,
            global_best_solutions_sequence_list)
        print("迭代后的距离",calculate_distance(best_solution, distance_matrix))
        # 更新当前最优解
        current_solution = solution[:]
        best_sequence = sequence[:]

        # 执行路径重连
        if i % 50 == 0:
            best_reliking_solution, best_sequence, global_best_solutions_list1, global_best_solutions_sequence_list1 = path_relinking(
                current_solution, best_sequence, distance_matrix, global_best_solutions_list,
                global_best_solutions_sequence_list)
            if calculate_distance(best_reliking_solution, distance_matrix) < calculate_distance(best_solution, distance_matrix):
               current_solution = best_reliking_solution
            else:
                current_solution=best_solution[:]
            global_best_solutions_list = global_best_solutions_list1
            global_best_solutions_sequence_list = global_best_solutions_sequence_list1
        else:
            current_solution = best_solution[:]




    # 执行比较代码
    best_distance, best_solution_ok = compare_best_solutions(global_best_solutions_list,
                                                          global_best_solutions_sequence_list)
    # 输出最终的最优解和对应的路径
    print(global_best_solutions_list)
    print(global_best_solutions_sequence_list)
    print(f"Best Solution Distance: {best_distance}")
    print(f"Best Solution: {best_solution_ok}")

    # 绘制最优路径图
    plot_best_solution(city_coordinates, best_solution)

    # 返回最优解和路径
    return best_solution, best_sequence, best_distance


# 主程序
if __name__ == "__main__":
    # 计算距离矩阵
    distance_matrix = calculate_distance_matrix(city_coordinates)

    # 生成初始路径解
    initial_solution = generate_initial_path_solution(distance_matrix)
    #initial_solution=[7, 9, 3, 8, 10, 11, 1, 27, 12, 29, 24, 13, 26, 14, 23, 5, 4, 28, 25, 20, 16, 17, 21, 18, 19, 15, 22, 0, 6, 2]
    # 设置参数
    iterations = 5000  # GRASP迭代次数
    alpha = 0.6  # 受限候选列表的比例
    long = 5
    # 调用总体方法
    best_solution, best_sequence, best_distance = overall_method(initial_solution, iterations, alpha, long,
                                                                 distance_matrix)
