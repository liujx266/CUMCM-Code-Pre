# M_B2_Dijkstra.py
import heapq

def dijkstra(graph, start_node):
    """
    使用Dijkstra算法计算单源最短路径

    参数:
    graph: dict, 图的邻接表示, e.g., {'A': {'B': 1, 'C': 4}, 'B': {'A': 1, ...}}
    start_node: a key in graph, 起始节点

    返回:
    distances: dict, 从起点到各节点的最短距离
    paths: dict, 从起点到各节点的最短路径
    """
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    paths = {node: [] for node in graph}
    paths[start_node] = [start_node]
    
    # 优先队列，存储 (距离, 节点)
    priority_queue = [(0, start_node)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
            
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[current_node] + [neighbor]
                heapq.heappush(priority_queue, (distance, neighbor))
                
    return distances, paths

if __name__ == '__main__':
    # --- 使用示例: 求解一个简单图的最短路径 ---
    # 1. 定义图结构 (邻接表)
    graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'C': 2, 'D': 5},
        'C': {'A': 4, 'B': 2, 'D': 1},
        'D': {'B': 5, 'C': 1}
    }
    start_node = 'A'

    # 2. 运行Dijkstra算法
    distances, paths = dijkstra(graph, start_node)

    # 3. 打印结果
    print(f"从节点 {start_node} 出发的最短路径:")
    for node, dist in distances.items():
        print(f"到节点 {node}: 距离 = {dist}, 路径 = {' -> '.join(paths[node])}")

    # 如何修改为你自己的问题:
    # 1. 根据你的问题构建 `graph` 字典，key是节点名，value是另一个字典，
    #    其中key是邻居节点名，value是边的权重/距离。
    # 2. 指定 `start_node`。
