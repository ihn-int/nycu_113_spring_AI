import csv
edgeFile = 'edges.csv'
heuristicFile = 'heuristic_values.csv'

import heapq

def astar(start, end):
    # Begin your code (Part 4)
    # raise NotImplementedError("To be implemented")

    # Create adjacent table
    graph = {}  # adjacency list
    with open(edgeFile, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            s, e, d = int(row[0]), int(row[1]), float(row[2])
            graph.setdefault(s, []).append((e, d))

    # Create heuristic value table
    heuristic = {}
    with open(heuristicFile, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Store the header
        target_idx = headers.index(str(end))  # Find the current column
        # `target_idx` should handle 10 test cases

        for row in reader:
            node = int(row[0])
            heuristic[node] = float(row[target_idx]) if row[target_idx] else float('inf')

    # Initialize A*
    pq = []  # priority queue
    heapq.heappush(pq, (heuristic.get(start, 0), 0, start, [start]))  # (f, g, current node, path)
    visited = {}  # Record the shortest g value
    num_visited = -1

    while pq:
        _, cost, node, path = heapq.heappop(pq)  # Pop the node with minimal f value
        num_visited += 1

        # Check if find end node
        if node == end:
            return path, cost, num_visited

        # Skip the expand if shorter path exists
        if node in visited and visited[node] < cost:
            continue

        visited[node] = cost  # Update the shortest g value

        # Check all the neighbors
        for neighbor, edge_cost in graph.get(node, []):
            new_cost = cost + edge_cost
            if neighbor not in visited or new_cost < visited[neighbor]:
                f_value = new_cost + heuristic.get(neighbor, float('inf'))  # f(n) = g(n) + h(n)
                heapq.heappush(pq, (f_value, new_cost, neighbor, path + [neighbor]))
                visited[neighbor] = new_cost  # Update g value

    return [], float('inf'), num_visited  # No path

    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
