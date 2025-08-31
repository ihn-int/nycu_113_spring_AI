import csv
edgeFile = 'edges.csv'

import queue

def bfs(start, end):
    # Begin your code (Part 1)
    # raise NotImplementedError("To be implemented")

    # Build adjacent list
    graph = {}      # adjacency list
    distances = {}  # (start, end) -> distance

    with open(edgeFile, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header

        for row in reader:
            s, e, d = int(row[0]), int(row[1]), float(row[2])
            graph.setdefault(s, []).append(e)
            distances[(s, e)] = d

    # Initialize BFS
    q = queue.Queue()
    q.put(start)
    visited = set([start])
    parent = {start: None}  # Path
    cost = {start: 0}       # Total distance
    num_visited = -1        # The start point shouldn't be record

    # BFS search
    while not q.empty():
        node = q.get()
        num_visited += 1

        # Check if find end node
        if node == end:
            break

        # Check if the node exists
        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.put(neighbor)
                    parent[neighbor] = node
                    cost[neighbor] = cost[node] + distances.get((node, neighbor), 0)

    # Get path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parent.get(current)
    path.reverse()

    # Not found, return
    if path[0] != start:
        return [], float('inf'), num_visited

    return path, cost[end], num_visited

    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
