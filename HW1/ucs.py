import csv
edgeFile = 'edges.csv'

import heapq

def ucs(start, end):
    # Begin your code (Part 3)
    # raise NotImplementedError("To be implemented")

    # Create adjacent table
    graph = {}  # adjacency list
    with open(edgeFile, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            s, e, d = int(row[0]), int(row[1]), float(row[2])
            graph.setdefault(s, []).append((e, d))

    # Initialize UCS
    pq = []  # Priority queue
    # (total distance, current node, path)
    heapq.heappush(pq, (0, start, [start]))
    visited = {}  # Record the shortest path
    num_visited = -1

    while pq:
        cost, node, path = heapq.heappop(pq)  # Pop the node with minimal distance
        num_visited += 1

        # Skip the expand if shorter path exists
        if node in visited and visited[node] < cost:
            continue

        visited[node] = cost  # Update the shortest distance

        # Check if find end node
        if node == end:
            return path, cost, num_visited

        # Check all the neighbors
        for neighbor, edge_cost in graph.get(node, []):
            new_cost = cost + edge_cost
            if neighbor not in visited or new_cost < visited[neighbor]:
                heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))
                visited[neighbor] = new_cost  # Update shortest path

    return [], float('inf'), num_visited  # No path

    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
