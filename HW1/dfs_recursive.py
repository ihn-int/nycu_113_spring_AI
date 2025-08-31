import csv
edgeFile = 'edges.csv'


def dfs(start, end):
    # Begin your code (Part 2)
    # raise NotImplementedError("To be implemented")

    # Create adjacent table
    graph = {}      # adjacency list
    distances = {}  # (start, end) -> distance

    with open(edgeFile, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header

        for row in reader:
            s, e, d = int(row[0]), int(row[1]), float(row[2])
            graph.setdefault(s, []).append(e)
            distances[(s, e)] = d

    # Record the visited node
    visited = set()
    num_visited = -1

    # Recursive DFS function
    def dfs_recursive(node, path, dist):
        nonlocal num_visited
        num_visited += 1
        visited.add(node)
        path.append(node)

        # Check if find end node
        if node == end:
            return path[:], dist

        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    new_dist = dist + distances.get((node, neighbor), 0)
                    result_path, result_dist = dfs_recursive(neighbor, path, new_dist)
                    if result_path:  # Find path
                        return result_path, result_dist

        # Not found, return
        path.pop()
        return [], float('inf')

    # Excecute recursive function
    # The `dfs` function need to create table, which isn't needed by recursive function
    # Use another function
    path, dist = dfs_recursive(start, [], 0)

    return path, dist, num_visited

    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
