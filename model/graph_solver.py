import heapq

def min_fuel_cost(adj_matrix, num_nodes):

    N = num_nodes

    dist = [[float('inf')] * 2 for _ in range(N)]

    dist[0][0] = 0
    pq = [(0, 0, 0)]
    while pq:
        cost, node, orient = heapq.heappop(pq)
        if cost != dist[node][orient]:
            continue

        if node == N - 1:
            return cost

        if orient == 0:

            for j in range(N):
                if adj_matrix[node][j] == 1:
                    new_cost = cost + 1
                    if new_cost < dist[j][orient]:
                        dist[j][orient] = new_cost
                        heapq.heappush(pq, (new_cost, j, orient))
        else:

            for j in range(N):
                if adj_matrix[j][node] == 1:
                    new_cost = cost + 1
                    if new_cost < dist[j][orient]:
                        dist[j][orient] = new_cost
                        heapq.heappush(pq, (new_cost, j, orient))

        new_orient = 1 - orient
        new_cost = cost + N
        if new_cost < dist[node][new_orient]:
            dist[node][new_orient] = new_cost
            heapq.heappush(pq, (new_cost, node, new_orient))

    return -1
