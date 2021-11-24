# class to represent a graph object
class Graph:

    # Constructor
    def __init__(self, edges, N):
        # A List of Lists to represent an adjacency list
        self.adjList = [[] for _ in range(N)]

        # stores in-degree of a vertex
        # initialize in-degree of each vertex by 0
        self.indegree = [0] * N

        # add edges to the undirected graph
        for (src, dest) in edges:
            # add an edge from source to destination
            self.adjList[src].append(dest)

            # increment in-degree of destination vertex by 1
            self.indegree[dest] = self.indegree[dest] + 1


# Recursive function to find
# all topological orderings of a given DAG
def findAllTopologicalOrders(graph, path, discovered, N, results):
    # do for every vertex
    for v in range(N):

        # proceed only if in-degree of current node is 0 and
        # current node is not processed yet
        if graph.indegree[v] == 0 and not discovered[v]:

            # for every adjacent vertex u of v,
            # reduce in-degree of u by 1
            for u in graph.adjList[v]:
                graph.indegree[u] = graph.indegree[u] - 1

            # include current node in the path
            # and mark it as discovered
            path.append(v)
            discovered[v] = True

            # recur
            findAllTopologicalOrders(graph, path, discovered, N, results)

            # backtrack: reset in-degree
            # information for the current node
            for u in graph.adjList[v]:
                graph.indegree[u] = graph.indegree[u] + 1

            # backtrack: remove current node from the path and
            # mark it as undiscovered
            path.pop()
            discovered[v] = False

    # print the topological order if
    # all vertices are included in the path
    if len(path) == N:
        # print(path)
        results.append(path[:])


# Print all topological orderings of a given DAG
def getAllTopologicalOrders(graph):
    # get number of nodes in the graph
    N = len(graph.adjList)

    # create an auxiliary space to keep track of whether vertex is discovered
    discovered = [False] * N

    # list to store the topological order
    path = []

    # find all topological ordering and print them
    results = []
    findAllTopologicalOrders(graph, path, discovered, N, results)

    return results


# Driver code
if __name__ == '__main__':
    # List of graph edges as per above diagram
    # edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]
    # edges = [(1, 2), (2, 3), (3, 1)]
    edges = [(0, 1), (1, 2), (2, 1), (2, 3), (3, 4)]

    # print("All Topological sorts")

    # Number of nodes in the graph
    N = 5

    # create a graph from edges
    graph = Graph(edges, N)

    # print all topological ordering of the graph
    results = getAllTopologicalOrders(graph)
    print(results)

# This code is contributed by Priyadarshini Kumari