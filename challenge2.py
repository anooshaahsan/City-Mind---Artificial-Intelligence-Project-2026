import heapq



#makes sure edge (a,b) and (b,a) are stored the same way
#so we don't accidentally treat the same road as two different roads
def _undirected_edge(a, b):
    return (a, b) if a <= b else (b, a)


#union-find data structure helps Kruskal's MST track
#which nodes are already connected to each other
#so we never create a cycle when adding roads
class _DSU:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


#scans the graph to find the first hospital and first ambulancedepot
#these are the two locations that MUST have two independent road paths between them
def _find_primary_positions(graph):
    primary_hospital = None
    primary_depot = None
    for pos, node in graph.nodes.items():
        if primary_hospital is None and node.location_type == "Hospital":
            primary_hospital = pos
        if primary_depot is None and node.location_type == "AmbulanceDepot":
            primary_depot = pos
        if primary_hospital is not None and primary_depot is not None:
            break
    return primary_hospital, primary_depot



# Converts a set of edges into a proper adjacency dictionary taake we can see konse edges are connected to konse
#used to trace the path between hospital and depot inside the MST
def _build_mst_adjacency(mst_edges):
    adj = {}
    for a, b in mst_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    return adj



#returns the unique path edges between start and goal in a tree as undirected edges
def _path_edges_in_tree(tree_adj, start, goal):
    if start == goal:
        return set()
    stack = [start]
    parent = {start: None}
    while stack:
        u = stack.pop()
        if u == goal:
            break
        for v in tree_adj.get(u, []):
            if v not in parent:
                parent[v] = u
                stack.append(v)
    if goal not in parent:
        return set()
    edges = set()
    cur = goal
    while parent[cur] is not None:
        p = parent[cur]
        edges.add(_undirected_edge(p, cur))
        cur = p
    return edges


# aik cell ke saray up down left right neighbours
def _grid_neighbors(pos, graph):
    r, c = pos
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        n = (nr, nc)
        if n in graph.nodes:
            yield n



#finds chepeast path from start node to every other node in the grid 
def _ucs_full_grid(graph, start, forbidden_edges=None):

    forbidden_edges = forbidden_edges or set()
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist.get(u, float("inf")):
            continue
        for v in _grid_neighbors(u, graph):
            e = _undirected_edge(u, v)
            if e in forbidden_edges:
                continue
            w = graph.get_edge_cost(u, v)
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def _reconstruct_path_edges(prev, start, goal):
 
    if start == goal:
        return set()
    if goal not in prev:
        return set()
    edges = set()
    cur = goal
    while cur != start:
        p = prev[cur]
        edges.add(_undirected_edge(p, cur))
        cur = p
        if cur != start and cur not in prev:
            return set()
    return edges







#pehle for every pair of meaningful locations we run UCS to find the actual cheapest grid path between them and these path costs become the edge
#weights in the metric closure phir kruskal picks the cheapest subset of these connections that links all meaningful locations together 
#each chosen connection is expanded back into its actual grid roads network
#we run UCS againnot using all edges already on the MST path between hospital and depot forcing it to find a completely
#independent second route
def build_road_network(graph, grid_size):

    meaningful_nodes = [pos for pos, node in graph.nodes.items() if node.location_type != "Empty"]

    #nothing to connect
    if len(meaningful_nodes) <= 1:
        return set(), set(), set(), set(), {
            "start": None,
            "goal": None,
            "mst_cost": 0.0,
            "extra_cost": 0.0,
            "total_cost": 0.0,
            "extra_path_found": False,
            "primary_path_edges": 0,
            "mst_edges": 0,
            "extra_edges": 0,
            "built_edges": 0,
            "meaningful_nodes": len(meaningful_nodes),
        }

    #precompute shortest paths from each meaningful node 
    dists = {}
    prevs = {}
    for s in meaningful_nodes:
        dist, prev = _ucs_full_grid(graph, s)
        dists[s] = dist
        prevs[s] = prev

    #metric closure edges between meaningful nodes
    closure_edges = []
    for i in range(len(meaningful_nodes)):
        a = meaningful_nodes[i]
        da = dists[a]
        for j in range(i + 1, len(meaningful_nodes)):
            b = meaningful_nodes[j]
            w = da.get(b, float("inf"))
            if w != float("inf"):
                closure_edges.append((w, a, b))
    closure_edges.sort(key=lambda x: x[0])

    #kruskal over meaningful nodes
    dsu = _DSU(meaningful_nodes)
    mst_pairs = set()
    mst_roads = set()
    mst_cost = 0.0
    for w, a, b in closure_edges:
        if dsu.union(a, b):
            mst_pairs.add(_undirected_edge(a, b))
            mst_cost += w
            path_edges = _reconstruct_path_edges(prevs[a], a, b)
            if not path_edges:
                path_edges = _reconstruct_path_edges(prevs[b], b, a)
            mst_roads |= path_edges
            if len(mst_pairs) == len(meaningful_nodes) - 1:
                break

    #adding a cheapest independent backup route 
    start, goal = _find_primary_positions(graph)
    extra_edges = set()
    backup_route = set()
    extra_cost = 0.0
    extra_path_found = False

    if start is not None and goal is not None and start != goal:
        mst_adj = _build_mst_adjacency(mst_roads)
        primary_path_edges = _path_edges_in_tree(mst_adj, start, goal)

        dist = {start: 0.0}
        prev = {}  
        pq = [(0.0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if d != dist.get(u, float("inf")):
                continue
            if u == goal:
                break

                e = _undirected_edge(u, v)
            for v in _grid_neighbors(u, graph):
                e = _undirected_edge(u, v)
                if e in primary_path_edges:
                    continue

              
                if e in mst_roads:
                    w = 0.0
                    is_new = False
                else:
                    w = graph.get_edge_cost(u, v)
                    is_new = True

                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = (u, is_new, e)
                    heapq.heappush(pq, (nd, v))

        if goal in dist:
            extra_path_found = True
            extra_cost = dist[goal]
            cur = goal
            while cur != start:
                p, is_new, e = prev[cur]
                backup_route.add(e)
                if is_new:
                    extra_edges.add(e)
                cur = p

    backup_roads = set(extra_edges)
    built_roads = mst_roads | backup_roads

    stats = {
        "start": start,
        "goal": goal,
        "mst_cost": mst_cost,
        "extra_cost": extra_cost,
        "total_cost": mst_cost + extra_cost,
        "extra_path_found": extra_path_found,
        "primary_path_edges": len(primary_path_edges) if start is not None and goal is not None and start != goal else 0,
        "mst_edges": len(mst_roads),
        "extra_edges": len(extra_edges),
        "built_edges": len(built_roads),
        "meaningful_nodes": len(meaningful_nodes),
    }

    return mst_roads, backup_roads, built_roads, backup_route, stats

