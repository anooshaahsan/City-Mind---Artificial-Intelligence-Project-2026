import heapq
import random
import time
from collections import deque

#roods flood hongay randomly every 2-5s
FLOOD_TIMER_MIN = 2   
FLOOD_TIMER_MAX =5

#calcuating straight line distance between two cells
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

#  A* search on ALL grid edges, not just built roads
def astar(graph, start, goal):
    if start == goal:
        return [start], 0.0

    open_set  = [(manhattan(start, goal), 0.0, start)]
    g_cost    = {start: 0.0}
    came_from = {}

    while open_set:
        f, g, current = heapq.heappop(open_set)

        if g > g_cost.get(current, float('inf')):
            continue

        if current == goal:
            path = []
            node = goal
            while node != start:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path, g_cost[goal]

        r, c = current
        for nr, nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            nb = (nr, nc)
            if nb not in graph.nodes:
                continue
            cost = graph.get_edge_cost(current, nb)
            if cost == float('inf'):
                continue
            tg = g + cost
            if tg < g_cost.get(nb, float('inf')):
                g_cost[nb]    = tg
                came_from[nb] = current
                heapq.heappush(open_set, (tg + manhattan(nb, goal), tg, nb))

    return [], float('inf')


#  FLOOD SYSTEM
class FloodSystem:
    def __init__(self):
        self.active          = False # by defaut flood inactive hai jb tk challenge 2 run nhi hota aur roads bn nhi jati
        self.next_flood_time = None
        self.flooded_edges   = {}

    def activate(self):
        self.active = True
        self._schedule_next()

    def _schedule_next(self):
        self.next_flood_time = time.time() + random.uniform(FLOOD_TIMER_MIN, FLOOD_TIMER_MAX)

    #picks a random built road and block it
    def check_and_flood(self, graph):
        if not self.active or not graph.built_roads:
            return None, None
        if time.time() < self.next_flood_time:
            return None, None

        # pick a random built road not already blocked
        available = [
            e for e in graph.built_roads
            if e not in graph.blocked and (e[1], e[0]) not in graph.blocked
        ]
        if not available:
            self._schedule_next()
            return None, None

        a, b = random.choice(available)
        # block only the edge do not mark nodes as inaccessible
        graph.blocked.add((a, b))
        graph.blocked.add((b, a))
        edge = (min(a,b), max(a,b))
        self.flooded_edges[edge] = time.time()
        self._schedule_next()

        return (a, b), f"Flood: road {a}↔{b} blocked."

    def check_and_unblock(self, graph):
       # Unblock roads whose flood timer has expired jo 15s hai
        if not self.active:
            return []
        now = time.time()
        done = []
        results = []
        for edge, t in self.flooded_edges.items():
            if now - t >= 15: #idhr flood timer 15s set huwa wa
                a, b = edge
                graph.blocked.discard((a, b))
                graph.blocked.discard((b, a))
                done.append(edge)
                results.append((edge, f"Road {a}↔{b} cleared."))
        for e in done:
            del self.flooded_edges[e]
        return results

    def reset(self):
        self.active          = False
        self.next_flood_time = None
        self.flooded_edges   = {}


#  MEDICAL TEAM
class MedicalTeam:
    def __init__(self):
        self.position  = None
        self.civilians = []
        self.path      = []
        self.path_cost = 0.0
        self.rescued   = []
        self.done      = False

    def initialise(self, graph, num_civilians=None):
        if num_civilians is None:
            num_civilians = random.randint(5, 8)

        logs = []

        # team starts at a Hospital or AmbulanceDepot
        start_pool = [
            pos for pos, node in graph.nodes.items()
            if node.location_type in ('Hospital', 'AmbulanceDepot')
        ]
        if not start_pool:
            start_pool = list(graph.nodes.keys())
        self.position = random.choice(start_pool)

        #civilians only on residential nodes not on team start or ambulance spots
        exclude = set(graph.ambulance_positions) | {self.position}
        civ_pool = [
            pos for pos, node in graph.nodes.items()
            if node.location_type == 'Residential' and pos not in exclude
        ]
        # fallback if not enough residential nodes
        if len(civ_pool) < num_civilians:
            civ_pool = [
                pos for pos, node in graph.nodes.items()
                if node.location_type != 'Empty' and pos not in exclude
            ]
        num_civilians = min(num_civilians, len(civ_pool))
        chosen = random.sample(civ_pool, num_civilians)

        # sort nearest ciivilian first using actual A* cost 
        def actual_dist(pos):
            _, cost = astar(graph, self.position, pos)
            return cost

        chosen.sort(key=actual_dist)
        self.civilians = chosen
        self.rescued   = []
        self.done      = False
        self.path      = []

        logs.append(
            f"C4: Team at {self.position} "
            f"({graph.nodes[self.position].location_type}). "
            f"{len(self.civilians)} civilians placed."
        )
        for i, cv in enumerate(self.civilians):
            d = actual_dist(cv)
            logs.append(f"  Civilian {i+1}: {cv}  dist={d:.1f}")
        return logs

    def plan_route_to_next(self, graph):
        if not self.civilians:
            self.done = True
            return "C4: All civilians rescued!"

        target = self.civilians[0]
        path, cost = astar(graph, self.position, target)

        if not path:
            skipped = self.civilians.pop(0)
            self._resort(graph)
            return f"C4: No path to {skipped}, skipping."

        self.path      = path[1:]
        self.path_cost = cost
        return (f"C4: Route to {target}. "
                f"{len(self.path)} steps, cost {cost:.2f}")

    def _resort(self, graph):
        #re sort remaining civilians by actual A* distance from current position
        if not self.civilians:
            return
        def d(pos):
            _, cost = astar(graph, self.position, pos)
            return cost
        self.civilians.sort(key=d)

    def move_one_step(self, graph):
        if self.done:
            return "C4: Mission complete.", False

        if not self.path:
            log = self.plan_route_to_next(graph)
            return log, False

        next_pos = self.path.pop(0)

        # blocked edge check
        edge_cost = graph.get_edge_cost(self.position, next_pos)
        if edge_cost == float('inf'):
           # re route immediately
            log = self._reroute(graph, reason="blocked step detected")
            return log, False

        self.position = next_pos

        if self.civilians and self.position == self.civilians[0]:
            rescued = self.civilians.pop(0)
            self.rescued.append(rescued)
            self.path = []
            if self.civilians:
                self._resort(graph)
                route_log = self.plan_route_to_next(graph)
                return (f"C4: Rescued civilian at {rescued}! "
                        f"{len(self.civilians)} remaining. {route_log}"), True
            else:
                self.done = True
                return f"C4: Final civilian rescued at {rescued}! Mission complete.", True

        return f"C4: Moved to {self.position}.", False

    def _reroute(self, graph, reason="flood"):
        if not self.civilians:
            return "C4: No civilians left to route to."
        target = self.civilians[0]
        path, cost = astar(graph, self.position, target)
        if not path:
            return f"C4: Re-route failed ({reason}). No path to {target}."
        self.path      = path[1:]
        self.path_cost = cost
        return (f"C4: Re-routed ({reason}) to {target}. "
                f"{len(self.path)} steps, cost {cost:.2f}")

    def check_path_affected(self, blocked_edge, graph):
        #Called after any flood event. 
        #ke current path affect huwa ke nhi, and re route agr huwa ho
        
        if not self.path or self.done:
            return None

        full = [self.position] + self.path
        for i in range(len(full) - 1):
            u, v = full[i], full[i+1]
            if graph.get_edge_cost(u, v) == float('inf'):
                return self._reroute(graph, reason="flood")

        return None


def setup_challenge4(graph, team, num_civilians=None):
    logs = team.initialise(graph, num_civilians)
    route_log = team.plan_route_to_next(graph)
    logs.append(route_log)
    return logs