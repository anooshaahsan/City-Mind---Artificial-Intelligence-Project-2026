import pygame
import sys
from challenge1 import run_csp
from challenge2 import build_road_network
from challenge3 import run_ga, apply_ambulance_placement, bfs_distances_from
from challenge4 import setup_challenge4, FloodSystem, MedicalTeam
from challenge5 import run_challenge5

# Window and Grid size initialisation which will remain 
# constant throughout the simulation
GRID_SIZE    = 20
CELL_SIZE    = 32
PANEL_WIDTH  = 240
INFO_HEIGHT  = 50

WINDOW_WIDTH  = GRID_SIZE * CELL_SIZE + PANEL_WIDTH
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + INFO_HEIGHT

WHITE     = (255, 255, 255)
BLACK     = (0,   0,   0)
GRAY      = (215, 215, 225)
DARK_GRAY = (35,  35,  45)

# each location/node type is assigned a unique colour
# draw_grid() directly renders these colours on the grid
LOCATION_COLORS = {
    'Empty':          (250, 250, 252),
    'Residential':    (255, 248, 180),
    'Hospital':       (255, 120, 120),
    'School':         (100, 160, 255),
    'Industrial':     (170, 170, 175),
    'PowerPlant':     (255, 175,  60),
    'AmbulanceDepot': (90,  210, 110),
}

#each button is assigned a colour
BTN_DEFAULT = (210, 228, 255)
BTN_ACTIVE  = (70,  130, 240)
BTN_HOVER   = (175, 205, 255)
BTN_START   = (160, 240, 160)
BTN_RESET   = (255, 175, 175)
BTN_TEXT    = (25,  25,  40)
ACCENT_BLUE = (100, 120, 200)

#medical team moves one cell every 1.25s 
TEAM_MOVE_INTERVAL_MS = 1250

# How long to pause after each challenge is done 
STEP_DELAY_MS = 2500

# Right-panel layout
BUTTONS_TOP_Y = 460  # y-position where the buttons stack begin


#  NODE class represents each cell in the grid
class Node:
    def __init__(self, row, col):
        self.row                = row
        self.col                = col
        self.location_type      = 'Empty' # set by C1
        self.population_density = 0.0  # used for risk calculation in C5
        self.risk_index         = 1.0 #set by C5
        self.accessibility      = True


#  CITY GRAPH represents the entire layout
class CityGraph:
    def __init__(self):
        self.nodes               = {}
        self.edges               = {}
        self.blocked             = set()
        self.built_roads         = set()
        self.backup_roads        = set()
        self.backup_route        = set()
        self.ambulance_positions = []
        self.team_path           = []
        self.risk_map            = {}

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self.nodes[(r, c)] = Node(r, c)

    # Registers every 4-directional adjacent pair as a road edge.
    # Residential-adjacent roads cost 0.8 (easier access); all others 1.0.
    def init_edges(self):
        self.edges.clear()
        self.built_roads.clear()
        self.backup_roads.clear()
        self.backup_route.clear()

        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                for dr, dc in directions:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                        a, b = (r,c), (nr,nc)
                        cost = (0.8 if (self.nodes[a].location_type == 'Residential'
                                        or self.nodes[b].location_type == 'Residential')
                                else 1.0)
                        self.edges[(a, b)] = cost
                        self.edges[(b, a)] = cost

    #get edge cost considers base cost, risk index fom C5, and blocked status
    def get_edge_cost(self, a, b):
        if (a, b) in self.blocked:
            return float('inf')
        base_cost = self.edges.get((a, b), float('inf'))
        return base_cost * self.nodes[b].risk_index

    def block_road(self, a, b):
        self.blocked.add((a, b))
        self.blocked.add((b, a))

    def unblock_road(self, a, b):
        self.blocked.discard((a, b))
        self.blocked.discard((b, a))



#  BUTTON class for right panel UI
class Button:
    def __init__(self, x, y, w, h, label,
                 color=BTN_DEFAULT, active_color=BTN_ACTIVE):
        self.rect         = pygame.Rect(x, y, w, h)
        self.label        = label
        self.color        = color
        self.active_color = active_color
        self.active       = False

    def draw(self, surface, font, mouse_pos):
        col = (self.active_color if self.active
               else BTN_HOVER if self.rect.collidepoint(mouse_pos)
               else self.color)
        pygame.draw.rect(surface, col, self.rect, border_radius=8)
        txt = font.render(self.label, True, BTN_TEXT)
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def is_clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.rect.collidepoint(event.pos))



#  SIMULATION class manages the overall progression of challenges
#   Steps 1-5 : structured setup (one challenge per step)
#   Steps 6-20: after completing C4, re running simulation in order
#   C5 - C3 - C4 until 20 steps are reached
class Simulation:
    def __init__(self, graph):
        self.graph            = graph
        self.step             = 0
        self.max_steps        = 20
        self.running          = False
        self.ended            = False
        self.event_log        = ["System initialised. Ready to start."]
        self.overlay          = None
        # Holds references to FloodSystem and MedicalTeam from challenge4 so it can be reset and re run as needed
        self.flood_system     = FloodSystem()
        self.team             = MedicalTeam()
        self.c4_active        = False
        self.last_move_time   = 0
        self._pause_time      = 0
        # tracks when the last auto-step completed so we can delay the next one
        self.last_step_time   = 0
        self.step_ready       = True   

        # Floods/reroutes remain independent and do NOT consume a step.
        # wave_queue puts C5 - C3 - C4 in list/queue after each rescue wave.
        self.wave_queue = []  
        self.team_movement_enabled = False
        self.log_scroll = 0  # 0 = pinned to bottom, >0 = scrolled up

    def end_simulation(self, msg="Simulation complete."):
        self.ended = True
        self.running = False
        self.c4_active = False
        self.team_movement_enabled = False
        self.flood_system.reset()
        self.log_unstepped(msg)

    def log(self, msg):
        self.event_log.append(f"Step {self.step}: {msg}")
        if len(self.event_log) > 100:
            self.event_log.pop(0)
        if self.log_scroll == 0:
            self.log_scroll = 0

    def log_unstepped(self, msg):
        self.event_log.append(msg)
        if len(self.event_log) > 100:
            self.event_log.pop(0)
        if self.log_scroll == 0:
            self.log_scroll = 0

    # executes one step at a time

    #   IMPORTANT ordering 
    #   Step 1: Challenge 1 generates city layout (run once per simulation)
    #   Step 2: Challenge 2 build roads (run once per simulation)
    #   Step 3: Challenge 5 updates risk weights (cost multipliers) 
    #   Step 4: Challenge 3 places ambulances (can be re-run as weights shift)
    #   Step 5: Challenge 4 routing uses the risk-weighted edge costs
    def next_step(self):
        if self.ended or self.step >= self.max_steps:
            self.end_simulation() #end simulation if steps exceed 20
            return

        if self.c4_active and not self.team.done:
            return

    #if not start with step 1(challenge 1)
        self.step += 1
        self.step_ready     = False
        self.last_step_time = pygame.time.get_ticks()

        if self.step == 1:
            self.log("Running CSP to generate city layout...")
            conflict = run_csp(self.graph, GRID_SIZE)
            self.graph.init_edges()
            if conflict:
                self.log(f"CSP conflict: {conflict}")
            else:
                self.log("City layout generated successfully.")
            return

        if self.step == 2:
            self.log("Optimising road network (Kruskal MST + UCS backup)...")
            mst_roads, backup_roads, built_roads, backup_route, stats = \
                build_road_network(self.graph, GRID_SIZE)
            self.graph.built_roads  = built_roads
            self.graph.backup_roads = backup_roads
            self.graph.backup_route = backup_route
            if stats["start"] is None or stats["goal"] is None:
                self.log("Roads built (MST). Hospital/Depot not found for redundancy.")
            elif stats["extra_path_found"]:
                self.log(
                    f"Road network built. MST edges: {stats['mst_edges']}, "
                    f"backup edges: {stats['extra_edges']}. "
                    f"Total cost: {stats['total_cost']:.1f}"
                )
            else:
                self.log("Road network built (MST). No independent backup path found.")
            self.flood_system.activate()
            self.log("Flood system activated.")
            return
        

        def debug_challenge5(graph, risk_map):
            print("challenge 5 dekhlo ke sai chalra")
    
            # show risk index before and after for sample nodes
            type_risk_summary = {}
            for pos, node in graph.nodes.items():
                if node.location_type == 'Empty':
                    continue
                t = node.location_type
                if t not in type_risk_summary:
                    type_risk_summary[t] = {'High': 0, 'Medium': 0, 'Low': 0}
                label = risk_map.get(pos, 'Low')
                type_risk_summary[t][label] += 1
    
            print("Risk distribution by location type:")
            for loc_type, counts in sorted(type_risk_summary.items()):
                print(f"  {loc_type:15s} -> High: {counts['High']:3d}, "
                      f"Medium: {counts['Medium']:3d}, Low: {counts['Low']:3d}")
    
            # show a few specific node examples
            print("\nSample node risk assignments:")
            shown = 0
            for pos, node in graph.nodes.items():
                if node.location_type in ('Hospital', 'Industrial', 'Residential') and shown < 9:
                    print(f"  {str(pos):12s} type={node.location_type:15s} "
                          f"density={node.population_density:.2f} "
                          f"risk_index={node.risk_index} "
                          f"label={risk_map.get(pos,'?')}")
                    shown += 1
    
            # verify risk_index values are actually different from default
            unique_risks = set(node.risk_index for node in graph.nodes.values())
            print(f"\nUnique risk_index values in graph: {sorted(unique_risks)}")
            print("")

        if self.step == 3:
            self.log("Running Crime Risk Prediction (K-Means + Decision Tree)...")
            risk_map, accuracy, c5_logs, stats = run_challenge5(self.graph, GRID_SIZE)
            self.graph.risk_map = risk_map
            for line in c5_logs:
                self.log(line)
            self.log(
                f"C5 done. k={stats.get('k','?')}, "
                f"accuracy={stats.get('accuracy',0)*100:.1f}%, "
                f"High={stats.get('high',0)}, "
                f"Medium={stats.get('medium',0)}, "
                f"Low={stats.get('low',0)}."
            )
            debug_challenge5(self.graph,self.graph.risk_map)
            return

        if self.step == 4:
            self.log("Running Genetic Algorithm for ambulance placement...")
            positions, worst_dist, ga_logs = run_ga(self.graph)
            apply_ambulance_placement(self.graph, positions)
            for line in ga_logs:
                self.log(line)
            if positions:
                self.log(f"Ambulances placed. Worst-case distance: {worst_dist} hops.")
            else:
                self.log("GA Warning: no positions returned.")
            return

        if self.step == 5:
            self.log("Setting up emergency routing (A* Search)...")
            logs = setup_challenge4(self.graph, self.team)
            for line in logs:
                self.log(line)
            if self.team.position:
                self.graph.team_path = [self.team.position] + self.team.path
            self.c4_active              = True
            self.team_movement_enabled  = False
            self.last_move_time         = pygame.time.get_ticks()
            return

    def _run_c5(self, label_prefix="Running"):
        self.log(f"{label_prefix} Crime Risk Prediction (K-Means + Decision Tree)...")
        risk_map, accuracy, c5_logs, stats = run_challenge5(self.graph, GRID_SIZE)
        self.graph.risk_map = risk_map
        for line in c5_logs:
            self.log(line)
        self.log(
            f"C5 done. k={stats.get('k','?')}, "
            f"accuracy={stats.get('accuracy',0)*100:.1f}%, "
            f"High={stats.get('high',0)}, "
            f"Medium={stats.get('medium',0)}, "
            f"Low={stats.get('low',0)}."
        )
        # After risk update, re plan current route to reflect updated costs.
        if self.c4_active and self.team.position and not self.team.done:
            reroute_msg = self.team._reroute(self.graph, reason="risk update")
            if reroute_msg:
                self.log_unstepped(reroute_msg)
            self.graph.team_path = (
                [self.team.position] + self.team.path
                if self.team.position else []
            )

    def _run_ga(self, label_prefix="Running"):
        self.log(f"{label_prefix} Genetic Algorithm for ambulance placement...")
        positions, worst_dist, ga_logs = run_ga(self.graph)
        apply_ambulance_placement(self.graph, positions)
        for line in ga_logs:
            self.log(line)
        if positions:
            self.log(f"Ambulances positioned. Worst-case distance: {worst_dist} hops.")
        else:
            self.log("GA Warning: no positions returned.")

    def _setup_c4(self, label_prefix="Setting up"):
        self.log(f"{label_prefix} emergency routing (A* Search)...")
        self.team = MedicalTeam()
        logs = setup_challenge4(self.graph, self.team)
        for line in logs:
            self.log(line)
        if self.team.position:
            self.graph.team_path = [self.team.position] + self.team.path
        self.c4_active             = True
        self.team_movement_enabled = False  

    def _queue_new_wave(self):
        # jb ek wave complete hojaye add ek aur C5 -> C3 -> C4 
        if self.step < self.max_steps:
            self.wave_queue.extend(['c5', 'ga', 'c4'])

    def _consume_step_action(self, action):
        if self.ended or self.step >= self.max_steps:
            return
        self.step += 1
        if action == 'c5':
            self._run_c5(label_prefix="Re-running" if self.step > 5 else "Running")
        elif action == 'ga':
            self._run_ga(label_prefix="Re-running" if self.step > 5 else "Running")
        elif action == 'c4':
            self._setup_c4(label_prefix="Re-running")
        if self.step >= self.max_steps:
            self.end_simulation("STEP 20 reached. Simulation ended.")

    def _move_until_next_rescue(self, safety_limit=2000):
        #used in Manual mode (agr next step button use krna ho to move rescue team cell by cell)
        if not self.c4_active or self.team.done:
            return False

        self.team_movement_enabled = True
        moved = 0
        while moved < safety_limit and not self.team.done:
            move_log, rescued = self.team.move_one_step(self.graph)
            moved += 1

            if not rescued:
                if ("re-route" in move_log.lower() or "reroute" in move_log.lower()
                        or "flood" in move_log.lower()):
                    self.log_unstepped(move_log)
            else:
                if self.step < self.max_steps:
                    self.step += 1
                    self.log(move_log)
                else:
                    self.log_unstepped(move_log)

                self.graph.team_path = (
                    [self.team.position] + self.team.path
                    if self.team.position else []
                )

                if self.team.done:
                    self.log_unstepped("ALL CIVILIANS RESCUED. Mission complete!")
                    self.c4_active = False
                    self.team_movement_enabled = False
                    self._queue_new_wave()
                if self.step >= self.max_steps:
                    self.end_simulation("STEP 20 reached. Simulation ended.")
                return True

            self.graph.team_path = (
                [self.team.position] + self.team.path
                if self.team.position else []
            )

        return False

    def tick_team(self, now_ms):
        #used in auto mode (start button se medical team will move 1 tick at a time with delay)
        if self.ended or self.step >= self.max_steps:
            return
        if not self.c4_active or self.team.done:
            return
        if not self.team_movement_enabled:
            return
        if now_ms - self.last_move_time < TEAM_MOVE_INTERVAL_MS:
            return

        self.last_move_time = now_ms
        move_log, rescued = self.team.move_one_step(self.graph)

        if rescued:
            self.step += 1
            self.log(move_log)
            if self.team.done:
                self.log_unstepped("ALL CIVILIANS RESCUED. Mission complete!")
                self.c4_active = False
                self.team_movement_enabled = False
                self._queue_new_wave()
        else:
            # reroutes/flood-related messages in logs
            if ("re-route" in move_log.lower() or "reroute" in move_log.lower()
                    or "flood" in move_log.lower()):
                self.log_unstepped(move_log)

        self.graph.team_path = (
            [self.team.position] + self.team.path
            if self.team.position else []
        )

        if self.step >= self.max_steps:
            self.end_simulation("STEP 20 reached. Simulation ended.")

    def advance_one_step(self):

        #   If step < 5  call next_step() 
        #   IF steps exceed 5 wave_queue execute next queued challenge step
        if self.ended or self.step >= self.max_steps:
            self.end_simulation()
            return

        if self.step < 5:
            self.next_step()
            return

        # reset queue after each wave is done
        if self.wave_queue:
            action = self.wave_queue.pop(0)
            self._consume_step_action(action)
            return

        self._move_until_next_rescue()


# Neeche saray visuals related functions hain

#  DRAW BLOCKED EDGES
def draw_blocked_roads(surface, graph, icon_blocked=None):
    drawn = set()
    for a, b in graph.built_roads:
        edge_key = (min(a,b), max(a,b))
        if edge_key in drawn:
            continue
        if (a, b) in graph.blocked or (b, a) in graph.blocked:
            drawn.add(edge_key)
            r1, c1 = a
            r2, c2 = b
            x1 = c1 * CELL_SIZE + CELL_SIZE // 2
            y1 = r1 * CELL_SIZE + INFO_HEIGHT + CELL_SIZE // 2
            x2 = c2 * CELL_SIZE + CELL_SIZE // 2
            y2 = r2 * CELL_SIZE + INFO_HEIGHT + CELL_SIZE // 2
            mx = (x1 + x2) // 2
            my = (y1 + y2) // 2
            if icon_blocked:
                icon_s = max(8, CELL_SIZE // 2)
                small  = pygame.transform.smoothscale(icon_blocked, (icon_s, icon_s))
                surface.blit(small, (mx - icon_s // 2, my - icon_s // 2))
            else:
                s = 7
                pygame.draw.line(surface, (220, 0, 0), (mx-s, my-s), (mx+s, my+s), 3)
                pygame.draw.line(surface, (220, 0, 0), (mx+s, my-s), (mx-s, my+s), 3)


#  DRAW GRID
def draw_grid(surface, graph, sim, font_small, icon_ambulance=None, icon_team=None, icon_civilian=None, icon_rescued=None, icon_blocked=None):

    def draw_dashed_line(color, start_xy, end_xy, width=2, dash_len=10, gap_len=7):
        x1, y1 = start_xy
        x2, y2 = end_xy
        dx, dy = x2-x1, y2-y1
        dist   = (dx*dx + dy*dy) ** 0.5
        if dist == 0:
            return
        ux, uy = dx/dist, dy/dist
        t = 0.0
        while t < dist:
            seg_end = min(t + dash_len, dist)
            pygame.draw.line(surface, color,
                             (x1+ux*t,       y1+uy*t),
                             (x1+ux*seg_end, y1+uy*seg_end), width)
            t += dash_len + gap_len

    coverage_dist = {}
    if sim.overlay == 'coverage' and graph.ambulance_positions:
        for amb in graph.ambulance_positions:
            d_map = bfs_distances_from(amb, graph)
            for pos, d in d_map.items():
                if pos not in coverage_dist or d < coverage_dist[pos]:
                    coverage_dist[pos] = d

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            node  = graph.nodes[(r, c)]
            x     = c * CELL_SIZE
            y     = r * CELL_SIZE + INFO_HEIGHT
            pos   = (r, c)
            color = LOCATION_COLORS.get(node.location_type, WHITE)

            if sim.overlay == 'heatmap':
                risk_label = graph.risk_map.get((r, c), None)
                if risk_label == 'High':
                    color = (220, 60, 60)
                elif risk_label == 'Medium':
                    color = (255, 165, 0)
                elif risk_label == 'Low':
                    color = (100, 200, 100)
                else:
                    risk  = node.risk_index
                    color = ((220, 60,  60)  if risk >= 1.5 else
                             (255, 165,  0)  if risk >= 1.2 else
                             (100, 200, 100))

            pygame.draw.rect(surface, color, (x, y, CELL_SIZE, CELL_SIZE))

            if sim.overlay == 'coverage' and pos in coverage_dist:
                d     = coverage_dist[pos]
                alpha = max(20, 140 - d * 18)
                cov   = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                cov.fill((0, 120, 255, alpha))
                surface.blit(cov, (x, y))

            pygame.draw.rect(surface, GRAY, (x, y, CELL_SIZE, CELL_SIZE), 1)

    draw_blocked_roads(surface, graph, icon_blocked)

    # ambulance icon 
    for pos in graph.ambulance_positions:
        rr, cc = pos
        x = cc * CELL_SIZE + 2
        y = rr * CELL_SIZE + INFO_HEIGHT + 2
        s = CELL_SIZE - 4
        if icon_ambulance:
            surface.blit(icon_ambulance, (x, y))
        else:
            # agr image nhi load hoti tou standard shape draw krke dikha do
            pygame.draw.rect(surface, (255, 255, 255), (x, y, s, s), border_radius=3)
            pygame.draw.rect(surface, (200, 0, 0), (x, y, s, s), 2, border_radius=3)
            bar = s // 5
            mid = s // 2
            pygame.draw.rect(surface, (200, 0, 0), (x + bar, y + mid - bar//2, s - bar*2, bar))
            pygame.draw.rect(surface, (200, 0, 0), (x + mid - bar//2, y + bar, bar, s - bar*2))

    if sim.overlay == 'roads' and graph.built_roads:
        def center(p):
            rr, cc = p
            return (cc * CELL_SIZE + CELL_SIZE // 2,
                    rr * CELL_SIZE + INFO_HEIGHT + CELL_SIZE // 2)

        extra = graph.backup_roads or set()
        route = graph.backup_route or set()

        for a, b in graph.built_roads - extra:
            draw_dashed_line((70, 70, 70), center(a), center(b), width=2)
        for a, b in extra:
            draw_dashed_line((155, 85, 255), center(a), center(b), width=2)
        for a, b in route:
            draw_dashed_line((0, 200, 255), center(a), center(b),
                             width=2, dash_len=14, gap_len=5)

    team_path = graph.team_path
    if team_path and len(team_path) > 1:
        for i in range(len(team_path) - 1):
            rr1, cc1 = team_path[i]
            rr2, cc2 = team_path[i+1]
            x1 = cc1 * CELL_SIZE + CELL_SIZE // 2
            y1 = rr1 * CELL_SIZE + INFO_HEIGHT + CELL_SIZE // 2
            x2 = cc2 * CELL_SIZE + CELL_SIZE // 2
            y2 = rr2 * CELL_SIZE + INFO_HEIGHT + CELL_SIZE // 2
            pygame.draw.line(surface, (255, 220, 0), (x1, y1), (x2, y2), 2)

    # medical team icon
    if team_path:
        tr, tc = team_path[0]
        x = tc * CELL_SIZE + 2
        y = tr * CELL_SIZE + INFO_HEIGHT + 2
        s = CELL_SIZE - 4
        if icon_team:
            surface.blit(icon_team, (x, y))
        else:
            # agr na load ho tou normal shap draw krlo
            pygame.draw.rect(surface, (30, 80, 220), (x, y, s, s), border_radius=4)
            pygame.draw.rect(surface, (200, 220, 255), (x, y, s, s), 2, border_radius=4)
            bar = max(2, s // 5)
            mid = s // 2
            pygame.draw.rect(surface, (255, 255, 255), (x + bar, y + mid - bar//2, s - bar*2, bar))
            pygame.draw.rect(surface, (255, 255, 255), (x + mid - bar//2, y + bar, bar, s - bar*2))

    # civilian icon
    for cv in sim.team.civilians:
        cr, cc = cv
        x = cc * CELL_SIZE + 2
        y = cr * CELL_SIZE + INFO_HEIGHT + 2
        s = CELL_SIZE - 4
        if icon_civilian:
            surface.blit(icon_civilian, (x, y))
        else:
            # agr na load ho tou standard shape draw krlo
            cx = cc * CELL_SIZE + CELL_SIZE // 2
            cy = cr * CELL_SIZE + INFO_HEIGHT + CELL_SIZE // 2
            pygame.draw.circle(surface, (255, 210, 40), (cx, cy), s // 2)
            pygame.draw.circle(surface, (160, 120, 0),  (cx, cy), s // 2, 2)

    # rescued civilians ka marker update krlo to a X 
    for rv in sim.team.rescued:
        rr, rc = rv
        rx = rc * CELL_SIZE
        ry = rr * CELL_SIZE + INFO_HEIGHT
        pygame.draw.line(surface, (150, 150, 150),
                         (rx+4,           ry+4),
                         (rx+CELL_SIZE-4, ry+CELL_SIZE-4), 2)
        pygame.draw.line(surface, (150, 150, 150),
                         (rx+CELL_SIZE-4, ry+4),
                         (rx+4,           ry+CELL_SIZE-4), 2)


def draw_legend(surface, font_title, font_small):
    x = GRID_SIZE * CELL_SIZE + 10
    y = INFO_HEIGHT + 10
    lbl = font_title.render("Legend", True, ACCENT_BLUE)
    surface.blit(lbl, (x, y))
    y += 4
    pygame.draw.line(surface, GRAY, (x, y + 16), (x + PANEL_WIDTH - 20, y + 16), 1)
    y += 24
    for name, color in LOCATION_COLORS.items():
        pygame.draw.rect(surface, color,     (x, y, 14, 14), border_radius=3)
        pygame.draw.rect(surface, DARK_GRAY, (x, y, 14, 14), 1, border_radius=3)
        surface.blit(font_small.render(name, True, DARK_GRAY), (x + 20, y))
        y += 19


def draw_event_log(surface, sim, font_title, font_small):
    x = GRID_SIZE * CELL_SIZE + 10
    y = INFO_HEIGHT + 200
    lbl = font_title.render("Event Log", True, ACCENT_BLUE)
    surface.blit(lbl, (x, y))
    y += 18
    pygame.draw.line(surface, GRAY, (x, y), (x + PANEL_WIDTH - 20, y), 1)
    y += 8
    max_chars = 34
    line_h = 13
    wrapped = []
    for entry in sim.event_log:
        words = entry.split()
        cur = ""
        for w in words:
            test = (cur + w + " ") if cur else (w + " ")
            if len(test) > max_chars:
                wrapped.append(cur.rstrip())
                cur = w + " "
            else:
                cur = test
        if cur:
            wrapped.append(cur.rstrip())

    # Scroll logic: sim.log_scroll == 0 means bottom, >0 means scrolled up.
    # Keep the log strictly above the button stack.
    bottom_limit = min(WINDOW_HEIGHT - 10, BUTTONS_TOP_Y - 10)
    available_h = max(0, bottom_limit - y)
    visible_lines = max(1, available_h // line_h)
    max_scroll = max(0, len(wrapped) - visible_lines)
    sim.log_scroll = max(0, min(sim.log_scroll, max_scroll))

    start = max(0, len(wrapped) - visible_lines - sim.log_scroll)
    end = min(len(wrapped), start + visible_lines)
    shown = wrapped[start:end]

    # Clip rendering so text cannot overlap buttons.
    clip_rect = pygame.Rect(x, y, PANEL_WIDTH - 20, available_h)
    prev_clip = surface.get_clip()
    surface.set_clip(clip_rect)

    yy = y
    for i, line in enumerate(shown):
        if yy + line_h > bottom_limit:
            break
        is_last_visible = (i == len(shown) - 1) and (sim.log_scroll == 0)
        color = ACCENT_BLUE if is_last_visible else DARK_GRAY
        surface.blit(font_small.render(line, True, color), (x, yy))
        yy += line_h

    surface.set_clip(prev_clip)


def draw_info_bar(surface, sim, font_title, font_small):
    pygame.draw.rect(surface, (245, 248, 255), (0, 0, WINDOW_WIDTH, INFO_HEIGHT))
    pygame.draw.line(surface, ACCENT_BLUE,
                     (0, INFO_HEIGHT - 2), (WINDOW_WIDTH, INFO_HEIGHT - 2), 2)

    rescued = len(sim.team.rescued)
    total   = rescued + len(sim.team.civilians)
    c4_info = f"  |  Civilians: {rescued}/{total}" if sim.c4_active else ""

    info_txt = (f"CITYMIND  |  STEP {sim.step}/{sim.max_steps}  |  "
                f"OVERLAY: {sim.overlay or 'None'}{c4_info}")
    txt = font_title.render(info_txt, True, ACCENT_BLUE)
    surface.blit(txt, (10, (INFO_HEIGHT - txt.get_height()) // 2))


#  MAIN

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("CityMind - Urban Intelligence System")
    clock = pygame.time.Clock()

    try:
        font_title = pygame.font.Font("FFF_Tusj.ttf", 16)
    except FileNotFoundError:
        font_title = pygame.font.SysFont("segoeui", 16)

    font_small = pygame.font.SysFont("segoeui", 12)

    icon_size = CELL_SIZE - 4

    def load_icon(filename):
        try:
            img = pygame.image.load(filename).convert_alpha()
            return pygame.transform.smoothscale(img, (icon_size, icon_size))
        except FileNotFoundError:
            return None   # falls back to shape drawing if file missing

    icon_ambulance = load_icon("ambulance.png")
    icon_team      = load_icon("team.png")
    icon_civilian  = load_icon("civilian.png")
    icon_blocked = load_icon("blocked.png")

    graph = CityGraph()
    sim   = Simulation(graph)

    bx = GRID_SIZE * CELL_SIZE + 10
    bw = PANEL_WIDTH - 20

    btn_start    = Button(bx, 460, bw, 32, "Start / Pause", BTN_START, (60, 180, 60))
    btn_step     = Button(bx, 500, bw, 32, "Next Step")
    btn_reset    = Button(bx, 540, bw, 32, "Reset",         BTN_RESET, (200, 60, 60))
    btn_roads    = Button(bx, 588, bw, 28, "Show Roads")
    btn_coverage = Button(bx, 622, bw, 28, "Show Coverage")
    btn_heatmap  = Button(bx, 656, bw, 28, "Show Heatmap")

    overlay_buttons = {
        'roads':    btn_roads,
        'coverage': btn_coverage,
        'heatmap':  btn_heatmap,
    }
    buttons = [btn_start, btn_step, btn_reset,
               btn_roads, btn_coverage, btn_heatmap]

    while True:
        now_ms    = pygame.time.get_ticks()
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # scroll kr sakte for event logs
            if event.type == pygame.MOUSEWHEEL:
                if mouse_pos[0] >= GRID_SIZE * CELL_SIZE:
                    sim.log_scroll = max(0, sim.log_scroll + event.y * 3)

            for btn in buttons:
                if btn.is_clicked(event):

                    if btn is btn_start:
                        was_running = sim.running
                        sim.running = not sim.running
                        if not was_running:
                          # resume after pausing will smoothly continue 
                            sim.last_move_time = now_ms
                            sim.last_step_time = now_ms - STEP_DELAY_MS
                        else:
                            pause_dur = now_ms - getattr(sim, '_pause_time', now_ms)
                            sim.last_move_time += pause_dur
                            
                            sim.last_step_time = now_ms - STEP_DELAY_MS
                            sim.step_ready     = True

                    elif btn is btn_step:
                        # manual mode mein next step krne se simulation will advance by one 
                        sim.last_step_time = now_ms
                        sim.advance_one_step()

                    elif btn is btn_reset:
                        sim.step           = 0
                        sim.running        = False
                        sim.c4_active      = False
                        sim.team_movement_enabled = False
                        sim.wave_queue = []
                        sim.ended          = False
                        sim.log_scroll     = 0
                        sim.last_move_time = 0
                        sim._pause_time    = 0
                        sim.last_step_time = 0
                        sim.step_ready     = True
                        sim.event_log      = ["System reset."]
                        sim.flood_system.reset()
                        sim.team = MedicalTeam()
                        for pos, node in graph.nodes.items():
                            node.accessibility      = True
                            node.location_type      = 'Empty'
                            node.population_density = 0.0
                            node.risk_index         = 1.0
                        graph.blocked.clear()
                        graph.edges.clear()
                        graph.built_roads.clear()
                        graph.backup_roads.clear()
                        graph.backup_route.clear()
                        graph.ambulance_positions = []
                        graph.team_path           = []
                        graph.risk_map            = {}

                    else:
                        for key, b in overlay_buttons.items():
                            if btn is b:
                                sim.overlay = None if sim.overlay == key else key
                                b.active    = (sim.overlay == key)
                                for k2, b2 in overlay_buttons.items():
                                    if k2 != key:
                                        b2.active = False

        if sim.running and not sim.ended:
            delay_ok = (now_ms - sim.last_step_time) >= STEP_DELAY_MS

            if sim.step < 5:
                if delay_ok:
                    sim.last_step_time = now_ms
                    sim.advance_one_step()
            else:
                if sim.wave_queue and delay_ok:
                    sim.last_step_time = now_ms
                    sim.advance_one_step()
                if sim.c4_active and not sim.team.done:
                    sim.team_movement_enabled = True
                sim.tick_team(now_ms)

        # flood and unblock is independent of the whole simulation process
        # Flood system runs even in manual mode (paused)
        if sim.flood_system.active and not sim.ended and sim.step < sim.max_steps:
            flooded_edge, flood_msg = sim.flood_system.check_and_flood(graph)
            if flooded_edge:
                sim.log_unstepped(flood_msg)
                reroute_msg = sim.team.check_path_affected(flooded_edge, graph)
                if reroute_msg:
                    sim.log_unstepped(reroute_msg)
                graph.team_path = (
                    [sim.team.position] + sim.team.path
                    if sim.team.position else []
                )
            unblocked = sim.flood_system.check_and_unblock(graph)
            for _, msg in unblocked:
                sim.log_unstepped(msg)

        # agr step limit lg jaye stop everything 
        if not sim.ended and sim.step >= sim.max_steps:
            sim.end_simulation("STEP 20 reached. Simulation ended.")

        screen.fill(WHITE)
        draw_info_bar(screen, sim, font_title, font_small)
        draw_grid(screen, graph, sim, font_small, icon_ambulance, icon_team, icon_civilian, icon_rescued=None, icon_blocked=icon_blocked)

        pygame.draw.rect(screen, (248, 249, 255),
                         (GRID_SIZE * CELL_SIZE, 0, PANEL_WIDTH, WINDOW_HEIGHT))
        pygame.draw.line(screen, GRAY,
                         (GRID_SIZE * CELL_SIZE, 0),
                         (GRID_SIZE * CELL_SIZE, WINDOW_HEIGHT), 1)

        draw_legend(screen, font_title, font_small)
        draw_event_log(screen, sim, font_title, font_small)

        for btn in buttons:
            btn.draw(screen, font_small, mouse_pos)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()