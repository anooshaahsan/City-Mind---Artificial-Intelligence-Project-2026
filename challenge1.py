import random
from collections import deque
import time

#location ki types
SPECIAL_TYPES = ['Hospital', 'School', 'Industrial', 'PowerPlant', 'AmbulanceDepot']

TYPE_COUNTS = {
    'Hospital':       10,
    'School':         15,
    'Industrial':     25,
    'PowerPlant':     10,
    'AmbulanceDepot': 10,
}

RESIDENTIAL_COUNT = 150
EMPTY_COUNT       = 180


#below are some helper fucntions 
def get_neighbors(pos, grid_size):
    r, c = pos
    result = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            result.append((nr, nc))
    return result


def bfs_within_hops(start, targets, assignment, grid_size, max_hops):
    """
    Returns True if a target type exists within max_hops.
    Unassigned cells do NOT count as potential  strict check.
    """
    visited = {start}
    queue   = deque([(start, 0)])
    while queue:
        pos, hops = queue.popleft()
        if hops > max_hops:
            continue
        if pos != start and assignment.get(pos) in targets:
            return True
        if hops < max_hops:
            for n in get_neighbors(pos, grid_size):
                if n not in visited:
                    visited.add(n)
                    queue.append((n, hops + 1))
    return False


#basically to make min conflicts a bit faster below function saves ke har cell ke 2 hops pe konsi location hai, 3 pe
#konsi and so on
def precompute_hop_neighborhoods(grid_size, max_hops):
    """
    Precompute positions within N hops (grid graph hops) for each cell.
    Returns dict: pos -> list where index k contains set of positions within <=k hops (excluding pos).
    """
    all_positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    neighborhoods = {}
    for start in all_positions:
        visited = {start}
        queue = deque([(start, 0)])
        within_exact = [set() for _ in range(max_hops + 1)]
        while queue:
            pos, hops = queue.popleft()
            if hops >= max_hops:
                continue
            for n in get_neighbors(pos, grid_size):
                if n in visited:
                    continue
                visited.add(n)
                within_exact[hops + 1].add(n)
                queue.append((n, hops + 1))

        within_upto = [set() for _ in range(max_hops + 1)]
        for k in range(1, max_hops + 1):
            within_upto[k] = set(within_exact[k])
            within_upto[k].update(within_upto[k - 1])

        neighborhoods[start] = within_upto
    return neighborhoods


#mathematical validity pre-check ke like is a location placement solution mathematically even possible
def check_mathematical_validity(grid_size):
    hospitals   = TYPE_COUNTS['Hospital']
    industrials = TYPE_COUNTS['Industrial']
    powerplants = TYPE_COUNTS['PowerPlant']

    
    if RESIDENTIAL_COUNT > hospitals * 20:
        return False, (
            f"Rule Violated: Every Residential must be within 3 hops of a Hospital. "
            f"{hospitals} hospitals can cover at most ~{hospitals * 20} residential "
            f"cells but {RESIDENTIAL_COUNT} are required. "
            f"Increase Hospital count or reduce Residential count."
        )

    if powerplants > 0 and industrials == 0:
        return False, (
            "Rule Violated: PowerPlants exist but no Industrial zones are defined. "
            "Every PowerPlant must be within 2 hops of an Industrial zone."
        )

    if powerplants > industrials * 8:
        return False, (
            f"Rule Violated: Every PowerPlant must be within 2 hops of an Industrial zone. "
            f"{industrials} industrial zones cannot cover {powerplants} power plants."
        )

    return True, None


#ye check karta if constraints are being followed by a certain value being assigned to a variable
#does not consider residential and empty
def is_consistent(pos, value, assignment, grid_size):
    neighbors = get_neighbors(pos, grid_size)

    #industrial not adjacent to school or hospital
    if value == 'Industrial':
        for n in neighbors:
            if assignment.get(n) in ('School', 'Hospital'):
                return False

    if value in ('School', 'Hospital'):
        for n in neighbors:
            if assignment.get(n) == 'Industrial':
                return False

    #powerPlant within 2 hops of industrial
    if value == 'PowerPlant':
        #checking if any Industrial already placed within 2 hops
        #if none placed yet at all allow it
        any_industrial = any(v == 'Industrial' for v in assignment.values())
        if any_industrial:
            if not bfs_within_hops(pos, {'Industrial'}, assignment, grid_size, 2):
                return False
    
    return True

#below is AC-3 preprocessing functions
def compatible(val_i, val_j):
    if val_i == 'Industrial' and val_j in ('School', 'Hospital'):
        return False
    if val_j == 'Industrial' and val_i in ('School', 'Hospital'):
        return False
    return True


def revise(domains, xi, xj):
    revised  = False
    to_remove = []
    for val_i in domains[xi]:
        if not any(compatible(val_i, val_j) for val_j in domains[xj]):
            to_remove.append(val_i)
            revised = True
    for val in to_remove:
        domains[xi].remove(val)
    return revised


def ac3(domains, grid_size):
    queue = deque()
    for pos in domains:
        for n in get_neighbors(pos, grid_size):
            if n in domains:
                queue.append((pos, n))
    while queue:
        xi, xj = queue.popleft()
        if revise(domains, xi, xj):
            if len(domains[xi]) == 0:
                return False
            for xk in get_neighbors(xi, grid_size):
                if xk in domains and xk != xj:
                    queue.append((xk, xi))
    return True


#below is forward checking function
def forward_check(pos, value, domains, assignment, grid_size):
    pruned = {}
    for n in get_neighbors(pos, grid_size):
        if n in domains and n not in assignment:
            pruned[n] = []
            for nval in list(domains[n]):
                if not compatible(value, nval):
                    domains[n].remove(nval)
                    pruned[n].append(nval)
            if len(domains[n]) == 0:
                return None
    return pruned


def undo_pruning(pruned, domains):
    for pos, values in pruned.items():
        domains[pos].extend(values)


#backtracking function
#backtracking and forward checking both dont use heuristics as they lowk lowk made it slower 
def get_remaining_counts(assignment):
    #how many more of each type still needs to be placed
    placed = {}
    for v in assignment.values():
        placed[v] = placed.get(v, 0) + 1
    remaining = {}
    for t, count in TYPE_COUNTS.items():
        remaining[t] = count - placed.get(t, 0)
    return remaining


def all_types_placed(assignment):
    remaining = get_remaining_counts(assignment)
    return all(v <= 0 for v in remaining.values())


def backtrack(assignment, domains, grid_size,startTime):

    if time.time() - startTime > 5:
        return None

    if all_types_placed(assignment):
        return assignment

    #pick next unassigned position from domains
    unassigned = [p for p in domains if p not in assignment]
    if not unassigned:
        return None

    pos = unassigned[0]

    #get remaining counts to know which types still needed
    remaining = get_remaining_counts(assignment)
    needed_types = [t for t, cnt in remaining.items() if cnt > 0]

    #only trying values still needed
    values_to_try = [v for v in domains[pos] if v in needed_types]
    if not values_to_try:
        values_to_try = list(domains[pos])

    for value in values_to_try:
        remaining_for_type = remaining.get(value, 0)
        if remaining_for_type <= 0:
            continue

        if is_consistent(pos, value, assignment, grid_size):
            assignment[pos] = value
            domain_backup   = {p: list(d) for p, d in domains.items()}

            pruned = forward_check(pos, value, domains, assignment, grid_size)
            if pruned is not None:
                result = backtrack(assignment, domains, grid_size,startTime)
                if result is not None:
                    return result

            del assignment[pos]
            for p, d in domain_backup.items():
                domains[p] = d

    return None


#csp se sab assign hogya ab we fill residential and empty we fill following constraints though
#ye isliye kiya taake csp smartly kaam karay ziada slow na karde
def fill_residential_and_empty(assignment, grid_size):
    all_positions = [(r, c)
                     for r in range(grid_size)
                     for c in range(grid_size)]
    unassigned = [p for p in all_positions if p not in assignment]
    random.shuffle(unassigned)

    res_placed = 0
    for pos in unassigned:
        if res_placed >= RESIDENTIAL_COUNT:
            break
        if bfs_within_hops(pos, {'Hospital'}, assignment, grid_size, 3):
            assignment[pos] = 'Residential'
            res_placed += 1

    for pos in all_positions:
        if pos not in assignment:
            assignment[pos] = 'Empty'

    return assignment


#checks if poori assignment valid hai ke nai
def full_assignment_valid(assignment, grid_size,startTime):

    if time.time() - startTime > 5:
        return None

    for pos, val in assignment.items():
        neighbors = get_neighbors(pos, grid_size)

        if val == 'Industrial':
            for n in neighbors:
                if assignment.get(n) in ('School', 'Hospital'):
                    return False, (
                        f"Rule Violated: Industrial at {pos} is adjacent to "
                        f"{assignment[n]} at {n}. Industrial zones cannot be "
                        f"placed next to Schools or Hospitals."
                    )

        if val == 'Residential':
            if not bfs_within_hops(pos, {'Hospital'}, assignment, grid_size, 3):
                return False, (
                    f"Rule Violated: Residential at {pos} has no Hospital "
                    f"within 3 hops."
                )

        if val == 'PowerPlant':
            if not bfs_within_hops(pos, {'Industrial'}, assignment, grid_size, 2):
                return False, (
                    f"Rule Violated: PowerPlant at {pos} has no Industrial "
                    f"zone within 2 hops."
                )

    return True, None


#min conflicts code
def count_violations_for(pos, value, assignment, grid_size):
    violations = 0
    temp       = dict(assignment)
    temp[pos]  = value
    neighbors  = get_neighbors(pos, grid_size)

    if value == 'Industrial':
        for n in neighbors:
            if temp.get(n) in ('School', 'Hospital'):
                violations += 1

    if value in ('School', 'Hospital'):
        for n in neighbors:
            if temp.get(n) == 'Industrial':
                violations += 1

    if value == 'Residential':
        if not bfs_within_hops(pos, {'Hospital'}, temp, grid_size, 3):
            violations += 1

    if value == 'PowerPlant':
        if not bfs_within_hops(pos, {'Industrial'}, temp, grid_size, 2):
            violations += 1

    return violations


def get_violated_variables(assignment, grid_size):
    return [
        pos for pos, val in assignment.items()
        if count_violations_for(pos, val, assignment, grid_size) > 0
    ]

#minconflicts fallback that ALWAYS assigns the full requested counts
#(including Residential and Empty)and then minimizes constraint violations and
#uses swap moves so type counts stay exact.
def minimum_conflicts_solver(grid_size, max_steps=25000, max_time_s=2.5, candidate_swaps=30):

    all_positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]

    #building a full random assignment with exact counts
    required_total = sum(TYPE_COUNTS.values()) + RESIDENTIAL_COUNT + EMPTY_COUNT
    if required_total != grid_size * grid_size:
        #adjusting empty to fill the grid while keeping other counts exact
        empty_count = grid_size * grid_size - (sum(TYPE_COUNTS.values()) + RESIDENTIAL_COUNT)
        if empty_count < 0:
            empty_count = 0
    else:
        empty_count = EMPTY_COUNT

    type_pool = []
    for t, count in TYPE_COUNTS.items():
        type_pool.extend([t] * count)
    type_pool.extend(['Residential'] * RESIDENTIAL_COUNT)
    type_pool.extend(['Empty'] * empty_count)

    #for safety
    if len(type_pool) < grid_size * grid_size:
        type_pool.extend(['Empty'] * (grid_size * grid_size - len(type_pool)))
    type_pool = type_pool[: grid_size * grid_size]

    random.shuffle(type_pool)
    assignment = {pos: typ for pos, typ in zip(all_positions, type_pool)}


    hop_n = precompute_hop_neighborhoods(grid_size, 3)

    def count_violations_fast(pos, value, assign):
        violations = 0
        neighbors = get_neighbors(pos, grid_size)

        #industrial not adjacent to school/hospital
        if value == 'Industrial':
            for n in neighbors:
                if assign.get(n) in ('School', 'Hospital'):
                    violations += 1

        if value in ('School', 'Hospital'):
            for n in neighbors:
                if assign.get(n) == 'Industrial':
                    violations += 1

        #residential must have hospital within 3 hops
        if value == 'Residential':
            if not any(assign.get(p) == 'Hospital' for p in hop_n[pos][3]):
                violations += 1

        #powerPlant must have industrial within 2 hops
        if value == 'PowerPlant':
            if not any(assign.get(p) == 'Industrial' for p in hop_n[pos][2]):
                violations += 1

        return violations

    def compute_violated_set(assign):
        return {p for p, v in assign.items() if count_violations_fast(p, v, assign) > 0}

    def total_violations(assign):
        return sum(count_violations_fast(p, v, assign) for p, v in assign.items())

    violated_set = compute_violated_set(assignment)
    best_assignment = dict(assignment)
    best_total = total_violations(assignment)

    def affected_positions(a, b):
        #constraints depend up to 3 hops so bas only those can change
        return {a, b} | hop_n[a][3] | hop_n[b][3]

    def refresh_violations(pos_set):
        for p in pos_set:
            if count_violations_fast(p, assignment[p], assignment) > 0:
                violated_set.add(p)
            else:
                violated_set.discard(p)

    start = time.time()

    #swap-based minconflicts search
    for step in range(max_steps):
        if time.time() - start > max_time_s:
            break

        if not violated_set:
            return assignment, 0

        pos_a = random.choice(list(violated_set))
        val_a = assignment[pos_a]

        #sample of candidate swap partners ofdifferent type
        candidates = []
        tries = 0
        while len(candidates) < candidate_swaps and tries < candidate_swaps * 6:
            tries += 1
            pos_b = random.choice(all_positions)
            if pos_b == pos_a:
                continue
            if assignment[pos_b] == val_a:
                continue
            candidates.append(pos_b)

        best_delta = 0
        best_pos_b = None
        best_aff = None

        for pos_b in candidates:
            val_b = assignment[pos_b]
            aff = affected_positions(pos_a, pos_b)

            before = 0
            for p in aff:
                before += count_violations_fast(p, assignment[p], assignment)

            #swap temporarily
            assignment[pos_a], assignment[pos_b] = val_b, val_a
            after = 0
            for p in aff:
                after += count_violations_fast(p, assignment[p], assignment)
            assignment[pos_a], assignment[pos_b] = val_a, val_b

            delta = before - after  #positive mtlb improvement
            if delta > best_delta:
                best_delta = delta
                best_pos_b = pos_b
                best_aff = aff

        if best_pos_b is not None and best_delta > 0:

            assignment[pos_a], assignment[best_pos_b] = assignment[best_pos_b], assignment[pos_a]
            refresh_violations(best_aff)

        else:
            #kabhi kabhi random swap to escape local minima
            if random.random() < 0.08:
                pos_b = random.choice(all_positions)
                if pos_b != pos_a and assignment[pos_b] != val_a:
                    aff = affected_positions(pos_a, pos_b)
                    assignment[pos_a], assignment[pos_b] = assignment[pos_b], assignment[pos_a]
                    refresh_violations(aff)

        #track best seen which is important when no perfect solution exists
        if step % 50 == 0:
            tv = total_violations(assignment)
            if tv < best_total:
                best_total = tv
                best_assignment = dict(assignment)

    return best_assignment, best_total



#below function assignment ko shared city grapg pe apply kardeta
def apply_assignment_to_graph(assignment, graph):
    density_map = {
        'Residential':    (80,  150),
        'Hospital':       (30,   60),
        'School':         (40,   80),
        'Industrial':     (20,   50),
        'PowerPlant':     (10,   30),
        'AmbulanceDepot': (10,   25),
        'Empty':          (0,     0),
    }
    for pos, loc_type in assignment.items():
        graph.nodes[pos].location_type = loc_type
        lo, hi = density_map.get(loc_type, (0, 0))
        graph.nodes[pos].population_density = (
            random.uniform(lo, hi) if hi > 0 else 0.0
        )











#pehle we check mathematical validiy wali cheez
#if valid then we backtrack wala csp on types except residential and empty
#if invalid ya if backtracking failed we go to min conflicts
def run_csp(graph, grid_size):

    print("Checking mathematical validity...")
    is_valid, reason = check_mathematical_validity(grid_size)

    if not is_valid:
        print(f"Mathematical check failed: {reason}")
        print("Running Minimum Conflicts...")
        assignment, violations_remaining = minimum_conflicts_solver(grid_size)
        apply_assignment_to_graph(assignment, graph)
        return f"{reason} | Minimum Conflicts: {violations_remaining} violations"


    all_positions = [(r, c)
                     for r in range(grid_size)
                     for c in range(grid_size)]
    random.shuffle(all_positions)

    total_special = sum(TYPE_COUNTS.values())  
   


    candidate_positions = all_positions[: total_special * 4]

    domains = {pos: list(SPECIAL_TYPES) for pos in candidate_positions}

    print("Running AC-3 preprocessing...")
    ac3(domains, grid_size)

    print("Running Backtracking on special types (no heuristics)...")
    assignment = {}

    backtrack_starttime = time.time()
    result     = backtrack(assignment, domains, grid_size,backtrack_starttime)

    if result is not None:
        print("Backtracking succeeded. Filling Residential and Empty...")
        fill_residential_and_empty(result, grid_size)
        assignment_starttime = time.time()
        valid, msg = full_assignment_valid(result, grid_size,assignment_starttime)
        if valid:
            print("All constraints satisfied.")
            apply_assignment_to_graph(result, graph)
            return None
        else:
            print(f"Post-fill validation failed: {msg}")
            print("Falling back to Minimum Conflicts...")
    else:
        print("Backtracking failed. Falling back to Minimum Conflicts...")

    assignment, violations_remaining = minimum_conflicts_solver(grid_size)
    apply_assignment_to_graph(assignment, graph)

    return f"Minimum Conflicts: {violations_remaining} violations"