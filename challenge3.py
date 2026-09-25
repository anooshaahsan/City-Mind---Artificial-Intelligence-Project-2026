import random
import heapq
from collections import deque

VALID_STATION_TYPES = {'Hospital', 'AmbulanceDepot'} #jinpe ambulance place ho sakti hai

POPULATION_SIZE   = 60 #number of solutions of combinations of ambulances 
NUM_GENERATIONS   = 300 #GA will run 300 times to find best solution
TOURNAMENT_SIZE   = 5 #number of individuals in each tournament
CROSSOVER_RATE    = 0.8 #80% chance ke dou parent will swap genes
MUTATION_RATE     = 0.15 #15% chance ke ambulance position in a chromosome will be swapped randomly with a different valid station
EARLY_STOP_GENS   = 50 #agr 50 generations tk fitness value improve nhi hota, stop GA and return best solution
NUM_AMBULANCES    = 3 


def get_valid_stations(graph):
    return [
        pos for pos, node in graph.nodes.items()
        if node.location_type in VALID_STATION_TYPES
    ]


def get_residential_nodes(graph):
    return [
        pos for pos, node in graph.nodes.items()
        if node.location_type == 'Residential'
    ]


def bfs_distances_from(source, graph):
    dist  = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        r, c = u
        for nr, nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            v = (nr, nc)
            if v not in graph.nodes:
                continue
            if v in dist:
                continue
            edge = (min(u,v), max(u,v))
            if edge not in graph.built_roads:
                continue
            if (u,v) in graph.blocked or (v,u) in graph.blocked:
                continue
            dist[v] = dist[u] + 1
            queue.append(v)
    return dist


def compute_fitness(chromosome, residential_nodes, bfs_cache):
    if not residential_nodes:
        return 0
    max_dist = 0
    INF = float('inf')
    for res in residential_nodes:
        closest = INF
        for station in chromosome:
            d = bfs_cache[station].get(res, INF)
            if d < closest:
                closest = d
        max_dist = max(max_dist, 9999 if closest == INF else closest)
    return max_dist


def random_chromosome(valid_stations):
    if len(valid_stations) < NUM_AMBULANCES:
        return tuple(random.choices(valid_stations, k=NUM_AMBULANCES))
    return tuple(random.sample(valid_stations, NUM_AMBULANCES))


def tournament_selection(population, fitnesses):
    contenders = random.sample(range(len(population)), min(TOURNAMENT_SIZE, len(population)))
    best = min(contenders, key=lambda i: fitnesses[i])
    return population[best]


def single_point_crossover(parent1, parent2, valid_stations):
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2
    point  = random.randint(1, NUM_AMBULANCES - 1)
    child1 = list(parent1[:point]) + list(parent2[point:])
    child2 = list(parent2[:point]) + list(parent1[point:])

    def repair(child):
        seen   = set()
        result = []
        for pos in child:
            if pos not in seen:
                seen.add(pos)
                result.append(pos)
            else:
                available = [s for s in valid_stations if s not in seen]
                replacement = random.choice(available) if available else random.choice(valid_stations)
                seen.add(replacement)
                result.append(replacement)
        return tuple(result)

    return repair(child1), repair(child2)


def mutate(chromosome, valid_stations):
    chromosome = list(chromosome)
    for i in range(NUM_AMBULANCES):
        if random.random() < MUTATION_RATE:
            alternatives = [s for s in valid_stations if s not in chromosome]
            chromosome[i] = random.choice(alternatives) if alternatives else random.choice(valid_stations)
    return tuple(chromosome)


def run_ga(graph):
    log_lines = []

    valid_stations    = get_valid_stations(graph)
    residential_nodes = get_residential_nodes(graph)

    log_lines.append(f"GA: {len(valid_stations)} valid stations, "
                     f"{len(residential_nodes)} residential nodes.")

    if len(valid_stations) < NUM_AMBULANCES:
        msg = (f"GA Warning: only {len(valid_stations)} valid stations found "
               f"(need {NUM_AMBULANCES}). Using all available.")
        log_lines.append(msg)
        if len(valid_stations) == 0:
            log_lines.append("GA Error: no valid stations. Skipping placement.")
            return [], 9999, log_lines

    log_lines.append("GA: Pre-computing BFS distances from all stations...")
    bfs_cache = {}
    for station in valid_stations:
        bfs_cache[station] = bfs_distances_from(station, graph)

    population       = [random_chromosome(valid_stations) for _ in range(POPULATION_SIZE)]
    best_chromosome  = None
    best_fitness     = float('inf')
    no_improve_count = 0

    for gen in range(NUM_GENERATIONS):

        # evaluate fitness
        improved_this_gen = False
        fitnesses = []
        for chrom in population:
            f = compute_fitness(chrom, residential_nodes, bfs_cache)
            fitnesses.append(f)
            if f < best_fitness:
                best_fitness      = f
                best_chromosome   = chrom
                improved_this_gen = True

        # update no-improvement counter
        if improved_this_gen:
            no_improve_count = 0
        else:
            no_improve_count += 1

        # early stopping
        if no_improve_count >= EARLY_STOP_GENS:
            log_lines.append(f"GA: Early stop at generation {gen} "
                             f"(no improvement for {EARLY_STOP_GENS} gens).")
            break

        # elitism kro so carry best directly to next generation
        new_population = [best_chromosome]

        while len(new_population) < POPULATION_SIZE:
            parent1        = tournament_selection(population, fitnesses)
            parent2        = tournament_selection(population, fitnesses)
            child1, child2 = single_point_crossover(parent1, parent2, valid_stations)
            child1         = mutate(child1, valid_stations)
            child2         = mutate(child2, valid_stations)
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    if best_chromosome is None:
        best_chromosome = population[0]
        best_fitness    = compute_fitness(best_chromosome, residential_nodes, bfs_cache)

    best_positions = list(best_chromosome)
    log_lines.append(
        f"GA Complete. Best placement: {best_positions}. "
        f"Worst-case response distance: {best_fitness} hops."
    )
    return best_positions, best_fitness, log_lines


def apply_ambulance_placement(graph, positions):
    graph.ambulance_positions = list(positions)