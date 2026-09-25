# City Mind - AI-Powered City Planning Simulation

City Mind is an interactive city-planning simulation built with Python and Pygame. It models a 20x20 grid-based city and solves five real-world-style planning problems using classic Artificial Intelligence techniques — from constraint satisfaction to genetic algorithms to unsupervised clustering.

The project was built to explore how different AI search and optimization methods can be applied to a single connected problem domain: designing, connecting, and protecting a city.

## Features / Challenges

The simulation is split into five modules, each tackling a distinct AI problem on the same city grid:

1. **Zone Placement (Constraint Satisfaction Problem)** — Assigns city locations (hospitals, schools, industrial zones, power plants, ambulance depots, residential areas) across the grid while satisfying proximity and spacing constraints, using backtracking/CSP logic with BFS-based checks.

2. **Road Network Design (Minimum Spanning Tree)** — Connects all city zones with roads using Kruskal's MST algorithm and a Union-Find (Disjoint Set Union) structure, while guaranteeing redundant, independent paths between critical points such as hospitals and ambulance depots.

3. **Ambulance Placement Optimization (Genetic Algorithm)** — Uses a genetic algorithm (population-based search with tournament selection, crossover, and mutation) to find near-optimal placements for a limited number of ambulances, minimizing average distance to residential areas.

4. **Emergency Routing with Dynamic Obstacles (A* Search)** — Simulates randomly occurring road flooding/blockages and uses A* pathfinding (with Manhattan-distance heuristic) to reroute medical teams to their destinations in real time as the city changes.

5. **Risk Zone Clustering (K-Means)** — Extracts features for each city zone (population density, proximity to industry, location type, road accessibility) and applies K-Means clustering (with an elbow-method search over k) to classify areas into risk levels.

All five systems run inside a single live, visual simulation with an interactive control panel — you can watch zones get placed, roads get built, ambulances reposition, floods disrupt routes, and risk zones update, all on one grid.

## Tech Stack

- **Python 3.9+**
- **Pygame** — for the grid rendering, animation, and UI panel
- Core algorithms implemented from scratch (no external ML/AI libraries) — CSP/backtracking, Kruskal's MST + Union-Find, Genetic Algorithm, A* search, K-Means clustering

## How to Run

1. Make sure you have Python 3.9+ installed.
2. Install the one dependency:
```bash
   pip install pygame
```
3. Run the main file:
```bash
   python AI_Project_2026.py
```
4. Use the on-screen control panel buttons to step through each challenge (zone placement → roads → ambulance placement → emergency routing → risk clustering).

## Project Structure

```
AI_Project_2026.py   # Main simulation loop, grid rendering, UI, and control panel
challenge1.py         # Zone placement (Constraint Satisfaction Problem)
challenge2.py         # Road network generation (Kruskal's MST + Union-Find)
challenge3.py         # Ambulance placement (Genetic Algorithm)
challenge4.py         # Emergency routing under dynamic flooding (A* search)
challenge5.py         # Risk zone classification (K-Means clustering)
ambulance.png, blocked.png, civilian.png, team.png   # Sprite assets
FFF_Tusj.ttf          # UI font
```

## What This Project Demonstrates

This project was built to apply and compare multiple core AI paradigms within one coherent simulation:
- **Search & pathfinding** (A*)
- **Constraint satisfaction** (backtracking/CSP)
- **Graph algorithms** (MST, Union-Find, BFS)
- **Evolutionary/optimization algorithms** (Genetic Algorithm)
- **Unsupervised machine learning** (K-Means clustering)

## Author

Built by Anoosha Ahsan as part of an AI coursework project, 2026.
