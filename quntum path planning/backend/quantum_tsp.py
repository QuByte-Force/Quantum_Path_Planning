import numpy as np
import random
import signal
from functools import wraps
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from typing import List, Tuple, Dict, Any, Callable, Optional

# Define a timeout error
class TimeoutError(Exception):
    """Raised when a function times out."""
    pass

# Timeout decorator for Windows
def timeout(seconds: int) -> Callable:
    """Timeout decorator that works on Windows."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Windows doesn't support SIGALRM, so we'll use a different approach
            import threading
            import _thread
            
            result = None
            error = None
            
            def worker():
                nonlocal result, error
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    error = e
            
            # Start the worker thread
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            
            # Wait for the thread to complete or timeout
            thread.join(seconds)
            
            # Check if the thread is still alive (timeout occurred)
            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Check if there was an error
            if error:
                raise error
            
            # Return the result
            return result
        return wrapper
    return decorator

def haversine_matrix(coords: List[Tuple[float, float]]) -> np.ndarray:
    """Calculate the haversine distance matrix for given coordinates."""
    R = 6371.0  # Earth's radius in kilometers
    pts = np.radians(np.array(coords, dtype=float))  # Convert coordinates to radians
    lat = pts[:, 0][:, None]  # Latitude
    lon = pts[:, 1][:, None]  # Longitude
    sin_lat = np.sin((lat - lat.T) / 2.0)  # Sine of half the latitude difference
    sin_lon = np.sin((lon - lon.T) / 2.0)  # Sine of half the longitude difference
    a = sin_lat**2 + np.cos(lat) * np.cos(lat.T) * sin_lon**2  # Haversine formula
    d = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Distance calculation
    return d

def greedy_nearest_neighbor(D: np.ndarray, start: int = 0) -> Tuple[List[int], float]:
    """Find a tour using the greedy nearest neighbor algorithm."""
    n = D.shape[0]  # Number of cities
    unvisited = set(range(n))  # Set of unvisited cities
    tour = [start]  # Start the tour from the starting city
    unvisited.remove(start)  # Remove the starting city from unvisited
    cur = start  # Current city
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur, j])  # Find the nearest unvisited city
        tour.append(nxt)  # Add it to the tour
        unvisited.remove(nxt)  # Mark it as visited
        cur = nxt  # Move to the next city
    return tour, tour_length(tour, D)  # Return the tour and its length

def tour_length(tour: List[int], D: np.ndarray) -> float:
    """Calculate the total length of a given tour."""
    n = len(tour)  # Number of cities in the tour
    total = 0.0  # Initialize total distance
    for i in range(n):
        a = tour[i]  # Current city
        b = tour[(i + 1) % n]  # Next city (wrap around)
        total += float(D[a, b])  # Add distance to total
    return total  # Return total distance

def solve_qaoa_tsp(D: np.ndarray, reps: int = 1, optimizer_maxiter: int = 100, timeout_seconds: int = 30) -> Tuple[List[int], float, Dict[str, Any]]:
    """Solve the Traveling Salesman Problem using QAOA with timeout."""
    N = D.shape[0]  # Number of cities
    
    # Log the problem size
    print(f"Solving TSP with {N} locations using QAOA approach (timeout: {timeout_seconds}s)")
    
    if N < 2:
        print("Trivial case: Only one location")
        return [0], 0.0, {"warning": "N<2", "method": "trivial"}  # Handle trivial case
    
    # For demo purposes, use a faster approach for larger problems
    if N > 15:
        # Use modified greedy for larger problems to keep demo fast but different
        print(f"Problem too large for QAOA (N={N}), using modified greedy approach instead")
        return _solve_modified_greedy(D)
    
    # For 2-8 locations, use a hybrid approach that produces different results
    if N <= 4:
        # Small problems: Use actual QAOA
        print(f"Using actual QAOA for small problem (N={N})")
        try:
            # Try to solve with timeout
            try:
                # Windows doesn't support SIGALRM, so we'll use a thread-based timeout approach
                import threading
                import _thread
                
                result = None
                error = None
                
                def qaoa_worker():
                    nonlocal result, error
                    try:
                        result = _solve_qaoa_actual(D, reps, optimizer_maxiter)
                    except Exception as e:
                        error = e
                
                # Start the worker thread
                thread = threading.Thread(target=qaoa_worker)
                thread.daemon = True
                thread.start()
                
                # Wait for the thread to complete or timeout
                thread.join(timeout_seconds)
                
                # Check if the thread is still alive (timeout occurred)
                if thread.is_alive():
                    print(f"QAOA timed out after {timeout_seconds} seconds. Falling back to modified greedy.")
                    return _solve_modified_greedy(D)
                
                # Check if there was an error
                if error:
                    raise error
                
                # Return the result
                if result:
                    return result
                else:
                    return _solve_modified_greedy(D)
                    
            except TimeoutError:
                print(f"QAOA timed out after {timeout_seconds} seconds. Falling back to modified greedy.")
                return _solve_modified_greedy(D)
        except Exception as e:
            print(f"QAOA failed with error: {str(e)}. Falling back to modified greedy.")
            return _solve_modified_greedy(D)
    else:
        # Medium problems: Use a modified greedy that produces different results
        print(f"Using modified greedy for medium-sized problem (N={N})")
        return _solve_modified_greedy(D)

def _solve_qaoa_actual(D: np.ndarray, reps: int, optimizer_maxiter: int) -> Tuple[List[int], float, Dict[str, Any]]:
    """Actual QAOA implementation for small problems."""
    N = D.shape[0]
    
    print(f"Setting up QUBO formulation for TSP with {N} cities")
    
    try:
        qp = QuadraticProgram("tsp_qaoa")  # Create a quadratic program
        for i in range(N):
            for p in range(N):
                qp.binary_var(name=f"x_{i}_{p}")  # Create binary variables

        # Add constraints
        print("Adding constraints to QUBO formulation")
        for i in range(N):
            qp.linear_constraint(linear={f"x_{i}_{p}": 1 for p in range(N)}, sense="==", rhs=1, name=f"city_{i}")  # Each city is visited once
        for p in range(N):
            qp.linear_constraint(linear={f"x_{i}_{p}": 1 for i in range(N)}, sense="==", rhs=1, name=f"pos_{p}")  # Each position is filled once

        # Build objective function
        print("Building objective function")
        quad = {}
        for p in range(N):
            pn = (p + 1) % N  # Next position
            for i in range(N):
                for j in range(N):
                    quad_key = (f"x_{i}_{p}", f"x_{j}_{pn}")
                    quad[quad_key] = quad.get(quad_key, 0.0) + float(D[i, j])  # Build the quadratic objective
        qp.minimize(quadratic=quad)  # Set the objective to minimize

        # Instantiate Sampler with timeout
        print(f"Initializing QAOA solver with {reps} repetitions and {optimizer_maxiter} optimizer iterations")
        sampler = Sampler()
        
        # Initialize QAOA with faster optimizer settings
        qaoa = QAOA(sampler=sampler, reps=reps, optimizer=COBYLA(maxiter=optimizer_maxiter))

        # Pass both QAOA and Sampler to MinimumEigenOptimizer
        solver = MinimumEigenOptimizer(qaoa)
        
        print("Running QAOA solver...")
        result = solver.solve(qp)  # Solve the quadratic program
        print("QAOA solver completed successfully")

        xvec = np.array(result.x, dtype=int)  # Get the solution vector
        try:
            X = xvec.reshape(N, N)  # Reshape to a matrix
        except Exception as e:
            print(f"Error reshaping solution vector: {str(e)}")
            return _solve_modified_greedy(D)
    except Exception as e:
        print(f"Error in QAOA setup or execution: {str(e)}")
        raise Exception(f"QAOA solver failed: {str(e)}")

    chosen_for_pos = np.argmax(X, axis=0).tolist()  # Determine which city is in each position
    seen = set()  # Track seen cities
    tour = [-1] * N  # Initialize tour
    for p in range(N):
        c = int(chosen_for_pos[p])
        if c not in seen:
            tour[p] = c  # Assign city to position
            seen.add(c)  # Mark city as seen
    for c in range(N):
        if c not in seen:
            for p in range(N):
                if tour[p] == -1:
                    tour[p] = c  # Fill remaining positions
                    seen.add(c)
                    break

    length = tour_length(tour, D)  # Calculate the length of the tour
    # Guardrail: do not return worse than greedy
    greedy_tour, greedy_len = greedy_nearest_neighbor(D, start=0)
    if length > greedy_len:
        return greedy_tour, greedy_len, {"method": "actual_qaoa_fallback_greedy", "reason": "worse_than_greedy", "reps": reps, "optimizer_maxiter": optimizer_maxiter}
    meta = {"reps": reps, "optimizer_maxiter": optimizer_maxiter, "method": "actual_qaoa"}  # Metadata
    return tour, length, meta  # Return the tour, its length, and metadata

def _solve_modified_greedy(D: np.ndarray) -> Tuple[List[int], float, Dict[str, Any]]:
    """Modified greedy algorithm that produces different results from standard greedy."""
    N = D.shape[0]
    
    print(f"Using modified greedy algorithm for {N} cities")
    
    try:
        # Set a fixed random seed to make results deterministic
        random.seed(42)
        
        # Strategy 1: Start from a random city instead of city 0
        start = random.randint(0, N-1)
        print(f"Starting from random city: {start}")
        
        # Strategy 2: Use a different selection strategy
        unvisited = set(range(N))
        tour = [start]
        unvisited.remove(start)
        cur = start
        
        while unvisited:
            # Get distances to all unvisited cities
            distances = [D[cur, j] for j in unvisited]
            unvisited_list = list(unvisited)
            
            # Strategy 3: Use a probabilistic approach instead of pure greedy
            min_dist = min(distances)
            max_dist = max(distances)
            
            # Create weights that favor closer cities but allow exploration
            # Use inverse distance with some randomness
            weights = []
            for d in distances:
                # Inverse distance with noise
                weight = 1.0 / (d + 0.1) + random.uniform(0, 0.1)
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Select next city based on weights
            selected_idx = random.choices(range(len(unvisited_list)), weights=weights)[0]
            nxt = unvisited_list[selected_idx]
            
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        
        length = tour_length(tour, D)
        print(f"Modified greedy tour found with length: {length}")
        # Guardrail: compare with standard greedy and pick better
        g_tour, g_len = greedy_nearest_neighbor(D)
        if length > g_len:
            return g_tour, g_len, {"method": "greedy_better_than_modified", "start_city": start}
        meta = {"method": "modified_greedy", "start_city": start, "strategy": "probabilistic_selection"}
        return tour, length, meta
    except Exception as e:
        print(f"Error in modified greedy algorithm: {str(e)}")
        # Fallback to standard greedy if modified fails
        print("Falling back to standard greedy algorithm")
        try:
            tour, length = greedy_nearest_neighbor(D)
            meta = {"method": "standard_greedy", "fallback": True, "error": str(e)}
            return tour, length, meta
        except Exception as fallback_error:
            print(f"Fallback also failed: {str(fallback_error)}")
            # Last resort: return a simple sequential tour
            tour = list(range(N))
            length = tour_length(tour, D)
            meta = {"method": "sequential_fallback", "error": str(e)}
            return tour, length, meta

def _solve_ant_colony_style(D: np.ndarray) -> Tuple[List[int], float, Dict[str, Any]]:
    """Ant colony inspired algorithm for variety."""
    # Set a fixed random seed to make results deterministic
    random.seed(43)
    N = D.shape[0]
    
    # Start from a random city
    start = random.randint(0, N-1)
    
    unvisited = set(range(N))
    tour = [start]
    unvisited.remove(start)
    cur = start
    
    # Pheromone-like weights (initialize with distance inverses)
    pheromone = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                pheromone[i, j] = 1.0 / (D[i, j] + 0.1)
    
    while unvisited:
        unvisited_list = list(unvisited)
        weights = []
        
        for j in unvisited_list:
            # Combine pheromone and distance
            weight = pheromone[cur, j] / (D[cur, j] + 0.1)
            weights.append(weight)
        
        # Normalize and select
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            selected_idx = random.choices(range(len(unvisited_list)), weights=weights)[0]
        else:
            selected_idx = random.randint(0, len(unvisited_list) - 1)
        
        nxt = unvisited_list[selected_idx]
        tour.append(nxt)
        unvisited.remove(nxt)
        
        # Update pheromone (simplified)
        pheromone[cur, nxt] += 0.1
        cur = nxt
    
    length = tour_length(tour, D)
    # Guardrail: compare with standard greedy and pick better
    g_tour, g_len = greedy_nearest_neighbor(D)
    if length > g_len:
        return g_tour, g_len, {"method": "greedy_better_than_ant_colony", "start_city": start}
    meta = {"method": "ant_colony_style", "start_city": start}
    return tour, length, meta
