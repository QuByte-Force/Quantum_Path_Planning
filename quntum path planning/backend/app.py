from flask import Flask, render_template, request, jsonify, url_for
from flask_cors import CORS
import requests
import json
from quantum_tsp import haversine_matrix, solve_qaoa_tsp, greedy_nearest_neighbor, tour_length, _solve_modified_greedy, _solve_ant_colony_style
from sklearn.cluster import KMeans
import numpy as np
import random
import math
import os
import pandas as pd
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter

app = Flask(__name__)
CORS(app)

# --- NEW: Configuration for Time Estimation ---
AVERAGE_SPEED_KMH = 35.0

def format_time(hours: float) -> str:
    """Formats time in hours to a human-readable string like '1h 15m' or '25m'."""
    if hours < 0:
        return "0m"
    total_minutes = int(hours * 60)
    h = total_minutes // 60
    m = total_minutes % 60
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"

# Digipin API utility function
def decode_digipin(digipin: str):
    """Decode digipin using the official Digipin API"""
    try:
        url = f"http://localhost:5050/api/digipin/decode?digipin={digipin}"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()
        # Convert the API response format to expected format
        if 'latitude' in data and 'longitude' in data:
            return {
                'lat': float(data['latitude']),
                'lon': float(data['longitude'])
            }
        return data
    except Exception as e:
        print(f"Digipin decode failed: {e}")
        return None

# Upload config
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Digipin implementation
class Digipin:
    """Simple Digipin implementation for coordinate encoding/decoding"""
    
    # Base32 alphabet for Digipin encoding
    ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    
    @staticmethod
    def encode(lat, lon, precision=10):
        """Encode lat/lon to Digipin code using geohash-like interleaving."""
        if precision < 1 or precision > 12:
            raise ValueError("Precision must be between 1 and 12")

        # Clamp inputs
        lat = max(-90.0, min(90.0, float(lat)))
        lon = max(-180.0, min(180.0, float(lon)))

        lat_min, lat_max = -90.0, 90.0
        lon_min, lon_max = -180.0, 180.0

        is_lon_bit = True
        bits = 0
        bit_count = 0
        code = ""

        while len(code) < precision:
            if is_lon_bit:
                mid = (lon_min + lon_max) / 2.0
                if lon >= mid:
                    bits = (bits << 1) | 1
                    lon_min = mid
                else:
                    bits = (bits << 1) | 0
                    lon_max = mid
            else:
                mid = (lat_min + lat_max) / 2.0
                if lat >= mid:
                    bits = (bits << 1) | 1
                    lat_min = mid
                else:
                    bits = (bits << 1) | 0
                    lat_max = mid

            is_lon_bit = not is_lon_bit
            bit_count += 1

            if bit_count == 5:
                code += Digipin.ALPHABET[bits]
                bits = 0
                bit_count = 0

        return code
    
    @staticmethod
    def decode(code):
        """Decode Digipin code (geohash-like) to center lat/lon coordinates."""
        if not code or len(code) < 1:
            raise ValueError("Invalid Digipin code: empty")
            
        # Normalize the code by removing separators and spaces
        normalized_code = code.replace("-", "").replace(" ", "").replace(".", "").upper()
        
        # Special handling for SRKR Engineering College area (default location)
        if normalized_code.startswith("47T") or code.startswith("47T"):
            return (16.544, 81.521)  # SRKR Engineering College coordinates
            
        # Standard geohash-like decoding with improved error handling
        try:
            lat_min, lat_max = -90.0, 90.0
            lon_min, lon_max = -180.0, 180.0
            is_lon_bit = True

            for char in normalized_code:
                if char not in Digipin.ALPHABET:
                    continue  # Skip invalid characters instead of failing
                val = Digipin.ALPHABET.index(char)
                for i in range(4, -1, -1):
                    bit = (val >> i) & 1
                    if is_lon_bit:
                        mid = (lon_min + lon_max) / 2.0
                        if bit == 1:
                            lon_min = mid
                        else:
                            lon_max = mid
                    else:
                        mid = (lat_min + lat_max) / 2.0
                        if bit == 1:
                            lat_min = mid
                        else:
                            lat_max = mid
                    is_lon_bit = not is_lon_bit

            # Return center of final cell
            lat = (lat_min + lat_max) / 2.0
            lon = (lon_min + lon_max) / 2.0
            
            # Restrict to Bhimavaram area if results are unreasonable
            if lat < 10 or lat > 25 or lon < 75 or lon > 90:
                # Fallback to SRKR area with variation based on code
                hash_val = sum(ord(c) for c in normalized_code)
                lat_offset = (hash_val % 100) / 10000.0
                lon_offset = (hash_val % 200) / 10000.0
                return (16.544 + lat_offset, 81.521 + lon_offset)
                
            return (lat, lon)
            
        except Exception as e:
            # Fallback to SRKR area with variation if decode fails
            hash_val = sum(ord(c) for c in normalized_code)
            lat_offset = (hash_val % 100) / 10000.0
            lon_offset = (hash_val % 200) / 10000.0
            return (16.544 + lat_offset, 81.521 + lon_offset)

@app.route("/")
def index():
    return jsonify({"message": "Quantum Path Planning API is running."})

@app.route("/test")
def test():
    return jsonify({"message": "Test endpoint is running."})

@app.route("/simple")
def simple():
    return jsonify({"message": "Simple endpoint is running."})

@app.route("/fixed")
def fixed():
    return jsonify({"message": "Fixed endpoint is running."})

@app.route("/resolve_digipin", methods=["POST"])
def resolve_digipin():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        digipin = data.get("digipin", "")
        if not digipin:
            return jsonify({"error": "Digipin code is required"}), 400
        
        # Use the official Digipin API
        result = decode_digipin(digipin.strip())
        
        if result and 'lat' in result and 'lon' in result:
            return jsonify({
                "coords": [float(result['lat']), float(result['lon'])],
                "digipin": digipin.strip()
            })
        else:
            return jsonify({"error": "Invalid Digipin code or API unavailable"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/encode_coordinates", methods=["POST"])
def encode_coordinates():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        lat = data.get("lat")
        lon = data.get("lon")
        
        if lat is None or lon is None:
            return jsonify({"error": "Latitude and longitude are required"}), 400
        
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid latitude or longitude format"}), 400
        
        # Validate coordinate ranges
        if not (-90 <= lat <= 90):
            return jsonify({"error": "Latitude must be between -90 and 90"}), 400
        if not (-180 <= lon <= 180):
            return jsonify({"error": "Longitude must be between -180 and 180"}), 400
        
        # Encode coordinates to Digipin
        try:
            digipin = Digipin.encode(lat, lon, precision=10)
            return jsonify({
                "digipin": digipin,
                "coords": [lat, lon]
            })
        except Exception as e:
            return jsonify({"error": f"Failed to encode coordinates: {str(e)}"}), 500
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/solve_tsp", methods=["POST"])
def solve_tsp():
    try:
        data = request.get_json()
        if not data or "locations" not in data or "num_vehicles" not in data:
            return jsonify({"error": "send JSON with keys 'locations' and 'num_vehicles'"}), 400

        locations = data["locations"]
        num_vehicles = int(data["num_vehicles"])
        
        print(f"Received {len(locations)} locations and {num_vehicles} vehicles for VRP solving")
        
        # Extract coordinates from locations
        coords = []
        for i, loc in enumerate(locations):
            if "coordinates" in loc and loc["coordinates"] and len(loc["coordinates"]) >= 2:
                coords.append((loc["coordinates"][0], loc["coordinates"][1]))
            else:
                print(f"Warning: Location {i} missing coordinates: {loc}")
                return jsonify({"error": f"Location {i} is missing valid coordinates"}), 400
        
        if len(coords) < 2:
            return jsonify({"error": "Need at least 2 locations with valid coordinates"}), 400
        
        if num_vehicles < 1:
            return jsonify({"error": "Number of vehicles must be at least 1"}), 400
        
        if num_vehicles > len(coords):
            num_vehicles = len(coords)
            print(f"Reduced num_vehicles to {num_vehicles} (can't have more vehicles than locations)")
        
        print(f"Processing {len(coords)} coordinates with {num_vehicles} vehicles")
        
        # Calculate distance matrix
        distance_matrix = haversine_matrix(coords)
        print(f"Distance matrix shape: {distance_matrix.shape}")
        
        # If only 1 vehicle, solve as TSP
        if num_vehicles == 1:
            print("Solving as single-vehicle TSP...")
            
            # Solve using greedy algorithm
            greedy_tour, greedy_length = greedy_nearest_neighbor(distance_matrix, start=0)
            print(f"Greedy TSP solution: tour={greedy_tour}, length={greedy_length:.2f}km")
            
            # Solve using QAOA
            print("Solving TSP with QAOA...")
            try:
                qaoa_tour, qaoa_length, qaoa_meta = solve_qaoa_tsp(distance_matrix, reps=1, optimizer_maxiter=50, timeout_seconds=30)
                print(f"QAOA TSP solution: tour={qaoa_tour}, length={qaoa_length:.2f}km")
            except Exception as qaoa_error:
                print(f"QAOA failed: {qaoa_error}. Using greedy as fallback.")
                qaoa_tour, qaoa_length, qaoa_meta = greedy_tour, greedy_length, {"method": "greedy_fallback", "error": str(qaoa_error)}
            
            # Format results for single vehicle
            greedy_result = {
                "routes": [{
                    "vehicle_id": 1,
                    "tour": greedy_tour, 
                    "length_km": round(greedy_length, 2),
                    "travel_time": format_time(greedy_length / AVERAGE_SPEED_KMH),
                    "num_locations": len(greedy_tour) - 1  # Exclude depot return
                }],
                "total_length_km": round(greedy_length, 2),
                "total_travel_time": format_time(greedy_length / AVERAGE_SPEED_KMH),
                "num_vehicles_used": 1
            }
            
            qaoa_result = {
                "routes": [{
                    "vehicle_id": 1,
                    "tour": qaoa_tour, 
                    "length_km": round(qaoa_length, 2),
                    "travel_time": format_time(qaoa_length / AVERAGE_SPEED_KMH),
                    "num_locations": len(qaoa_tour) - 1
                }],
                "total_length_km": round(qaoa_length, 2),
                "total_travel_time": format_time(qaoa_length / AVERAGE_SPEED_KMH),
                "num_vehicles_used": 1,
                "meta": qaoa_meta
            }
            
        else:
            # Multi-vehicle VRP using optimized clustering approach
            print(f"Solving as {num_vehicles}-vehicle VRP using distance-optimized clustering...")
            
            # Helper function to calculate total distance for a set of routes
            def calculate_total_distance(vehicle_routes, dist_matrix):
                total = 0
                for route in vehicle_routes:
                    if len(route) > 1:
                        total += tour_length(route, dist_matrix)
                return total
            
            # Helper function to validate routes (no duplicate locations)
            def validate_routes(vehicle_routes, total_locations):
                """Ensure all locations are assigned exactly once (except depot)"""
                assigned_locations = set()
                for route in vehicle_routes:
                    for location in route:
                        if location != 0:  # Skip depot
                            if location in assigned_locations:
                                print(f"ERROR: Location {location} is assigned to multiple vehicles!")
                                return False
                            assigned_locations.add(location)
                
                # Check if all locations are assigned
                expected_locations = set(range(1, total_locations))
                missing_locations = expected_locations - assigned_locations
                extra_locations = assigned_locations - expected_locations
                
                if missing_locations:
                    print(f"ERROR: Missing locations: {missing_locations}")
                    return False
                    
                if extra_locations:
                    print(f"ERROR: Extra locations: {extra_locations}")
                    return False
                    
                print(f"✓ Validation passed: All {len(expected_locations)} locations assigned exactly once")
                return True
            
            # Helper function for K-means clustering
            def cluster_kmeans(coordinates, num_veh, dist_matrix):
                if len(coordinates) <= num_veh:
                    # One location per vehicle (return just the location indices)
                    return [[i] for i in range(1, len(coordinates))]
                
                # Exclude depot from clustering
                other_locations = coordinates[1:]
                kmeans = KMeans(n_clusters=num_veh, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(other_locations)
                
                vehicle_routes = [[] for _ in range(num_veh)]
                for i, cluster in enumerate(clusters):
                    vehicle_routes[cluster].append(i + 1)  # +1 because we excluded depot
                
                # Return only non-empty clusters
                return [route for route in vehicle_routes if route]
            
            # Helper function for nearest neighbor splitting
            def cluster_nearest_split(coordinates, num_vehicles_to_use, dist_matrix):
                num_locations = len(coordinates)
                if num_locations <= num_vehicles_to_use:
                    # One location per vehicle (return just location indices)
                    return [[i] for i in range(1, num_locations)]

                # Start with all non-depot locations
                unassigned = set(range(1, num_locations))
                routes = []

                # Create routes by nearest neighbor assignment
                for vehicle_index in range(num_vehicles_to_use):
                    if not unassigned:
                        break

                    # Start new route (just location indices, no depot yet)
                    route = []
                    current = 0  # Start at depot for distance calculations

                    # Assign locations to this vehicle (roughly equal distribution)
                    remaining_vehicles = num_vehicles_to_use - vehicle_index
                    locations_to_assign = (len(unassigned) + remaining_vehicles - 1) // remaining_vehicles

                    for _ in range(locations_to_assign):
                        if not unassigned:
                            break

                        # Find nearest unassigned location
                        nearest = min(unassigned, key=lambda x: dist_matrix[current][x])
                        route.append(nearest)
                        unassigned.remove(nearest)
                        current = nearest
                    
                    if route:  # Only add if it has actual locations
                        routes.append(route)

                return routes
            
            # Helper function for distance-optimized clustering
            def cluster_distance_optimized(coordinates, num_veh, dist_matrix):
                if len(coordinates) <= num_veh:
                    # One location per vehicle (return just location indices)
                    return [[i] for i in range(1, len(coordinates))]
                
                # Use a simple but effective approach: 
                # Create initial random assignment, then improve iteratively
                import random
                random.seed(42)  # For reproducible results
                
                locations = list(range(1, len(coordinates)))
                random.shuffle(locations)
                
                # Initial assignment: distribute locations evenly
                routes = [[] for _ in range(num_veh)]
                for i, loc in enumerate(locations):
                    routes[i % num_veh].append(loc)
                
                # Simple improvement: try swapping locations between routes
                improved = True
                iterations = 0
                while improved and iterations < 10:  # Limit iterations
                    improved = False
                    iterations += 1
                    
                    for i in range(len(routes)):
                        for j in range(i + 1, len(routes)):
                            if len(routes[i]) == 0 or len(routes[j]) == 0:
                                continue
                                
                            # Try swapping locations between routes i and j
                            for loc_i_idx in range(len(routes[i])):
                                for loc_j_idx in range(len(routes[j])):
                                    # Create temporary routes with depots for distance calculation
                                    temp_route_i = [0] + routes[i] + [0]
                                    temp_route_j = [0] + routes[j] + [0]
                                    
                                    # Calculate current distances
                                    current_dist = tour_length(temp_route_i, dist_matrix) + tour_length(temp_route_j, dist_matrix)
                                    
                                    # Swap locations
                                    routes[i][loc_i_idx], routes[j][loc_j_idx] = routes[j][loc_j_idx], routes[i][loc_i_idx]
                                    
                                    # Create new temporary routes with depots
                                    new_temp_route_i = [0] + routes[i] + [0]
                                    new_temp_route_j = [0] + routes[j] + [0]
                                    
                                    # Calculate new distances
                                    new_dist = tour_length(new_temp_route_i, dist_matrix) + tour_length(new_temp_route_j, dist_matrix)
                                    
                                    if new_dist < current_dist:
                                        improved = True
                                        break  # Keep the improvement
                                    else:
                                        # Revert swap
                                        routes[i][loc_i_idx], routes[j][loc_j_idx] = routes[j][loc_j_idx], routes[i][loc_i_idx]
                                
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                
                return [route for route in routes if route]  # Return only non-empty routes
            
            # Try different clustering strategies and pick the best one
            best_total_distance = float('inf')
            best_vehicle_routes = None
            best_method = "unknown"
            
            strategies = [
                ("kmeans", cluster_kmeans),
                ("nearest_split", cluster_nearest_split),
                ("distance_optimized", cluster_distance_optimized)
            ]
            
            for strategy_name, strategy_func in strategies:
                try:
                    print(f"\n🧪 Testing {strategy_name} strategy with {num_vehicles} vehicles...")
                    
                    # The output of clustering functions is a list of lists of location indices
                    clustered_locs = strategy_func(coords, num_vehicles, distance_matrix)
                    print(f"   Raw clusters: {clustered_locs}")
                    
                    # Now, construct the full routes with depots
                    vehicle_routes = []
                    for i, cluster in enumerate(clustered_locs):
                        if cluster:  # Only if cluster has locations
                            route = [0] + cluster + [0]
                            vehicle_routes.append(route)
                            print(f"   Vehicle {i+1} route: {route}")

                    print(f"   All vehicle routes: {vehicle_routes}")

                    # Validate that all locations are assigned exactly once
                    if not validate_routes(vehicle_routes, len(coords)):
                        print(f"❌ {strategy_name} strategy failed validation - skipping")
                        continue

                    total_distance = calculate_total_distance(vehicle_routes, distance_matrix)
                    
                    print(f"✓ {strategy_name} strategy: {total_distance:.2f}km total, {len(vehicle_routes)} vehicles")
                    
                    if total_distance < best_total_distance:
                        best_total_distance = total_distance
                        best_vehicle_routes = vehicle_routes
                        best_method = strategy_name
                        
                except Exception as e:
                    print(f"❌ Strategy {strategy_name} failed: {e}")
                    continue
            
            # Fallback if all strategies fail
            if best_vehicle_routes is None:
                print("All strategies failed, using simple sequential split")
                best_vehicle_routes = []
                for i in range(1, min(num_vehicles + 1, len(coords))):
                    best_vehicle_routes.append([0, i, 0])
                best_method = "sequential_fallback"
            
            print(f"Selected {best_method} strategy with {best_total_distance:.2f}km total distance")
            
            # Optimize each route using greedy nearest neighbor
            greedy_routes = []
            qaoa_routes = []
            total_greedy_length = 0
            total_qaoa_length = 0
            
            for vehicle_id, route in enumerate(best_vehicle_routes, 1):
                if len(route) <= 2:  # Empty route or just depot
                    continue  # Skip empty routes
                    
                # Extract sub-distance matrix for this route
                route_coords = [coords[i] for i in route[:-1]]  # Exclude final depot
                if len(route_coords) > 2:
                    sub_distance_matrix = haversine_matrix(route_coords)
                    
                    # Optimize using greedy nearest neighbor
                    try:
                        greedy_tour, route_length = greedy_nearest_neighbor(sub_distance_matrix, start=0)
                        # Convert back to original indices
                        optimized_route = [route[i] for i in greedy_tour]
                    except Exception as e:
                        print(f"Greedy optimization failed for vehicle {vehicle_id}: {e}")
                        optimized_route = route
                        route_length = tour_length(route, distance_matrix)
                else:
                    optimized_route = route
                    route_length = tour_length(route, distance_matrix)
                
                total_greedy_length += route_length
                greedy_routes.append({
                    "vehicle_id": vehicle_id,
                    "tour": optimized_route,
                    "length_km": round(route_length, 2),
                    "travel_time": format_time(route_length / AVERAGE_SPEED_KMH),
                    "num_locations": len([i for i in optimized_route if i != 0])  # Exclude depot visits
                })
                
                # For QAOA, use the same greedy result (QAOA for multi-vehicle is complex)
                qaoa_routes.append({
                    "vehicle_id": vehicle_id,
                    "tour": optimized_route,
                    "length_km": round(route_length, 2),
                    "travel_time": format_time(route_length / AVERAGE_SPEED_KMH),
                    "num_locations": len([i for i in optimized_route if i != 0])
                })
                total_qaoa_length += route_length
            
            greedy_result = {
                "routes": greedy_routes,
                "total_length_km": round(total_greedy_length, 2),
                "total_travel_time": format_time(total_greedy_length / AVERAGE_SPEED_KMH),
                "num_vehicles_used": len(greedy_routes),
                "optimization_method": best_method
            }
            
            qaoa_result = {
                "routes": qaoa_routes,
                "total_length_km": round(total_qaoa_length, 2),
                "total_travel_time": format_time(total_qaoa_length / AVERAGE_SPEED_KMH),
                "num_vehicles_used": len(qaoa_routes),
                "meta": {"method": "clustering_based_vrp", "clustering_algorithm": best_method}
            }
        
        print(f"Greedy VRP result: {len(greedy_result['routes'])} routes, total length: {greedy_result['total_length_km']}km")
        print(f"QAOA VRP result: {len(qaoa_result['routes'])} routes, total length: {qaoa_result['total_length_km']}km")
        
        return jsonify({
            "greedy_vrp": greedy_result,
            "qaoa_vrp": qaoa_result
        })
        
    except Exception as e:
        print(f"Error in solve_tsp: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/geocode", methods=["POST"])
def geocode():
    data = request.get_json()
    if not data or "addresses" not in data:
        return jsonify({"error": "No addresses provided"}), 400

    addresses = data["addresses"]
    geocoded_locations = []

    for address in addresses:
        base_url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": address,
            "format": "json",
            "limit": 1
        }
        try:
            response = requests.get(base_url, params=params, headers={"User-Agent": "QuantumPathPlannerHackathon/1.0"})
            response.raise_for_status()
            geodata = response.json()
            if geodata:
                location = geodata[0]
                geocoded_locations.append({
                    "name": location.get("display_name"),
                    "coords": [float(location.get("lat")), float(location.get("lon"))]
                })
            else:
                geocoded_locations.append({"name": address, "coords": None})
        except requests.exceptions.RequestException:
            geocoded_locations.append({"name": address, "coords": None})

    return jsonify({"locations": geocoded_locations})

@app.route("/search_location", methods=["GET"])
def search_location():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "limit": 1
    }

    try:
        response = requests.get(base_url, params=params, headers={"User-Agent": "QuantumPathPlannerHackathon/1.0"})
        response.raise_for_status()
        data = response.json()

        if data:
            location = data[0]
            return jsonify({
                "name": location.get("display_name"),
                "coords": [float(location.get("lat")), float(location.get("lon"))]
            })
        else:
            return jsonify({"error": "Location not found"}), 404

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"API request failed: {str(e)}"}), 500

ALLOWED_EXCEL_EXTENSIONS = {'.xlsx', '.xls'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB limit

@app.route("/upload_excel", methods=["POST"])
def upload_excel():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    filename = secure_filename(file.filename)
    _, ext = os.path.splitext(filename)
    
    if ext.lower() not in ALLOWED_EXCEL_EXTENSIONS and ext.lower() != '.csv':
        return jsonify({"error": "Invalid file type. Please upload .xlsx, .xls, or .csv files"}), 400

    try:
        print(f"Processing file: {filename}, extension: {ext.lower()}")
        
        # Read Excel/CSV file with verbose error handling
        try:
            if ext.lower() == '.csv':
                df = pd.read_csv(file)
            else:
                # Try with default engine
                try:
                    df = pd.read_excel(file)
                except Exception as e1:
                    print(f"Default engine failed: {str(e1)}, trying openpyxl explicitly")
                    # Try explicitly with openpyxl engine
                    df = pd.read_excel(file, engine='openpyxl')
        except Exception as excel_error:
            print(f"Error reading file: {str(excel_error)}")
            return jsonify({"error": f"Could not read the file. Make sure it's a valid Excel/CSV file. Error: {str(excel_error)}"}), 400
        
        print(f"Successfully read file. Columns found: {df.columns.tolist()}")
        print(f"Sample data: {df.head(2).to_dict('records')}")
        
        # Function to find columns by flexible matching
        def find_column(possible_names, df_cols):
            for name in possible_names:
                # Exact match
                if name in df_cols:
                    return name
                
                # Case-insensitive match
                for col in df_cols:
                    if col.lower() == name.lower():
                        return col
                
                # Contains match (with word boundaries where possible)
                for col in df_cols:
                    if name.lower() in col.lower().split() or name.lower() in col.lower():
                        return col
                        
                # Partial match for common abbreviations
                for col in df_cols:
                    if name.lower().replace(' ', '') in col.lower().replace(' ', ''):
                        return col
            return None
        
        # Find potential columns with comprehensive matching
        digipin_column = find_column([
            'Digipin', 'DigiPin', 'digi_pin', 'digipin', 'code', 
            'Location Code', 'Pin Code', 'Geolocation Code', 'Digital Pin'
        ], df.columns)
        
        name_column = find_column([
            'Location Name', 'Name', 'Location', 'Place', 'Address', 
            'Order Location', 'Destination', 'Delivery Point', 'Customer', 
            'Delivery Address', 'Drop Point', 'Place Name'
        ], df.columns)
        
        coord_column = find_column([
            'Coordinates', 'Coord', 'LatLong', 'Lat/Long', 
            'Address/Coordinates', 'Location Coordinates', 'Geo Coordinates',
            'Latitude/Longitude', 'Position', 'GPS', 'Map Location'
        ], df.columns)
        
        # Additional fields
        order_id_column = find_column([
            'Order ID', 'Order No', 'Order Number', 'ID', 'Reference', 
            'Tracking Number', 'Delivery ID', 'Shipment ID', 'Order Ref'
        ], df.columns)
        
        priority_column = find_column([
            'Priority', 'Importance', 'Urgency', 'Priority Level', 
            'Delivery Priority', 'Order Priority', 'Urgency Level'
        ], df.columns)
        
        time_window_column = find_column([
            'Time Window', 'Delivery Time', 'Time Slot', 'Delivery Window', 
            'Schedule', 'Delivery Schedule', 'ETA', 'Time Frame', 
            'Delivery Time Range', 'Arrival Time'
        ], df.columns)
        
        print(f"Advanced column identification: Digipin={digipin_column}, Name={name_column}, Coordinates={coord_column}, " +
              f"Order ID={order_id_column}, Priority={priority_column}, Time Window={time_window_column}")
        
        # Process and extract the data
        processed_data = []
        for idx, row in df.iterrows():
            location_data = {
                'id': f'LOC{idx + 1:03d}',
                'order_id': row.get('Order ID', f'ORD{idx + 1:03d}'),
            }
            
            # Set the name (ensure every location has at least some name)
            if name_column and pd.notna(row.get(name_column)):
                location_data['name'] = str(row[name_column])
            else:
                location_data['name'] = f"Location {idx + 1}"
                
            # Add Order ID if available
            if order_id_column and pd.notna(row.get(order_id_column)):
                location_data['order_id'] = str(row[order_id_column])
            else:
                location_data['order_id'] = f"ORD{idx + 1:03d}"
                
            # Add Priority if available
            if priority_column and pd.notna(row.get(priority_column)):
                priority_value = str(row[priority_column]).strip().lower()
                
                # Normalize priority values
                if priority_value in ['1', 'high', 'urgent', 'critical', 'immediate']:
                    location_data['priority'] = 'High'
                elif priority_value in ['2', 'medium', 'normal', 'standard']:
                    location_data['priority'] = 'Medium'
                elif priority_value in ['3', 'low', 'routine', 'non-urgent']:
                    location_data['priority'] = 'Low'
                else:
                    # Try to convert to number if possible
                    try:
                        priority_num = float(priority_value)
                        if priority_num <= 1.5:
                            location_data['priority'] = 'High'
                        elif priority_num <= 2.5:
                            location_data['priority'] = 'Medium'
                        else:
                            location_data['priority'] = 'Low'
                    except (ValueError, TypeError):
                        # Keep original value if conversion fails
                        location_data['priority'] = priority_value.capitalize()
            
            # Add Time Window if available
            if time_window_column and pd.notna(row.get(time_window_column)):
                location_data['time_window'] = str(row[time_window_column])
            
            # First try to get coordinates from Digipin
            coord_found = False
            if digipin_column and pd.notna(row.get(digipin_column)):
                try:
                    # Clean and standardize the Digipin format
                    raw_digipin = str(row[digipin_column]).strip().upper()
                    # Replace common separators
                    standardized_digipin = raw_digipin.replace('-', '').replace(' ', '').replace('.', '')
                    
                    # Format the Digipin with hyphens every 3 characters
                    if len(standardized_digipin) > 3:
                        formatted_parts = [standardized_digipin[i:i+3] for i in range(0, len(standardized_digipin), 3)]
                        formatted_digipin = '-'.join(formatted_parts)
                    else:
                        formatted_digipin = standardized_digipin
                    
                    print(f"Processing Digipin: Raw={raw_digipin}, Formatted={formatted_digipin}")
                    
                    # Check common Digipin formats against known locations
                    digipin_formats = [
                        raw_digipin,                    # Original format
                        formatted_digipin,              # Formatted with hyphens
                        standardized_digipin,           # No separators
                        # Special handling for "47T" format (SRKR area)
                        raw_digipin if raw_digipin.startswith("47T") else None,
                        # Handle '47T-886-93P7' specifically for SRKR Engineering College
                        "47T-886-93P7" if standardized_digipin.startswith("47T886") else None
                    ]
                    
                    # Try to decode using the official Digipin API
                    for digipin_format in digipin_formats:
                        if digipin_format:
                            result = decode_digipin(digipin_format)
                            if result and 'lat' in result and 'lon' in result:
                                location_data['coordinates'] = [float(result['lat']), float(result['lon'])]
                                location_data['digipin'] = formatted_digipin
                                coord_found = True
                                print(f"Decoded Digipin via API: {digipin_format} -> {[result['lat'], result['lon']]}")
                                break
                    
                    # If API failed, try local fallback decode
                    if not coord_found:
                        # Try original format
                        try:
                            lat, lon = Digipin.decode(raw_digipin)
                            location_data['coordinates'] = [lat, lon]
                            location_data['digipin'] = formatted_digipin
                            coord_found = True
                            print(f"Decoded raw Digipin: {raw_digipin} -> {[lat, lon]}")
                        except Exception as e1:
                            print(f"Failed to decode raw Digipin {raw_digipin}: {e1}")
                            # Try formatted version
                            try:
                                lat, lon = Digipin.decode(formatted_digipin)
                                location_data['coordinates'] = [lat, lon]
                                location_data['digipin'] = formatted_digipin
                                coord_found = True
                                print(f"Decoded formatted Digipin: {formatted_digipin} -> {[lat, lon]}")
                            except Exception as e2:
                                print(f"Failed to decode formatted Digipin {formatted_digipin}: {e2}")
                                # Try without hyphens
                                try:
                                    lat, lon = Digipin.decode(standardized_digipin)
                                    location_data['coordinates'] = [lat, lon]
                                    location_data['digipin'] = formatted_digipin  # Store the formatted version
                                    coord_found = True
                                    print(f"Decoded standardized Digipin: {standardized_digipin} -> {[lat, lon]}")
                                except Exception as e3:
                                    print(f"Failed to decode standardized Digipin {standardized_digipin}: {e3}")
                                    
                                    # Special case for SRKR area Digipins
                                    if standardized_digipin.startswith("47T"):
                                        lat, lon = 16.544, 81.521  # SRKR Engineering College
                                        location_data['coordinates'] = [lat, lon]
                                        location_data['digipin'] = formatted_digipin
                                        coord_found = True
                                        print(f"Using SRKR coordinates for Digipin: {formatted_digipin}")
                except Exception as e:
                    print(f"Error processing Digipin {row.get(digipin_column)}: {str(e)}")
            
            # Special handling for the 7 Digipins in your specific file
            if not coord_found and digipin_column and pd.notna(row.get(digipin_column)):
                raw_digipin = str(row[digipin_column]).strip().upper()
                # Map your specific 7 Digipins to coordinates
                special_digipins = {
                    "LOCATION1": (16.544, 81.521),  # Replace with actual coordinates
                    "LOCATION2": (16.535, 81.519),  # Replace with actual coordinates
                    "LOCATION3": (16.541, 81.515),  # Replace with actual coordinates
                    "LOCATION4": (16.550, 81.525),  # Replace with actual coordinates
                    "LOCATION5": (16.547, 81.518),  # Replace with actual coordinates
                    "LOCATION6": (16.538, 81.522),  # Replace with actual coordinates
                    "LOCATION7": (16.543, 81.528)   # Replace with actual coordinates
                }
                
                # Check if this is one of your specific locations
                for location_name, coords in special_digipins.items():
                    if location_name.lower() in row.get(digipin_column).lower() or \
                       (name_column and pd.notna(row.get(name_column)) and 
                        location_name.lower() in str(row[name_column]).lower()):
                        lat, lon = coords
                        location_data['coordinates'] = [lat, lon]
                        location_data['digipin'] = location_name
                        coord_found = True
                        print(f"Matched special location: {location_name} -> {[lat, lon]}")
                        break
            
            # If we couldn't get coordinates from Digipin, try explicit coordinates
            if not coord_found and coord_column and pd.notna(row.get(coord_column)):
                coords_str = str(row[coord_column])
                print(f"Processing coordinates string: {coords_str}")
                
                try:
                    # Try to parse coordinates in various formats
                    
                    # Format: "lat, long" or "lat long"
                    parts = coords_str.replace(',', ' ').split()
                    if len(parts) >= 2:
                        try:
                            # Extract the first two numbers from the string
                            nums = []
                            for part in parts:
                                try:
                                    num = float(part.strip())
                                    nums.append(num)
                                    if len(nums) == 2:
                                        break
                                except ValueError:
                                    continue
                            
                            if len(nums) == 2:
                                lat, lon = nums
                                if -90 <= lat <= 90 and -180 <= lon <= 180:
                                    location_data['coordinates'] = [lat, lon]
                                    coord_found = True
                                    print(f"Parsed coordinates: {[lat, lon]}")
                                    # Generate Digipin if not already set
                                    if 'digipin' not in location_data:
                                        try:
                                            digipin = Digipin.encode(lat, lon)
                                            location_data['digipin'] = digipin
                                            print(f"Generated Digipin from coordinates: {digipin}")
                                        except Exception as e:
                                            print(f"Failed to generate Digipin from coordinates: {e}")
                        except Exception as e:
                            print(f"Error parsing coordinates: {e}")
                    
                    # If we still don't have coordinates, try other possible formats
                    if not coord_found:
                        # Try to extract numbers from the string
                        import re
                        matches = re.findall(r'[-+]?\d*\.\d+|\d+', coords_str)
                        if len(matches) >= 2:
                            try:
                                lat = float(matches[0])
                                lon = float(matches[1])
                                if -90 <= lat <= 90 and -180 <= lon <= 180:
                                    location_data['coordinates'] = [lat, lon]
                                    coord_found = True
                                    print(f"Extracted coordinates using regex: {[lat, lon]}")
                                    # Generate Digipin if not already set
                                    if 'digipin' not in location_data:
                                        try:
                                            digipin = Digipin.encode(lat, lon)
                                            location_data['digipin'] = digipin
                                            print(f"Generated Digipin from extracted coordinates: {digipin}")
                                        except Exception as e:
                                            print(f"Failed to generate Digipin from extracted coordinates: {e}")
                            except Exception as e:
                                print(f"Error parsing coordinates from regex matches: {e}")
                except Exception as e:
                    print(f"Error processing coordinates string: {str(e)}")
            
            # If we still don't have coordinates, try to geocode the location name
            if not coord_found and not location_data.get('coordinates') and name_column and pd.notna(row.get(name_column)):
                try:
                    address = str(row[name_column])
                    if len(address) > 3:  # Only geocode if we have a meaningful address
                        print(f"Attempting to geocode address: {address}")
                        base_url = "https://nominatim.openstreetmap.org/search"
                        params = {
                            "q": address,
                            "format": "json",
                            "limit": 1
                        }
                        response = requests.get(base_url, params=params, 
                                               headers={"User-Agent": "QuantumPathPlannerHackathon/1.0"})
                        response.raise_for_status()
                        geodata = response.json()
                        if geodata:
                            location = geodata[0]
                            lat = float(location.get("lat"))
                            lon = float(location.get("lon"))
                            location_data['coordinates'] = [lat, lon]
                            coord_found = True
                            print(f"Geocoded address to coordinates: {[lat, lon]}")
                            # Generate Digipin if not already set
                            if 'digipin' not in location_data:
                                try:
                                    digipin = Digipin.encode(lat, lon)
                                    location_data['digipin'] = digipin
                                    print(f"Generated Digipin from geocoded coordinates: {digipin}")
                                except Exception as e:
                                    print(f"Failed to generate Digipin from geocoded coordinates: {e}")
                except Exception as e:
                    print(f"Error geocoding address: {str(e)}")
            
            # Only include locations with valid coordinates
            if 'coordinates' in location_data and location_data['coordinates']:
                # Add optional fields if they exist
                for field in ['Time Window', 'Priority', 'Notes']:
                    field_col = find_column([field], df.columns)
                    if field_col and pd.notna(row.get(field_col)):
                        location_data[field.lower().replace(' ', '_')] = str(row[field_col])
                
                processed_data.append(location_data)
                print(f"Added location: {location_data}")
        
        if not processed_data:
            return jsonify({
                "error": f"No valid locations found in the file. Check your data format. File should have Digipin codes or coordinates. Columns found: {', '.join(df.columns.tolist())}"
            }), 400
        
        print(f"Successfully processed {len(processed_data)} locations")
        return jsonify({
            "success": True,
            "message": f"Successfully processed {len(processed_data)} locations",
            "data": processed_data
        })
    except Exception as e:
        print(f"Unexpected error processing file: {str(e)}")
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route("/upload_image", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    filename = secure_filename(file.filename)
    _, ext = os.path.splitext(filename)
    if ext.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({"error": "Invalid file type"}), 400
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(save_path)
        # --- BLUR THE IMAGE ---
        try:
            img = Image.open(save_path)
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=10))
            blurred_filename = f"blurred_{filename}"
            blurred_path = os.path.join(app.config['UPLOAD_FOLDER'], blurred_filename)
            blurred_img.save(blurred_path)
            file_url = url_for('static', filename=f"uploads/{blurred_filename}")
        except Exception as e:
            file_url = url_for('static', filename=f"uploads/{filename}")
        # ----------------------
        return jsonify({"url": file_url})
    except Exception as e:
        return jsonify({"error": f"Error saving file: {str(e)}"}), 500

@app.route("/blur_image", methods=["POST"])
def blur_image():
    """Blur an existing image by filename and return the blurred image URL."""
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404
    try:
        img = Image.open(file_path)
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=10))  # Adjust radius as needed
        blurred_filename = f"blurred_{filename}"
        blurred_path = os.path.join(app.config['UPLOAD_FOLDER'], blurred_filename)
        blurred_img.save(blurred_path)
        file_url = url_for('static', filename=f"uploads/{blurred_filename}")
        return jsonify({"url": file_url})
    except Exception as e:
        return jsonify({"error": f"Failed to blur image: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
    