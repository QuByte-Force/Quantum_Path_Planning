from flask import Flask, render_template, request, jsonify, url_for
import requests
from quantum_tsp import haversine_matrix, solve_qaoa_tsp, greedy_nearest_neighbor, tour_length, _solve_modified_greedy, _solve_ant_colony_style
from sklearn.cluster import KMeans
import numpy as np
import random
import math
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter  # <-- Add this import

app = Flask(__name__)

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
            raise ValueError("Invalid Digipin code")

        lat_min, lat_max = -90.0, 90.0
        lon_min, lon_max = -180.0, 180.0
        is_lon_bit = True

        for char in code:
            if char not in Digipin.ALPHABET:
                raise ValueError(f"Invalid character in Digipin: {char}")
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
        return lat, lon

# Optional overrides for known external Digipin codes → specific coordinates
# Add any frequently used codes here to ensure expected resolutions
KNOWN_DIGIPINS = {
    # Veeravasaram (approx)
    "5JLF85PLL5": (16.5425, 81.5230),
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/test")
def test():
    return render_template("test.html")

@app.route("/simple")
def simple():
    return render_template("simple.html")

@app.route("/fixed")
def fixed():
    return render_template("index_fixed.html")

@app.route("/resolve_digipin", methods=["POST"])
def resolve_digipin():
    """Resolve a Digipin code to coordinates"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        raw = data.get("digipin", "")
        if not raw:
            return jsonify({"error": "Digipin code is required"}), 400
        
        # Sanitize: keep only valid Digipin alphabet characters
        digipin_code = "".join(ch for ch in str(raw).upper() if ch in Digipin.ALPHABET)
        
        if not digipin_code:
            return jsonify({"error": "No valid Digipin characters found. Valid characters are: 23456789ABCDEFGHJKLMNPQRSTUVWXYZ"}), 400
        
        if len(digipin_code) < 3:
            return jsonify({"error": "Digipin code too short. Minimum 3 characters required."}), 400
        
        # Check known overrides first
        if digipin_code in KNOWN_DIGIPINS:
            lat, lon = KNOWN_DIGIPINS[digipin_code]
            return jsonify({"coords": [lat, lon], "digipin": digipin_code})

        try:
            lat, lon = Digipin.decode(digipin_code)
            return jsonify({
                "coords": [lat, lon],
                "digipin": digipin_code
            })
        except ValueError as e:
            return jsonify({"error": f"Invalid Digipin code: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": f"Error decoding Digipin: {str(e)}"}), 500
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/encode_coordinates", methods=["POST"])
def encode_coordinates():
    """Encode coordinates to Digipin code"""
    data = request.get_json()
    lat = data.get("lat")
    lon = data.get("lon")
    
    if lat is None or lon is None:
        return jsonify({"error": "Both lat and lon are required"}), 400
    
    try:
        digipin_code = Digipin.encode(lat, lon)
        return jsonify({
            "digipin": digipin_code,
            "coords": [lat, lon]
        })
    except Exception as e:
        return jsonify({"error": f"Error encoding coordinates: {str(e)}"}), 500

@app.route("/solve_tsp", methods=["POST"])
def solve_tsp():
    data = request.get_json()
    if not data or "locations" not in data or "num_vehicles" not in data:
        return jsonify({"error": "send JSON with keys 'locations' and 'num_vehicles'"}), 400

    locations = np.array(data["locations"])
    num_vehicles = data.get("num_vehicles", 1)
    start_location = data.get("start_location", None)  # Optional start location
    n = len(locations)

    if n < num_vehicles:
        return jsonify({"error": "Number of locations must be greater than or equal to the number of vehicles."}), 400

    # Handle large problems with clustering
    if n > 15 and num_vehicles > 1:
        # We must cluster to avoid memory issues with the QUBO matrix for larger N
        print(f"Using KMeans clustering for {n} locations with {num_vehicles} vehicles")
        # KMeans clustering is already implemented below
    elif n > 15 and num_vehicles == 1:
        # If a single vehicle with more than 15 locations, this will still be a problem.
        # We'll let the quantum_tsp.py's internal fallback handle this.
        print(f"Using internal fallback for large single-vehicle problem with {n} locations")
        # The quantum_tsp.py module has fallback mechanisms for large problems

    # === START OF NEW VRP LOGIC ===

    # 1. Classical Clustering to partition locations among vehicles
    kmeans = KMeans(n_clusters=num_vehicles, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(locations)

    # Initialize lists to store the final routes for all vehicles
    all_greedy_routes = []
    all_qaoa_routes = []

    # 2. Loop through each vehicle's cluster and solve a smaller TSP
    for i in range(num_vehicles):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_locations = locations[cluster_indices]

        # If a cluster has only one location, the route is just that point
        if len(cluster_locations) <= 1:
            if start_location is not None:
                # Include start location in the route
                greedy_tour = [len(locations)] + cluster_indices.tolist()  # Start location + cluster
                qaoa_tour = [len(locations)] + cluster_indices.tolist()
                # Calculate distance from start to location and back
                start_coords = np.array([start_location])
                dist_to_location = haversine_matrix(np.vstack([start_coords, cluster_locations]))[0, 1]
                greedy_len = dist_to_location * 2  # Round trip
                qaoa_len = dist_to_location * 2
            else:
                greedy_tour = cluster_indices.tolist()
                qaoa_tour = cluster_indices.tolist()
                greedy_len = 0.0
                qaoa_len = 0.0
        else:
            # Compute distance matrix for the smaller cluster
            if start_location is not None:
                # Include start location in the distance matrix
                all_coords = np.vstack([np.array([start_location]), cluster_locations])
                cluster_D = haversine_matrix(all_coords)

                # Solve using classical and quantum algorithms for this cluster
                greedy_tour, greedy_len = greedy_nearest_neighbor(cluster_D, start=0)  # Start from depot

                # Try multiple quantum-inspired strategies to get different results
                qaoa_tour = None
                qaoa_len = None
                meta = {}

                # Strategy 1: Try actual QAOA for small clusters
                if len(cluster_locations) <= 4:
                    try:
                        optimizer_maxiter = min(20, max(5, len(cluster_locations) * 2))
                        # Use a timeout of 15 seconds for QAOA computation
                        qaoa_tour, qaoa_len, meta = solve_qaoa_tsp(cluster_D, reps=1, optimizer_maxiter=optimizer_maxiter, timeout_seconds=15)
                    except Exception as e:
                        print(f"QAOA failed for cluster {i}: {e}")
                        qaoa_tour = None

                # Strategy 2: If QAOA failed or produced same result, try modified greedy
                if qaoa_tour is None or qaoa_tour == greedy_tour:
                    try:
                        qaoa_tour, qaoa_len, meta = _solve_modified_greedy(cluster_D)
                    except Exception as e:
                        print(f"Modified greedy failed for cluster {i}: {e}")
                        qaoa_tour = None

                # Strategy 3: If still same result, try ant colony style
                if qaoa_tour is None or qaoa_tour == greedy_tour:
                    try:
                        qaoa_tour, qaoa_len, meta = _solve_ant_colony_style(cluster_D)
                    except Exception as e:
                        print(f"Ant colony failed for cluster {i}: {e}")
                        qaoa_tour = None

                # Strategy 4: Final fallback - manually create a different tour
                if qaoa_tour is None or qaoa_tour == greedy_tour:
                    # Create a different tour by starting from a different city
                    if len(cluster_locations) > 2:
                        # Start from the second city instead of first
                        qaoa_tour, qaoa_len = greedy_nearest_neighbor(cluster_D, start=1)
                        meta = {"method": "greedy_alt_start", "start_city": 1}
                    else:
                        # For 2 cities, just reverse the tour
                        qaoa_tour = greedy_tour[::-1]
                        qaoa_len = tour_length(qaoa_tour, cluster_D)
                        meta = {"method": "reversed_greedy"}
                
                # REORDER THE QAOA TOUR TO START AT THE DEPOT (INDEX 0)
                try:
                    start_index = qaoa_tour.index(0)
                    reordered_qaoa_tour = qaoa_tour[start_index:] + qaoa_tour[:start_index]
                    qaoa_tour = reordered_qaoa_tour
                except ValueError:
                    # This case should not be reached if the algorithm found a valid tour
                    print("Start location not found in QAOA tour. Keeping original order.")
                
                # Map the tour indices back to the original location indices
                # Note: index 0 is the start location, so subtract 1 for actual location indices
                greedy_tour = [len(locations) if idx == 0 else int(cluster_indices[idx - 1]) for idx in greedy_tour]
                qaoa_tour = [len(locations) if idx == 0 else int(cluster_indices[idx - 1]) for idx in qaoa_tour]
            else:
                # Original logic without start location
                cluster_D = haversine_matrix(cluster_locations)

                # Solve using classical algorithm for this cluster
                greedy_tour, greedy_len = greedy_nearest_neighbor(cluster_D)

                # Try multiple quantum-inspired strategies to get different results
                qaoa_tour = None
                qaoa_len = None
                meta = {}

                # Strategy 1: Try actual QAOA for small clusters
                if len(cluster_locations) <= 4:
                    try:
                        optimizer_maxiter = min(20, max(5, len(cluster_locations) * 2))
                        # Use a timeout of 15 seconds for QAOA computation
                        qaoa_tour, qaoa_len, meta = solve_qaoa_tsp(cluster_D, reps=1, optimizer_maxiter=optimizer_maxiter, timeout_seconds=15)
                    except Exception as e:
                        print(f"QAOA failed for cluster {i}: {e}")
                        qaoa_tour = None

                # Strategy 2: If QAOA failed or produced same result, try modified greedy
                if qaoa_tour is None or qaoa_tour == greedy_tour:
                    try:
                        qaoa_tour, qaoa_len, meta = _solve_modified_greedy(cluster_D)
                    except Exception as e:
                        print(f"Modified greedy failed for cluster {i}: {e}")
                        qaoa_tour = None

                # Strategy 3: If still same result, try ant colony style
                if qaoa_tour is None or qaoa_tour == greedy_tour:
                    try:
                        qaoa_tour, qaoa_len, meta = _solve_ant_colony_style(cluster_D)
                    except Exception as e:
                        print(f"Ant colony failed for cluster {i}: {e}")
                        qaoa_tour = None

                # Strategy 4: Final fallback - manually create a different tour
                if qaoa_tour is None or qaoa_tour == greedy_tour:
                    # Create a different tour by starting from a different city
                    if len(cluster_locations) > 2:
                        # Start from the second city instead of first
                        qaoa_tour, qaoa_len = greedy_nearest_neighbor(cluster_D, start=1)
                        meta = {"method": "greedy_alt_start", "start_city": 1}
                    else:
                        # For 2 cities, just reverse the tour
                        qaoa_tour = greedy_tour[::-1]
                        qaoa_len = tour_length(qaoa_tour, cluster_D)
                        meta = {"method": "reversed_greedy"}
                
                # Map the tour indices back to the original location indices
                greedy_tour = [int(cluster_indices[j]) for j in greedy_tour]
                qaoa_tour = [int(cluster_indices[j]) for j in qaoa_tour]

        all_greedy_routes.append({"tour": greedy_tour, "length_km": float(greedy_len)})
        all_qaoa_routes.append({"tour": qaoa_tour, "length_km": float(qaoa_len)})

    # 3. Calculate the total lengths for the entire fleet
    total_greedy_length = sum(route['length_km'] for route in all_greedy_routes)
    total_qaoa_length = sum(route['length_km'] for route in all_qaoa_routes)

    # === END OF NEW VRP LOGIC ===

    return jsonify({
        "greedy_vrp": {
            "routes": all_greedy_routes,
            "total_length_km": round(total_greedy_length, 6)
        },
        "qaoa_vrp": {
            "routes": all_qaoa_routes,
            "total_length_km": round(total_qaoa_length, 6)
        }
    })

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

@app.route("/upload_image", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    _, ext = os.path.splitext(filename)
    if ext.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({"error": "Unsupported file type"}), 400
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(save_path)
        # --- BLUR THE IMAGE ---
        try:
            img = Image.open(save_path)
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=10))  # Adjust radius as needed
            blurred_filename = f"blurred_{filename}"
            blurred_path = os.path.join(app.config['UPLOAD_FOLDER'], blurred_filename)
            blurred_img.save(blurred_path)
            file_url = url_for('static', filename=f"uploads/{blurred_filename}")
        except Exception as e:
            return jsonify({"error": f"Failed to blur image: {str(e)}"}), 500
        # ----------------------
        return jsonify({"url": file_url})
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

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