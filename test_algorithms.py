import requests
import json

# Test with different scenarios
test_cases = [
    {
        "name": "4 locations, 2 vehicles with start location",
        "locations": [[40.7589, -73.9851], [40.6892, -73.9442], [40.7282, -73.7949], [40.8448, -73.8648]],
        "num_vehicles": 2,
        "start_location": [40.7505, -73.9934]  # Central location
    },
    {
        "name": "6 locations, 3 vehicles with start location", 
        "locations": [
            [40.7589, -73.9851], [40.6892, -73.9442], [40.7282, -73.7949], [40.8448, -73.8648], 
            [40.7128, -74.0060], [40.7505, -73.9934]
        ],
        "num_vehicles": 3,
        "start_location": [40.7829, -73.9654]  # Central Park as depot
    }
]

for test_case in test_cases:
    print(f"\n{'='*50}")
    print(f"Testing: {test_case['name']}")
    print(f"{'='*50}")
    
    try:
        response = requests.post('http://localhost:5000/solve_tsp', 
                               json={'locations': test_case['locations'], 
                                    'num_vehicles': test_case['num_vehicles'],
                                    'start_location': test_case['start_location']})
        
        if response.status_code == 200:
            data = response.json()
            
            classical_total = data['greedy_vrp']['total_length_km']
            quantum_total = data['qaoa_vrp']['total_length_km']
            
            print(f"Classical total distance: {classical_total:.2f} km")
            print(f"Quantum total distance: {quantum_total:.2f} km")
            print(f"Different results: {classical_total != quantum_total}")
            
            print("\nClassical routes:")
            for i, route in enumerate(data['greedy_vrp']['routes']):
                print(f"  Vehicle {i+1}: {route['tour']} ({route['length_km']:.2f}km)")
            
            print("\nQuantum routes:")
            for i, route in enumerate(data['qaoa_vrp']['routes']):
                print(f"  Vehicle {i+1}: {route['tour']} ({route['length_km']:.2f}km)")
                
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Exception: {e}")
