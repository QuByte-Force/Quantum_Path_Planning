import requests
import json

# Test data matching your screenshot scenario
test_data = {
    'locations': [
        {'name': 'Depot', 'coordinates': [16.544, 81.521]},      # Depot
        {'name': 'Rice Delivery', 'coordinates': [16.545, 81.522]},      
        {'name': 'Fish Market Delivery', 'coordinates': [16.546, 81.523]},
        {'name': 'Clothing Distribution', 'coordinates': [16.547, 81.524]},
        {'name': 'Medical Shipment', 'coordinates': [16.548, 81.525]}
    ],
    'num_vehicles': 2
}

print('🧪 Testing duplicate detection with realistic coordinates...')
response = requests.post('http://127.0.0.1:5000/solve_tsp', json=test_data)
print(f'Status: {response.status_code}')

if response.status_code == 200:
    result = response.json()
    
    # Check both algorithms
    for algo_name in ['greedy_vrp', 'qaoa_vrp']:
        if algo_name in result:
            print(f'\n🧠 {algo_name.upper()} Results:')
            routes = result[algo_name]['routes']
            
            # Track all assigned locations
            all_locations = []
            location_to_routes = {}
            
            for route in routes:
                tour = route['tour']
                vehicle_id = route['vehicle_id']
                print(f'  🚛 Vehicle {vehicle_id}: {tour}')
                
                # Process each location in the tour
                for idx in tour:
                    if idx == 0:
                        print(f'      {idx}: Depot')
                    else:
                        location_name = test_data['locations'][idx]['name']
                        print(f'      {idx}: {location_name}')
                        
                        # Track assignments
                        all_locations.append(idx)
                        if idx not in location_to_routes:
                            location_to_routes[idx] = []
                        location_to_routes[idx].append(vehicle_id)
            
            # Check for duplicates
            duplicates = {loc: vehicles for loc, vehicles in location_to_routes.items() if len(vehicles) > 1}
            
            print(f'\n📊 Analysis for {algo_name}:')
            print(f'  All location assignments: {all_locations}')
            print(f'  Unique locations: {set(all_locations)}')
            
            if duplicates:
                print(f'  🚨 DUPLICATES FOUND: {duplicates}')
                for loc, vehicles in duplicates.items():
                    location_name = test_data['locations'][loc]['name']
                    print(f'    - "{location_name}" appears in vehicles: {vehicles}')
            else:
                print(f'  ✅ NO DUPLICATES - Each location assigned to exactly one vehicle')
                
else:
    print(f'❌ Error: {response.text}')
