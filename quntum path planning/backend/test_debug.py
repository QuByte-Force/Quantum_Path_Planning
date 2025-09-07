import requests
import json

test_data = {
    'locations': [
        {'name': 'Depot', 'coordinates': [0, 0]},
        {'name': 'Rice Delivery', 'coordinates': [1, 1]}, 
        {'name': 'Fish Market Delivery', 'coordinates': [2, 2]},
        {'name': 'Clothing Distribution', 'coordinates': [3, 3]},
        {'name': 'Medical Shipment', 'coordinates': [4, 4]}
    ],
    'num_vehicles': 2
}

print('Testing with names similar to your screenshot...')
response = requests.post('http://127.0.0.1:5000/solve_tsp', json=test_data)
print('Status:', response.status_code)
if response.status_code == 200:
    result = response.json()
    print('\nGreedy routes:')
    for route in result.get('greedy_vrp', {}).get('routes', []):
        tour = route.get('tour', [])
        print(f"  Vehicle {route.get('vehicle_id')}: {tour}")
        for idx in tour:
            if idx == 0:
                print(f'    {idx}: Depot')
            else:
                print(f"    {idx}: {test_data['locations'][idx]['name']}")
else:
    print('Error:', response.text)
