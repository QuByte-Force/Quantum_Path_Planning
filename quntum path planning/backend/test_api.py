#!/usr/bin/env python3
"""
Test script to verify the VRP location duplication fix via API call
"""

import requests
import json

def test_vrp_api():
    """Test the VRP API to ensure no location duplication"""
    
    # Test data - 7 locations as mentioned by the user (depot + 6 locations)
    test_data = {
        "locations": [
            {"name": "Depot", "coordinates": [0, 0]},
            {"name": "Location 1", "coordinates": [1, 1]},
            {"name": "Location 2", "coordinates": [2, 2]},
            {"name": "Location 3", "coordinates": [3, 3]},
            {"name": "Location 4", "coordinates": [4, 4]},
            {"name": "Location 5", "coordinates": [5, 5]},
            {"name": "Location 6", "coordinates": [6, 6]}
        ],
        "num_vehicles": 3
    }
    
    try:
        print("🧪 Testing VRP API with 7 locations (6 + depot) and 3 vehicles")
        print("=" * 60)
        
        response = requests.post("http://127.0.0.1:5000/solve_tsp", json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if we got the new multi-algorithm response format
            if "greedy_vrp" in result:
                print(f"✅ API call successful (multi-algorithm response)")
                
                for algo_name, algo_result in result.items():
                    if isinstance(algo_result, dict) and "routes" in algo_result:
                        routes = algo_result["routes"]
                        
                        print(f"\n🧠 Algorithm: {algo_name.upper()}")
                        print(f"📊 Total routes returned: {len(routes)}")
                        print(f"🎯 Optimization method: {algo_result.get('meta', {}).get('clustering_algorithm', 'Unknown')}")
                        print(f"📏 Total distance: {algo_result.get('total_length_km', 'Unknown')}km")
                        
                        # Analyze location assignments
                        all_assigned_locations = []
                        for i, route in enumerate(routes):
                            tour = route.get("tour", [])
                            print(f"\n🚛 Vehicle {i+1} tour: {tour}")
                            
                            # Extract location indices (skip depot at start)
                            if len(tour) > 1:
                                # Check if tour returns to depot (ends with 0)
                                if tour[-1] == 0:
                                    visited_locations = tour[1:-1]  # Skip first (depot) and last (return to depot)
                                else:
                                    visited_locations = tour[1:]  # Skip only first (depot)
                                print(f"   Visits locations: {visited_locations}")
                                all_assigned_locations.extend(visited_locations)
                            else:
                                print(f"   Empty route (only depot)")
                        
                        # Check for duplicates
                        unique_locations = set(all_assigned_locations)
                        expected_locations = set(range(1, len(test_data["locations"])))  # {1, 2, 3, 4, 5, 6}
                        
                        print(f"\n📋 Analysis for {algo_name}:")
                        print(f"   All assigned locations: {sorted(all_assigned_locations)}")
                        print(f"   Expected locations: {sorted(expected_locations)}")
                        print(f"   Total assignments: {len(all_assigned_locations)}")
                        print(f"   Unique assignments: {len(unique_locations)}")
                        
                        # Validation checks
                        has_duplicates = len(all_assigned_locations) != len(unique_locations)
                        all_covered = unique_locations == expected_locations
                        
                        print(f"\n🔍 Validation Results for {algo_name}:")
                        print(f"   No duplicates: {'❌ FAIL' if has_duplicates else '✅ PASS'}")
                        print(f"   All locations covered: {'❌ FAIL' if not all_covered else '✅ PASS'}")
                        
                        if has_duplicates:
                            from collections import Counter
                            counts = Counter(all_assigned_locations)
                            duplicates = {loc: count for loc, count in counts.items() if count > 1}
                            print(f"   🚨 Duplicate locations found: {duplicates}")
                        
                        if not all_covered:
                            missing = expected_locations - unique_locations
                            extra = unique_locations - expected_locations
                            if missing:
                                print(f"   🚨 Missing locations: {missing}")
                            if extra:
                                print(f"   🚨 Extra locations: {extra}")
                                
            else:
                # Handle old format
                routes = result.get("routes", [])
                print(f"✅ API call successful")
                print(f"📊 Total routes returned: {len(routes)}")
                print(f"🎯 Algorithm used: {result.get('method', 'Unknown')}")
                print(f"📏 Total distance: {result.get('total_distance', 'Unknown')}km")
                    
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend server. Make sure it's running on http://127.0.0.1:5000")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vrp_api()
