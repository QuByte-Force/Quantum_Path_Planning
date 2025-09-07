#!/usr/bin/env python3
"""
Test script to verify the VRP location duplication fix
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from app import cluster_kmeans, cluster_nearest_split, cluster_distance_optimized, validate_routes, calculate_distance_matrix

def test_clustering_functions():
    """Test that all clustering functions work correctly without duplicating locations"""
    
    # Test coordinates (depot + 6 locations)
    coords = [
        [0, 0],      # Depot
        [1, 1],      # Location 1
        [2, 2],      # Location 2
        [3, 3],      # Location 3
        [4, 4],      # Location 4
        [5, 5],      # Location 5
        [6, 6]       # Location 6
    ]
    
    distance_matrix = calculate_distance_matrix(coords)
    num_vehicles = 3
    
    print(f"Testing with {len(coords)-1} locations and {num_vehicles} vehicles")
    print("=" * 60)
    
    strategies = [
        ("K-means", cluster_kmeans),
        ("Nearest Split", cluster_nearest_split),
        ("Distance Optimized", cluster_distance_optimized)
    ]
    
    for strategy_name, strategy_func in strategies:
        try:
            print(f"\n🧪 Testing {strategy_name} strategy:")
            
            # Get clustering results
            clustered_locs = strategy_func(coords, num_vehicles, distance_matrix)
            print(f"   Raw clusters: {clustered_locs}")
            
            # Build vehicle routes with depots
            vehicle_routes = []
            for cluster in clustered_locs:
                if cluster:  # Only if cluster has locations
                    vehicle_routes.append([0] + cluster + [0])
            
            print(f"   Vehicle routes: {vehicle_routes}")
            
            # Validate routes
            is_valid = validate_routes(vehicle_routes, len(coords))
            print(f"   Validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
            
            # Check for duplicates manually
            all_assigned_locations = []
            for route in vehicle_routes:
                # Skip depot (0) at start and end
                locations_in_route = route[1:-1]
                all_assigned_locations.extend(locations_in_route)
            
            unique_locations = set(all_assigned_locations)
            expected_locations = set(range(1, len(coords)))  # Should be {1, 2, 3, 4, 5, 6}
            
            print(f"   Assigned locations: {sorted(all_assigned_locations)}")
            print(f"   Expected locations: {sorted(expected_locations)}")
            print(f"   Duplicates check: {'✅ NO DUPLICATES' if len(all_assigned_locations) == len(unique_locations) else '❌ DUPLICATES FOUND'}")
            print(f"   Completeness check: {'✅ ALL COVERED' if unique_locations == expected_locations else '❌ MISSING LOCATIONS'}")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_clustering_functions()
