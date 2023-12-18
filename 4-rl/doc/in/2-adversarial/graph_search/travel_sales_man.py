# Adjusted code based on user input
import heapq
from typing import List, Tuple
import numpy as np
from priority_queue import PriorityQueueOptimized
from utils import min_distance_position

# Adapted Greedy Algorithm with Balanced Total Distances
def balanced_multi_salesmen_greedy_tsp(remaining_cities: List[Tuple[int, int]]
                                       , num_salesmen: int
                                       , start_cities: List[Tuple[int, int]]
                                       , finish_cities: List[Tuple[int, int]]):
    """Given a list of cities coordinates, returns a list of cities visited by each salesman
    in the order that minimizes the total distance traveled.
    """
    
    routes = {f"Salesman_{i+1}": [start_cities[i]] for i in range(num_salesmen)}
    distances = {f"Salesman_{i+1}": 0.0 for i in range(num_salesmen)}

    while remaining_cities:
        for salesman in routes.keys():
            if not remaining_cities:
                break
            
            current_city = routes[salesman][-1]
            nearest_city, nearest_distance = min_distance_position(routes[salesman][-1], remaining_cities)
            distances[salesman] += nearest_distance
            routes[salesman].append(nearest_city)
            remaining_cities.remove(nearest_city)

    for salesman in routes.keys():
        current_city = routes[salesman][-1]
        finish_city, final_distance = min_distance_position(current_city, finish_cities)
        distances[salesman] += final_distance
        routes[salesman].append(finish_city)
        
    total_distance = sum(distances.values())
    return routes, distances, total_distance

