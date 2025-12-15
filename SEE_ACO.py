import numpy as np
import random

class AntColony:
    def __init__(self, dist_matrix, num_ants, num_iterations, alpha=1, beta=2, rho=0.1, q0=0.9):

        self.dist_matrix = dist_matrix
        self.num_cities = len(dist_matrix)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0

        # Initialize pheromone matrix
        self.pheromone = np.ones((self.num_cities, self.num_cities)) / self.num_cities
        self.best_tour = None
        self.best_tour_length = float('inf')

    def _initialize_ant_tour(self):

        tour = []
        visited = [False] * self.num_cities
        start_city = random.randint(0, self.num_cities - 1)
        current_city = start_city
        visited[current_city] = True
        tour.append(current_city)

        for _ in range(self.num_cities - 1):
            next_city = self._select_next_city(current_city, visited)
            tour.append(next_city)
            visited[next_city] = True
            current_city = next_city

        # Add return to the start city
        tour.append(start_city)
        return tour

    def _select_next_city(self, current_city, visited):

        probabilities = []
        total_heuristic_value = 0.0 

        for city in range(self.num_cities):
            if not visited[city]:
                pheromone_term = self.pheromone[current_city][city] ** self.alpha
                distance_term = (1.0 / self.dist_matrix[current_city][city]) ** self.beta if self.dist_matrix[current_city][city] != 0 else 0 
                total_heuristic_value += pheromone_term * distance_term

        if total_heuristic_value == 0:
            unvisited_cities = [city for city, v in enumerate(visited) if not v]
            if unvisited_cities:
                return random.choice(unvisited_cities)
            else:
                return -1 

        for city in range(self.num_cities):
            if not visited[city]:
                pheromone_term = self.pheromone[current_city][city] ** self.alpha
                distance_term = (1.0 / self.dist_matrix[current_city][city]) ** self.beta if self.dist_matrix[current_city][city] != 0 else 0
                probability = (pheromone_term * distance_term) / total_heuristic_value
                probabilities.append((city, probability))

        if not probabilities:
           
            return -1

        if random.random() < self.q0:
            # Exploit: choose the best city
            next_city = max(probabilities, key=lambda x: x[1])[0]
        else:
            
            cities, probs = zip(*probabilities)
            probs_sum = sum(probs)
            if probs_sum == 0: 
                return random.choice(cities) 
            normalized_probs = np.array(probs) / probs_sum
            next_city = np.random.choice(cities, p=normalized_probs)

        return next_city

    def _update_pheromones(self, all_tours):
       
        # Evaporate pheromones
        self.pheromone *= (1 - self.rho)

        
        for tour in all_tours:
            tour_length = self._calculate_tour_length(tour)
            pheromone_deposit = 1.0 / tour_length

            for i in range(self.num_cities):
                city_from = tour[i]
                city_to = tour[(i + 1) % self.num_cities]
                self.pheromone[city_from][city_to] += pheromone_deposit
                self.pheromone[city_to][city_from] += pheromone_deposit

    def _calculate_tour_length(self, tour):

        length = 0.0
        for i in range(self.num_cities):
            city_from = tour[i]
            city_to = tour[(i + 1) % self.num_cities]
            length += self.dist_matrix[city_from][city_to]
        return length

    def solve(self):

        for iteration in range(self.num_iterations):
            all_tours = []
            all_tour_lengths = []

            # Generate all ant tours
            for _ in range(self.num_ants):
                tour = self._initialize_ant_tour()
                tour_length = self._calculate_tour_length(tour)
                all_tours.append(tour)
                all_tour_lengths.append(tour_length)

                # Update the best tour found so far
                if tour_length < self.best_tour_length:
                    self.best_tour_length = tour_length
                    self.best_tour = tour

            # Update pheromones based on the tours
            self._update_pheromones(all_tours)

            print(f"Iteration {iteration + 1}/{self.num_iterations} - Best Tour Length: {self.best_tour_length}")

        return self.best_tour, self.best_tour_length


if __name__ == "__main__":

    num_cities = 10
    dist_matrix = np.random.randint(5, 50, size=(num_cities, num_cities)).astype(float)
    np.fill_diagonal(dist_matrix, 0) 
  
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            dist_matrix[j][i] = dist_matrix[i][j]


    aco = AntColony(dist_matrix, num_ants=20, num_iterations=100, alpha=1, beta=2, rho=0.1, q0=0.9)

   
    best_tour, best_tour_length = aco.solve()

    print("Best Tour:", best_tour)
    print("Best Tour Length:", best_tour_length)
