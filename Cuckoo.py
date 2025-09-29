import numpy as np
import random

# Number of nests (solutions)
num_nests = 20

# Discovery probability (P)
P = 0.25

# Maximum number of iterations
max_iter = 100

# Number of jobs (tasks) and machines
num_jobs = 10
num_machines = 3

# Random job processing times on each machine
job_times = np.random.randint(1, 10, size=(num_jobs, num_machines)).tolist()

print("Job Processing Times (rows = jobs, cols = machines):")
for i, row in enumerate(job_times):
    print(f"Job {i}: {row}")
print("-" * 50)


# flow shop fitness function (makespan)
def fitness(schedule, job_times):
    num_jobs = len(schedule)
    num_machines = len(job_times[0])
    completion = np.zeros((num_jobs, num_machines))

    for i, job in enumerate(schedule):
        for m in range(num_machines):
            if i == 0 and m == 0:
                completion[i, m] = job_times[job][m]
            elif i == 0:
                completion[i, m] = completion[i, m-1] + job_times[job][m]
            elif m == 0:
                completion[i, m] = completion[i-1, m] + job_times[job][m]
            else:
                completion[i, m] = max(completion[i-1, m], completion[i, m-1]) + job_times[job][m]

    return completion[-1, -1]  # makespan


# Mutation function (swap or reverse subsequence)
def mutate_permutation(nest):
    new_nest = list(nest)
    if random.random() < 0.5:
        # Swap two jobs
        i, j = random.sample(range(len(new_nest)), 2)
        new_nest[i], new_nest[j] = new_nest[j], new_nest[i]
    else:
        # Reverse a subsequence
        i, j = sorted(random.sample(range(len(new_nest)), 2))
        new_nest[i:j] = reversed(new_nest[i:j])
    return new_nest


# Cuckoo Search Algorithm
def cuckoo_search(num_nests, num_jobs, job_times, P, max_iter=100, verbose=True):
    nests = [random.sample(range(num_jobs), num_jobs) for _ in range(num_nests)]
    fitness_values = [fitness(nest, job_times) for nest in nests]

    best_nest = nests[np.argmin(fitness_values)]
    best_fitness = min(fitness_values)

    if verbose:
        print(f"Init Best -> {best_nest}  Makespan: {best_fitness}")

    for iteration in range(1, max_iter + 1):
        for i, nest in enumerate(nests):
            new_nest = mutate_permutation(nest)
            new_fitness = fitness(new_nest, job_times)

            if new_fitness < fitness_values[i]:
                nests[i] = new_nest
                fitness_values[i] = new_fitness

                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_nest = new_nest

        # Discovery step
        for i in range(num_nests):
            if random.random() < P:
                new_nest = random.sample(range(num_jobs), num_jobs)
                new_fitness = fitness(new_nest, job_times)
                if new_fitness < fitness_values[i]:
                    nests[i] = new_nest
                    fitness_values[i] = new_fitness
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_nest = new_nest

        if verbose:
            print(f"Iter {iteration:<3} Best -> {best_nest}  Makespan: {best_fitness}")

    return best_nest, best_fitness


# Run the algorithm
best_schedule, makespan = cuckoo_search(
    num_nests=num_nests, num_jobs=num_jobs, job_times=job_times, P=P, max_iter=max_iter
)

print("-" * 50)
print("Best Schedule:", best_schedule)
print("Final Makespan:", makespan)
