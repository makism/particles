""" Boilerplate code for PSO (gbest).

In this demo we optimize DeJong's function.

"""
# author Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
from joblib import Parallel, delayed


def evaluate_particle(particle, pos):
    curr_fitness = pos[0] ** 2 + pos[1] ** 2
    return particle, pos, curr_fitness


if __name__ == "__main__":
    n = 400000
    DATA = np.zeros((n, 3))
    i = 0
    for x in np.arange(-1.0, 1.0, 0.0125):
        for y in np.arange(-1.0, 1.0, 0.0125):
            z = x ** 2 + y ** 2
            DATA[i, :] = x, z, y
            i += 1

    VIEW_DATA = DATA
    DRAW_DATA = np.unique(DATA, axis=0)

    rng = np.random.RandomState(0)

    n_particles = 100
    n_dims = 3
    omega = 0.25
    phi_p, phi_g = [0.1, 0.01]
    initial_indices = rng.choice(DRAW_DATA.shape[0], size=n_particles, replace=False)
    positions = DRAW_DATA[initial_indices, :]
    best_positions = positions
    fitness_scores = np.zeros((n_particles))
    best_fitness_scores = np.zeros((n_particles, 1), dtype=np.float32)
    velocities = rng.rand(n_particles, n_dims)
    results = Parallel(n_jobs=5)(
        delayed(evaluate_particle)(particle, pos)
        for particle, pos in enumerate(positions)
    )
    fitness_scores = np.array([f[2] for f in results], dtype=np.float32)
    best_fitness_scores = fitness_scores
    global_best_position = np.zeros((n_dims))
    global_best_fitness = np.inf

    for step in range(100):
        Rp = rng.rand()
        Rg = rng.rand()

        velocities *= omega
        velocities += (best_positions - positions) * (phi_p * Rp)
        velocities += (global_best_position - positions) * (phi_g * Rg)

        positions += velocities

        results = Parallel(n_jobs=5)(
            delayed(evaluate_particle)(particle, pos)
            for particle, pos in enumerate(positions)
        )

        fitness_scores = np.array([f[2] for f in results], dtype=np.float32)
        diff_indices = np.where(fitness_scores < best_fitness_scores)[0]

        if len(diff_indices) > 0:
            best_positions[diff_indices, :] = positions[diff_indices, :]
            best_fitness_scores[diff_indices] = fitness_scores[diff_indices]

        new_best_fitness_indices = np.where(fitness_scores < global_best_fitness)[0]
        if len(new_best_fitness_indices) > 0:
            new_best_fitness_value = np.min(fitness_scores[new_best_fitness_indices])
            new_best_fitness_arg = np.argmin(fitness_scores[new_best_fitness_indices])

            global_best_fitness = new_best_fitness_value
            global_best_position = positions[new_best_fitness_arg]
