"""Particle Swarm Optimization (PSO)."""
# author Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
import numpy.typing as npt
from typing import Any, Union
from joblib import Parallel, delayed
from dataclasses import dataclass, field
from typing import List


@dataclass
class Swarm:
    """Swarm."""

    n_particles: int = field(default_factory=int)
    n_dims: int = 0
    omega: float = field(default_factory=float)
    phi: list[float] = field(default_factory=list)

    random_state: np.random.RandomState = field(default=np.random.RandomState(0))

    positions: npt.NDArray = field(init=False, repr=False)
    velocities: npt.NDArray = field(init=False, repr=False)
    # fitness_scores: npt.NDArray = field(init=False, repr=False)

    best_positions: npt.NDArray = field(init=False, repr=False)
    best_fitness_scores: npt.NDArray = field(init=False, repr=False)

    global_best_positions: npt.NDArray = field(init=False, repr=False)
    global_best_fitness_scores: npt.NDArray = field(init=False, repr=False)

    def prepare(self, initial_positions: npt.NDArray) -> None:
        """."""
        self.n_dims = np.shape(initial_positions)[1]
        self.positions = initial_positions
        self.velocities = self.random_state.rand(self.n_particles, self.n_dims)
        # self.fitness_scores = np.zeros([self.n_particles])
        self.best_positions = self.positions
        self.best_fitness_scores = np.zeros([self.n_particles, 1], dtype=np.float32)
        self.global_best_position = np.zeros([self.n_dims])
        self.global_best_fitness = np.inf

    def set_eval_func(self, func):
        """Set callback evaluation function."""
        self.__func = func

    def step(self, init: bool = False) -> None:
        """Step."""

        if init:
            results = Parallel(n_jobs=5)(
                delayed(self.__func)(particle, pos)
                for particle, pos in enumerate(self.positions)
            )

            fitness_scores = np.array([row[2] for row in results], dtype=np.float32)
            self.best_fitness_scores = fitness_scores

            return

        r_p = self.random_state.rand()
        r_g = self.random_state.rand()

        phi_p, phi_g = self.phi

        self.velocities *= self.omega
        self.velocities += (self.best_positions - self.positions) * (phi_p * r_p)
        self.velocities += (self.global_best_position - self.positions) * (phi_g * r_g)

        self.positions += self.velocities

        results = Parallel(n_jobs=5)(
            delayed(self.__func)(particle, pos)
            for particle, pos in enumerate(self.positions)
        )

        fitness_scores = np.array([f[2] for f in results], dtype=np.float32)
        diff_indices = np.where(fitness_scores < self.best_fitness_scores)[0]

        if len(diff_indices) > 0:
            self.best_positions[diff_indices, :] = self.positions[diff_indices, :]
            self.best_fitness_scores[diff_indices] = fitness_scores[diff_indices]

        new_best_fitness_indices = np.where(fitness_scores < self.global_best_fitness)[
            0
        ]

        if len(new_best_fitness_indices) > 0:
            new_best_fitness_value = np.min(fitness_scores[new_best_fitness_indices])
            new_best_fitness_arg = np.argmin(fitness_scores[new_best_fitness_indices])

            self.global_best_fitness = new_best_fitness_value
            self.global_best_position = self.positions[new_best_fitness_arg]
