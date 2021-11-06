import numpy as np

from particles import Swarm


def eval_func(particle: int, position: list[float]) -> Union[int, list[float], float]:
    """DeJong's function."""
    score = position[0] ** 2 + position[1] ** 2
    return particle, position, score


if __name__ == "__main__":
    rng = np.random.RandomState(0)

    # Data distribution
    n = 400000
    DATA = np.zeros((n, 3))
    i = 0
    for x in np.arange(-1.0, 1.0, 0.0125):
        for y in np.arange(-1.0, 1.0, 0.0125):
            z = x ** 2 + y ** 2
            DATA[i, :] = x, z, y
            i += 1

    DRAW_DATA = np.unique(DATA, axis=0)

    n_particles = 100
    initial_indices = rng.choice(DRAW_DATA.shape[0], size=n_particles, replace=False)
    positions = DRAW_DATA[initial_indices, :]

    # swarm settings
    sw = Swarm(n_particles=100, omega=0.25, phi=[0.1, 0.01], random_state=rng)
    sw.prepare(positions)
    sw.set_eval_func(eval_func)
    sw.step(init=True)
