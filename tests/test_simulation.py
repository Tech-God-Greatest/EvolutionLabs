"""Tests for World physics, SpatialGrid, Ecosystem dynamics, and Energy conservation."""
import pytest
import numpy as np

from src.core.config import Config
from src.simulation.spatial_grid import SpatialGrid
from src.simulation.world import World, ROCK
from src.simulation.ecosystem import Ecosystem


def test_spatial_grid_accuracy():
    # Verify spatial grid finds the same neighbors as exhaustive brute-force search
    grid = SpatialGrid(width=500.0, height=500.0, cell=50.0)
    rng = np.random.default_rng(42)
    n = 200
    pos = rng.uniform(20.0, 480.0, (n, 2)).astype(np.float32)
    grid.build(pos)

    query_pos = np.array([250.0, 250.0], dtype=np.float32)
    query_radius = 45.0

    # Brute-force candidates
    bf_dists = np.hypot(pos[:, 0] - query_pos[0], pos[:, 1] - query_pos[1])
    bf_matches = set(np.flatnonzero(bf_dists <= query_radius))

    # SpatialGrid candidates
    grid_matches = set(grid.query_radius(pos, query_pos, query_radius))

    assert bf_matches == grid_matches


def test_world_boundaries_and_blocked():
    cfg = Config()
    cfg.world_width = 300.0
    cfg.world_height = 300.0
    world = World(cfg)
    rng = np.random.default_rng(42)
    world.generate(rng)

    # Obstacle query
    blocked = world.blocked_at(np.array([10.0, -5.0, 400.0]), np.array([10.0, 10.0, 400.0]))
    assert blocked[1]  # outside boundary is blocked
    assert blocked[2]  # outside boundary is blocked


def test_ecosystem_step_and_energy_conservation():
    cfg = Config()
    cfg.population = 100
    cfg.population_cap = 500
    cfg.seed = 42
    eco = Ecosystem(config=cfg, seed=42)

    initial_pop = eco.org.n
    assert initial_pop == 100

    # Step simulation 50 ticks
    for _ in range(50):
        eco.step()

    assert eco.tick == 50
    assert eco.org.n > 0
    assert eco.food.n > 0

    # Energy balance check: strictly conserved (accounting for all calories in/out/stored)
    assert eco.energy_balance_error() < 1e-1


def test_simulation_determinism():
    # Two ecosystems started with the same seed must produce identical states after 30 ticks
    seed = 9999
    cfg = Config()
    cfg.population = 60
    cfg.population_cap = 200

    eco1 = Ecosystem(config=cfg, seed=seed)
    eco2 = Ecosystem(config=cfg, seed=seed)

    for _ in range(30):
        eco1.step()
        eco2.step()

    assert eco1.tick == eco2.tick
    assert eco1.org.n == eco2.org.n
    np.testing.assert_array_equal(eco1.org.id[:eco1.org.n], eco2.org.id[:eco2.org.n])
    np.testing.assert_allclose(eco1.org.pos[:eco1.org.n], eco2.org.pos[:eco2.org.n], atol=1e-5)
    np.testing.assert_allclose(eco1.org.energy[:eco1.org.n], eco2.org.energy[:eco2.org.n], atol=1e-4)
