"""Tests for Evolutionary genetics, mutation, crossover, selection, and speciation."""
import pytest
import numpy as np

from src.core.config import Config
from src.evolution.genome import Genome, clamp_traits, N_TRAITS, TRAIT_LO, TRAIT_HI
from src.evolution.mutation import mutate
from src.evolution.crossover import crossover
from src.evolution.selection import tournament, truncation
from src.evolution.species import SpeciesManager, pair_distance
from src.neural.network import NetSpec, init_params
from src.simulation.ecosystem import Ecosystem


def test_trait_clamping():
    traits = np.array([[-100.0] * N_TRAITS, [100.0] * N_TRAITS], dtype=np.float32)
    clamp_traits(traits)
    for j in range(N_TRAITS):
        assert traits[0, j] >= TRAIT_LO[j]
        assert traits[1, j] <= TRAIT_HI[j]


def test_genome_mutation():
    spec = NetSpec(n_in=8, hidden=[6], recurrent=False)
    rng = np.random.default_rng(42)
    n = 10
    traits = np.full((n, N_TRAITS), 5.0, dtype=np.float32)
    clamp_traits(traits)
    w = init_params(spec, rng, n)
    mask = np.ones((n, spec.n_params), dtype=np.uint8)

    traits_orig = traits.copy()
    w_orig = w.copy()

    rate = np.full(n, 1.0, dtype=np.float32)
    mutate(traits, w, mask, rate, 0.2, rng, spec, 0.0, False)

    # All should have mutated with rate=1.0
    assert not np.allclose(traits, traits_orig)
    assert not np.allclose(w, w_orig)


def test_crossover_operators():
    rng = np.random.default_rng(42)
    k = 5
    tA = np.ones((k, N_TRAITS), dtype=np.float32)
    wA = np.ones((k, 20), dtype=np.float32)
    mA = np.ones((k, 20), dtype=np.uint8)

    tB = np.zeros((k, N_TRAITS), dtype=np.float32)
    wB = np.zeros((k, 20), dtype=np.float32)
    mB = np.zeros((k, 20), dtype=np.uint8)

    t_u, w_u, m_u = crossover(tA, wA, mA, tB, wB, mB, rng, mode="uniform")
    assert t_u.shape == (k, N_TRAITS)
    assert w_u.shape == (k, 20)

    t_sp, w_sp, m_sp = crossover(tA, wA, mA, tB, wB, mB, rng, mode="single_point")
    assert t_sp.shape == (k, N_TRAITS)
    assert w_sp.shape == (k, 20)


def test_selection():
    fitness = np.array([1.0, 5.0, 20.0, 50.0, 100.0], dtype=np.float32)
    rng = np.random.default_rng(42)

    # Truncation selection should strictly pick top 40% (top 2)
    trunc = truncation(fitness, fraction=0.4)
    assert set(trunc) == {3, 4}

    # Tournament selection should bias toward top performers
    tourn = tournament(rng, fitness, n=20, size=3)
    assert np.mean(fitness[tourn]) > np.mean(fitness)


def test_speciation_clustering():
    cfg = Config()
    cfg.population = 40
    eco = Ecosystem(config=cfg, seed=42)

    # Initial population creates founder species
    assert len(eco.species.species) > 0
    active_sids = eco.species.active_ids()
    assert len(active_sids) >= 1

    # Check that species have valid names and colors
    for sid in active_sids:
        sp = eco.species.species[sid]
        assert len(sp.name) > 0
        assert len(sp.color) == 3
