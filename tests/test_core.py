"""Tests for Core configuration, RNG determinism, and utility functions."""
import pytest
import numpy as np

from src.core.config import Config
from src.core.errors import ConfigError
from src.core.rng import RngHub
from src.core.util import wrap_angle, ramp, species_name, hsv_rgb


def test_config_defaults():
    cfg = Config()
    assert cfg.population > 0
    assert cfg.population_cap >= cfg.population
    assert cfg.world_width > 0
    assert cfg.world_height > 0
    assert cfg.dt > 0


def test_config_validation():
    cfg = Config()
    # Invalid population > cap
    with pytest.raises(ConfigError):
        cfg.update({"population": 5000, "population_cap": 2000})

    # Negative values
    with pytest.raises(ConfigError):
        cfg.dt = -0.5

    # Out of bounds range
    with pytest.raises(ConfigError):
        cfg.predator_ratio = 1.5


def test_rng_determinism():
    seed = 12345
    rng1 = RngHub(seed)
    rng2 = RngHub(seed)

    # Independent streams produce same sequence
    v1_world = rng1.world.uniform(0, 1, 100)
    v2_world = rng2.world.uniform(0, 1, 100)
    np.testing.assert_array_equal(v1_world, v2_world)

    v1_mut = rng1.mutation.normal(0, 1, 100)
    v2_mut = rng2.mutation.normal(0, 1, 100)
    np.testing.assert_array_equal(v1_mut, v2_mut)

    # Different streams produce different sequence
    assert not np.array_equal(v1_world, v1_mut)


def test_util_angles():
    assert wrap_angle(np.pi * 3) == pytest.approx(-np.pi, abs=1e-5)
    assert wrap_angle(-np.pi * 3) == pytest.approx(-np.pi, abs=1e-5)
    assert ramp(5.0, 0.0, 10.0) == 0.5


def test_util_species_name_and_color():
    names = set()
    n1 = species_name(42, 1, names)
    names.add(n1)
    n2 = species_name(42, 2, names)
    assert n1 != n2
    assert len(n1) > 0

    rgb = hsv_rgb(0.5, 0.8, 0.9)
    assert len(rgb) == 3
    assert all(0 <= c <= 255 for c in rgb)
