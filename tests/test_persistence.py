"""Tests for State Persistence (JSON / Gzip serialization and restoration)."""
import os
import tempfile
import pytest
import numpy as np

from src.core.config import Config
from src.core.errors import SaveError
from src.persistence.save import save_ecosystem
from src.persistence.load import load_ecosystem
from src.simulation.ecosystem import Ecosystem


def test_save_and_load_roundtrip():
    cfg = Config()
    cfg.population = 50
    cfg.population_cap = 200
    eco = Ecosystem(config=cfg, seed=123)

    for _ in range(15):
        eco.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_save.json")
        save_ecosystem(eco, save_path)

        assert os.path.exists(save_path)
        loaded_eco = load_ecosystem(save_path)

        assert loaded_eco.tick == eco.tick
        assert loaded_eco.time == pytest.approx(eco.time, abs=1e-5)
        assert loaded_eco.org.n == eco.org.n
        assert loaded_eco.food.n == eco.food.n

        np.testing.assert_array_equal(loaded_eco.org.id[:loaded_eco.org.n], eco.org.id[:eco.org.n])
        np.testing.assert_allclose(loaded_eco.org.pos[:loaded_eco.org.n], eco.org.pos[:eco.org.n], atol=1e-5)


def test_gzip_compression_roundtrip():
    cfg = Config()
    cfg.population = 40
    eco = Ecosystem(config=cfg, seed=456)
    for _ in range(5):
        eco.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_save.json.gz")
        save_ecosystem(eco, save_path, compress=True)

        assert os.path.exists(save_path)
        loaded_eco = load_ecosystem(save_path)

        assert loaded_eco.tick == eco.tick
        assert loaded_eco.org.n == eco.org.n


def test_corrupt_save_rejection():
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupt_path = os.path.join(tmpdir, "bad.json")
        with open(corrupt_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json content ::: 123 }")

        with pytest.raises(SaveError):
            load_ecosystem(corrupt_path)


def test_invalid_format_header():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_path = os.path.join(tmpdir, "wrong_format.json")
        with open(fake_path, "w", encoding="utf-8") as f:
            f.write('{"format": "wrong_format", "version": 1}')

        with pytest.raises(SaveError):
            load_ecosystem(fake_path)
