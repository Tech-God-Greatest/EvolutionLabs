"""Tests for Evolutionary Experiments, What-If comparisons, and Event detection."""
import os
import tempfile
import pytest
import numpy as np

from src.core.config import Config
from src.analysis.experiments import EXPERIMENT_PRESETS, ExperimentRunner, WhatIfComparison
from src.analysis.events import EventDetector
from src.analysis.statistics import StatsRecorder
from src.simulation.ecosystem import Ecosystem


def test_presets_registered():
    assert "predator_prey" in EXPERIMENT_PRESETS
    assert "bottleneck" in EXPERIMENT_PRESETS
    assert "environmental_timeline" in EXPERIMENT_PRESETS
    assert "arms_race" in EXPERIMENT_PRESETS
    assert "maze" in EXPERIMENT_PRESETS


def test_experiment_runner():
    cfg = Config()
    cfg.population = 40
    cfg.population_cap = 150
    preset = EXPERIMENT_PRESETS["predator_prey"]

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(config=cfg, preset_name="predator_prey", generations=0.2, seed=777, output_dir=tmpdir)
        eco = runner.run()

        assert eco.tick > 0
        assert len(eco.stats.history["population"]) > 0


def test_what_if_comparison():
    base_cfg = Config()
    base_cfg.population = 40
    base_cfg.population_cap = 150

    # Modify world B: higher mutation rate
    comparison = WhatIfComparison(
        base_config=base_cfg,
        modifications={"mutation_rate": 0.15},
        generations=0.2,
        seed=101,
    )
    results = comparison.run()

    assert "world_a" in results
    assert "world_b" in results
    assert results["world_a"]["final_pop"] >= 0
    assert results["world_b"]["final_pop"] >= 0


def test_event_logging():
    cfg = Config()
    cfg.population = 40
    eco = Ecosystem(config=cfg, seed=555)

    # Initial setup logs speciation events
    assert len(eco.events.events) > 0
    first_ev = eco.events.events[0]
    assert first_ev.kind in ("species_emerged", "speciation", "boom", "behavior", "init")


def test_stats_export_csv():
    cfg = Config()
    cfg.population = 30
    eco = Ecosystem(config=cfg, seed=888)

    for _ in range(10):
        eco.step()
    eco.stats.record(force=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "telemetry.csv")
        eco.stats.export_csv(csv_path)

        assert os.path.exists(csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) >= 2  # header + at least 1 record
            assert "population" in lines[0]
