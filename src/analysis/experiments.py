"""Evolutionary Experiment Framework and Presets.

Implements all specialized experiment modes (§15 - §21, §36 - §39):
1. Predator-Prey population cycles
2. Genetic Bottleneck & diversity recovery
3. Mutation rate sweeps & evolutionary stability
4. Environmental Pressure timeline
5. Evolutionary Arms Race (predator vs prey speed/agility)
6. Maze navigation (generational neuroevolution)
7. "What If?" A/B scenario comparison
8. Challenge Mode (survival, diversity, speed, scarcity)
9. Neuroevolution Algorithm Comparison (GA vs NEAT vs Mutation+Selection)
"""
from __future__ import annotations

import copy
import os
import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from ..core.config import Config
from ..evolution.evolutionary_algorithms import make_generational
from ..evolution.genome import G_SPEED, G_VIS, G_ATTACK, G_DIET
from ..simulation.ecosystem import Ecosystem
from .report import ReportGenerator


class ExperimentPreset:
    """Descriptor for an experiment preset."""

    def __init__(
        self,
        name: str,
        title: str,
        description: str,
        config_overrides: dict,
        setup_fn: Optional[Callable[[Ecosystem], None]] = None,
        step_hook: Optional[Callable[[Ecosystem], None]] = None,
    ):
        self.name = name
        self.title = title
        self.description = description
        self.config_overrides = config_overrides
        self.setup_fn = setup_fn
        self.step_hook = step_hook


# --- Preset Setup Hooks -------------------------------------------------------------
def setup_bottleneck(eco: Ecosystem) -> None:
    # Schedule bottleneck disaster at generation 50
    eco.env.add_timeline(
        trigger={"generation": 50},
        action="bottleneck",
        params={"keep": 10},
        label="Genetic Bottleneck Event: population collapsed to 10 survivors",
    )


def setup_environmental_timeline(eco: Ecosystem) -> None:
    # Schedule staged environmental pressures
    eco.env.add_timeline(
        trigger={"generation": 50},
        action="food_scale",
        params={"value": 0.5},
        label="Environmental Shift: Food availability reduced by 50%",
    )
    eco.env.add_timeline(
        trigger={"generation": 100},
        action="temperature_offset",
        params={"value": 0.45},
        label="Global Warming: Climate temperature increased by +0.45",
    )
    eco.env.add_timeline(
        trigger={"generation": 150},
        action="spawn_predators",
        params={"count": 25},
        label="Invasive Apex Species Introduced",
    )


def setup_arms_race(eco: Ecosystem) -> None:
    # Start with equal balanced pressures to drive speed coevolution
    pass


EXPERIMENT_PRESETS: Dict[str, ExperimentPreset] = {
    "predator_prey": ExperimentPreset(
        name="predator_prey",
        title="Predator-Prey Population Cycles",
        description="Tracks natural Lotka-Volterra oscillations between carnivores, herbivores, and food crops.",
        config_overrides={
            "population": 1200,
            "population_cap": 4000,
            "predator_ratio": 0.15,
            "food_density": 0.55,
            "world_type": "meadow",
        },
    ),
    "bottleneck": ExperimentPreset(
        name="bottleneck",
        title="Genetic Bottleneck & Recovery",
        description="Forces population down to 10 founders to witness genetic diversity collapse and subsequent speciation recovery.",
        config_overrides={
            "population": 1000,
            "population_cap": 3000,
            "predator_ratio": 0.05,
            "mutation_rate": 0.03,
        },
        setup_fn=setup_bottleneck,
    ),
    "environmental_timeline": ExperimentPreset(
        name="environmental_timeline",
        title="Environmental Stress Timeline",
        description="Sequential environmental pressures (scarcity, heating, invasive predators) testing ecosystem adaptability.",
        config_overrides={
            "population": 1000,
            "population_cap": 3500,
            "food_density": 0.6,
        },
        setup_fn=setup_environmental_timeline,
    ),
    "arms_race": ExperimentPreset(
        name="arms_race",
        title="Evolutionary Arms Race",
        description="Continuous coevolution between predator pursuit capabilities and prey evasion traits.",
        config_overrides={
            "population": 1200,
            "predator_ratio": 0.20,
            "mutation_rate": 0.03,
            "attack_damage": 15.0,
        },
        setup_fn=setup_arms_race,
    ),
    "maze": ExperimentPreset(
        name="maze",
        title="Evolutionary Maze Navigation",
        description="Neural organisms evolve navigation and spatial orientation purely through sensory inputs and reward.",
        config_overrides={
            "world_type": "maze",
            "maze_size": 7,
            "population": 300,
            "population_cap": 400,
            "sensor_groups": ["basic", "social", "env", "memory"],
            "strategy": "neat",
        },
    ),
}


# --- High-Performance Experiment Runner ----------------------------------------------
class ExperimentRunner:
    """Executes experiments headlessly with progress callbacks and reporting."""

    def __init__(
        self,
        config: Config,
        preset_name: Optional[str] = None,
        generations: int = 200,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
    ):
        self.preset_name = preset_name
        self.generations = generations
        self.seed = seed
        self.output_dir = output_dir or os.path.join("experiments", preset_name or "custom")

        cfg = config.copy()
        preset = EXPERIMENT_PRESETS.get(preset_name) if preset_name else None
        if preset:
            cfg.update(preset.config_overrides)
        if seed is not None:
            cfg.seed = seed

        self.cfg = cfg
        self.preset = preset
        self.eco = Ecosystem(config=cfg, seed=seed, init=True)
        if preset and preset.setup_fn:
            preset.setup_fn(self.eco)

    def run(self, progress_callback: Optional[Callable[[float, int, int], None]] = None) -> Ecosystem:
        """Run simulation until generation limit or organism extinction."""
        start_wall = time.time()
        eco = self.eco
        target_gen = self.generations
        step = 0

        while eco.mean_generation < target_gen and eco.org.n > 0:
            eco.step()
            step += 1

            if self.preset and self.preset.step_hook:
                self.preset.step_hook(eco)

            if step % 50 == 0 and progress_callback:
                frac = min(1.0, eco.mean_generation / target_gen)
                progress_callback(frac, int(eco.mean_generation), eco.org.n)

        # Final record and export
        eco.stats.record(force=True)
        elapsed = time.time() - start_wall

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            report = ReportGenerator(eco)
            report.export_full_report(self.output_dir)

        return eco


# --- What-If? Scenario Comparative Runner (§36) --------------------------------------
class WhatIfComparison:
    """Runs World A (baseline) and World B (modified) in parallel and contrasts outcomes."""

    def __init__(
        self,
        base_config: Config,
        modifications: dict,
        generations: int = 150,
        seed: int = 42,
    ):
        self.generations = generations

        # World A: Baseline
        cfg_a = base_config.copy()
        cfg_a.seed = seed
        self.eco_a = Ecosystem(config=cfg_a, seed=seed)

        # World B: Changed environment (same seed for identical initial conditions)
        cfg_b = base_config.copy()
        cfg_b.seed = seed
        cfg_b.update(modifications)
        self.eco_b = Ecosystem(config=cfg_b, seed=seed)
        self.modifications = modifications

    def run(self, progress_callback: Optional[Callable[[float], None]] = None) -> dict:
        target = self.generations
        step = 0
        while (self.eco_a.mean_generation < target or self.eco_b.mean_generation < target) and (self.eco_a.org.n > 0 or self.eco_b.org.n > 0):
            if self.eco_a.mean_generation < target and self.eco_a.org.n > 0:
                self.eco_a.step()
            if self.eco_b.mean_generation < target and self.eco_b.org.n > 0:
                self.eco_b.step()
            step += 1
            if step % 50 == 0 and progress_callback:
                max_gen = max(self.eco_a.mean_generation, self.eco_b.mean_generation)
                progress_callback(min(1.0, max_gen / target))

        self.eco_a.stats.record(force=True)
        self.eco_b.stats.record(force=True)

        return {
            "world_a": {
                "final_pop": self.eco_a.org.n,
                "species_count": len(self.eco_a.species.active_ids()),
                "total_species": self.eco_a.species.total_created,
                "avg_speed": float(self.eco_a.org.traits[:self.eco_a.org.n, G_SPEED].mean()) if self.eco_a.org.n else 0.0,
                "avg_vision": float(self.eco_a.org.traits[:self.eco_a.org.n, G_VIS].mean()) if self.eco_a.org.n else 0.0,
            },
            "world_b": {
                "final_pop": self.eco_b.org.n,
                "species_count": len(self.eco_b.species.active_ids()),
                "total_species": self.eco_b.species.total_created,
                "avg_speed": float(self.eco_b.org.traits[:self.eco_b.org.n, G_SPEED].mean()) if self.eco_b.org.n else 0.0,
                "avg_vision": float(self.eco_b.org.traits[:self.eco_b.org.n, G_VIS].mean()) if self.eco_b.org.n else 0.0,
            },
            "modifications": self.modifications,
        }


# --- Generational Maze Runner (§38) --------------------------------------------------
class MazeExperimentRunner:
    """Runs generational neuroevolution inside the procedural maze."""

    def __init__(self, config: Config, generations: int = 50, cohort_time: float = 30.0):
        cfg = config.copy()
        cfg.world_type = "maze"
        self.cfg = cfg
        self.generations = generations
        self.cohort_time = cohort_time
        self.eco = Ecosystem(config=cfg, init=True)
        self.history: List[dict] = []

    def run(self, progress_callback: Optional[Callable[[int, float, float], None]] = None) -> List[dict]:
        eco = self.eco
        alg = make_generational(eco.cfg.strategy, eco.spec, eco.cfg, eco.rng.mutation)

        for gen in range(self.generations):
            # Run one cohort for cohort_time simulated seconds
            t0 = eco.time
            while (eco.time - t0) < self.cohort_time:
                eco.step()

            # Evaluate fitness: based on final distance to goal + finish speed
            gx, gy = eco.world.goal
            s = eco.org
            n = s.n
            dist = np.hypot(s.pos[:n, 0] - gx, s.pos[:n, 1] - gy)
            finished = s.finish_time[:n] >= 0
            # Fitness: high for finishing fast, otherwise inverse distance
            fit = np.where(finished, 100.0 + (self.cohort_time - s.finish_time[:n]) * 5.0, 100.0 / (dist + 1.0))

            fin_count = int(np.count_nonzero(finished))
            best_time = float(s.finish_time[:n][finished].min()) if fin_count > 0 else -1.0
            avg_dist = float(dist.mean())

            rec = {
                "generation": gen + 1,
                "finishers": fin_count,
                "best_time": best_time,
                "avg_distance": avg_dist,
            }
            self.history.append(rec)

            if progress_callback:
                progress_callback(gen + 1, best_time, avg_dist)

            # Evolve next generation
            nT, nW, nM, _ = alg.next_generation(s.traits[:n], s.w[:n], s.mask[:n], fit)
            next_gens = np.full(len(nT), gen + 1, dtype=np.int32)
            eco.maze_load_generation(nT, nW, nM, next_gens)

        return self.history
