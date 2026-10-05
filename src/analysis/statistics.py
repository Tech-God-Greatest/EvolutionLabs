"""Real-time Statistics Recorder and Population Dashboard.

Collects time-series metrics at regular simulation intervals:
- Total population, predator count, herbivore count
- Food items and total ecosystem food energy
- Average fitness, lifespan, speed, vision, size, metabolism
- Genetic diversity (mean pairwise trait distance)
- Birth rate and death rate
- Active and extinct species counts
- Energy balance validation

Maintains sliding history buffers for graph rendering and supports CSV/JSON export.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional
import numpy as np

from ..evolution.genome import (
    G_SPEED,
    G_VIS,
    G_SIZE,
    G_METAB,
    G_DIET,
    G_ATTACK,
    TRAIT_RANGE,
)
from .behavior import classify_population


class StatsRecorder:
    def __init__(self, eco, max_samples: int = 4000):
        self.eco = eco
        self.max_samples = max_samples
        self.interval = float(eco.cfg.stats_interval)
        self.last_sample_time = -1e9
        self.last_sample_tick = 0
        self.history: Dict[str, List[float]] = {
            "time": [],
            "generation": [],
            "population": [],
            "predators": [],
            "herbivores": [],
            "food_count": [],
            "food_energy": [],
            "avg_fitness": [],
            "avg_lifespan": [],
            "genetic_diversity": [],
            "avg_speed": [],
            "avg_vision": [],
            "avg_size": [],
            "avg_metabolism": [],
            "avg_attack": [],
            "birth_rate": [],
            "death_rate": [],
            "active_species": [],
            "total_extinctions": [],
            "total_energy": [],
        }
        self.species_history: Dict[int, List[int]] = {}
        self.prev_births = 0
        self.prev_deaths = 0

    def maybe_record(self) -> bool:
        if self.eco.time - self.last_sample_time >= self.interval:
            self.record()
            return True
        return False

    def record(self, force: bool = False) -> None:
        eco = self.eco
        if not force and (eco.time - self.last_sample_time < self.interval * 0.95):
            return

        dt = max(eco.time - self.last_sample_time, eco.cfg.dt)
        if self.last_sample_time < 0:
            dt = eco.cfg.dt
        self.last_sample_time = eco.time
        self.last_sample_tick = eco.tick

        s = eco.org
        n = s.n
        T = s.traits[:n] if n > 0 else np.zeros((0, len(TRAIT_RANGE)), dtype=np.float32)

        # Classify behaviors periodically
        classify_population(eco)

        # Predator vs Herbivore counts
        if n > 0:
            diet = T[:, G_DIET]
            preds = int(np.count_nonzero(diet > 0.5))
            herbs = n - preds
            avg_fit = float(eco.fitness().mean())
            avg_spd = float(T[:, G_SPEED].mean())
            avg_vis = float(T[:, G_VIS].mean())
            avg_sz = float(T[:, G_SIZE].mean())
            avg_met = float(T[:, G_METAB].mean())
            avg_atk = float(T[:, G_ATTACK].mean())

            # Genetic diversity: mean standard deviation across normalized traits
            norm_traits = T / TRAIT_RANGE
            diversity = float(np.mean(np.std(norm_traits, axis=0)))
        else:
            preds, herbs = 0, 0
            avg_fit = 0.0
            avg_spd, avg_vis, avg_sz, avg_met, avg_atk = 0.0, 0.0, 0.0, 0.0, 0.0
            diversity = 0.0

        # Lifespans
        avg_life = float(np.mean(eco.recent_lifespans[-50:])) if eco.recent_lifespans else 0.0

        # Rates per simulated second
        births = eco.births_total - self.prev_births
        deaths = eco.deaths_total - self.prev_deaths
        self.prev_births = eco.births_total
        self.prev_deaths = eco.deaths_total
        birth_rate = births / dt
        death_rate = deaths / dt

        # Record values
        h = self.history
        h["time"].append(round(eco.time, 2))
        h["generation"].append(round(eco.mean_generation, 2))
        h["population"].append(n)
        h["predators"].append(preds)
        h["herbivores"].append(herbs)
        h["food_count"].append(eco.food.n)
        h["food_energy"].append(round(eco.food.total_energy(), 1))
        h["avg_fitness"].append(round(avg_fit, 3))
        h["avg_lifespan"].append(round(avg_life, 2))
        h["genetic_diversity"].append(round(diversity, 4))
        h["avg_speed"].append(round(avg_spd, 2))
        h["avg_vision"].append(round(avg_vis, 2))
        h["avg_size"].append(round(avg_sz, 2))
        h["avg_metabolism"].append(round(avg_met, 2))
        h["avg_attack"].append(round(avg_atk, 2))
        h["birth_rate"].append(round(birth_rate, 2))
        h["death_rate"].append(round(death_rate, 2))
        h["active_species"].append(len(eco.species.active_ids()))
        h["total_extinctions"].append(eco.species.total_extinct)
        h["total_energy"].append(round(eco.total_energy(), 1))

        # Trim history if exceeding max_samples
        if len(h["time"]) > self.max_samples:
            for k in h:
                h[k] = h[k][-self.max_samples:]

        # Update event detector
        eco.events.update()

    def get_series(self, metric: str) -> np.ndarray:
        return np.asarray(self.history.get(metric, []), dtype=np.float32)

    def current_summary(self) -> dict:
        h = self.history
        if not h["time"]:
            return {}
        return {k: h[k][-1] for k in h}

    def export_csv(self, path: str) -> None:
        h = self.history
        keys = list(h.keys())
        n_rows = len(h["time"])
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(n_rows):
                writer.writerow([h[k][i] for k in keys])

    def state_dict(self) -> dict:
        return {
            "interval": self.interval,
            "last_sample_time": self.last_sample_time,
            "last_sample_tick": self.last_sample_tick,
            "history": {k: list(v) for k, v in self.history.items()},
            "prev_births": self.prev_births,
            "prev_deaths": self.prev_deaths,
        }

    def load_state_dict(self, d: dict) -> None:
        self.interval = float(d.get("interval", self.interval))
        self.last_sample_time = float(d.get("last_sample_time", -1e9))
        self.last_sample_tick = int(d.get("last_sample_tick", 0))
        self.history = {k: list(v) for k, v in d.get("history", {}).items()}
        self.prev_births = int(d.get("prev_births", 0))
        self.prev_deaths = int(d.get("prev_deaths", 0))
