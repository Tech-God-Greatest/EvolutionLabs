"""Validated configuration.

All tunable parameters live in :data:`SCHEMA`.  ``Config`` loads/validates/coerces
values, supports attribute access and runtime changes, and raises
:class:`ConfigError` listing *every* problem at once (never silently ignoring a
typo or an out-of-range value).
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Iterable, Optional

from .errors import ConfigError

WORLD_TYPES = ("meadow", "desert", "forest", "arctic", "toxic", "custom", "maze")
STRATEGIES = ("mutation_selection", "genetic_algorithm", "neat")
SENSOR_GROUPS = ("basic", "social", "env", "memory")

# key: (default, kind, lo, hi, description)
# kinds: int float bool str choice:<a|b|c> intlist groups seed
SCHEMA: Dict[str, tuple] = {
    # --- run -------------------------------------------------------------
    "seed": (None, "seed", None, None, "master random seed (None = random)"),
    "dt": (0.1, "float", 0.01, 0.5, "simulated seconds per tick"),
    "generation_limit": (10000, "int", 1, 10_000_000, "stop headless runs after this generation"),
    "stats_interval": (2.0, "float", 0.1, 600.0, "simulated seconds between statistics samples"),
    "backend": ("numpy", "choice:numpy|torch", None, None, "neural inference backend (torch optional)"),
    "optimized_mode": (False, "bool", None, None, "cheaper sensing/analysis for 10k+ organisms"),
    # --- world -----------------------------------------------------------
    "world_width": (2000, "int", 200, 8000, "world width in units"),
    "world_height": (2000, "int", 200, 8000, "world height in units"),
    "world_type": ("meadow", "choice:" + "|".join(WORLD_TYPES), None, None, "procedural world preset"),
    "terrain_cell": (20, "int", 5, 100, "terrain grid resolution in units"),
    "food_density": (0.4, "float", 0.0, 4.0, "relative food carrying capacity"),
    "food_max_energy": (18.0, "float", 1.0, 500.0, "energy of a mature food item"),
    "food_grow_time": (25.0, "float", 1.0, 600.0, "seconds for food to mature"),
    "food_regrowth": (0.10, "float", 0.0, 2.0, "per-second logistic regrowth rate of the food crop"),
    "season_amplitude": (0.04, "float", 0.0, 0.5, "seasonal temperature swing"),
    "disaster_rate": (0.0003, "float", 0.0, 1.0, "random disasters per simulated second"),
    "gravity_x": (0.0, "float", -500.0, 500.0, "constant drift field (x)"),
    "gravity_y": (0.0, "float", -500.0, 500.0, "constant drift field (y)"),
    "temperature_offset": (0.0, "float", -1.0, 1.0, "global temperature offset"),
    "custom_water": (0.25, "float", 0.0, 0.8, "custom world: water fraction"),
    "custom_rock": (0.05, "float", 0.0, 0.5, "custom world: rock density"),
    "custom_tree": (0.08, "float", 0.0, 0.6, "custom world: tree density"),
    "custom_temperature": (0.5, "float", 0.0, 1.0, "custom world: base temperature"),
    "custom_temp_range": (0.2, "float", 0.0, 0.8, "custom world: temperature variation"),
    "custom_fertility": (1.0, "float", 0.1, 3.0, "custom world: fertility scale"),
    "custom_toxic": (0.05, "float", 0.0, 0.8, "custom world: toxic zone fraction"),
    "maze_size": (7, "int", 3, 25, "maze experiment: cells per side"),
    # --- population --------------------------------------------------------
    "population": (1000, "int", 1, 100_000, "initial population"),
    "population_cap": (4000, "int", 1, 200_000, "hard population cap"),
    "predator_ratio": (0.10, "float", 0.0, 1.0, "fraction of founders seeded with carnivore genes"),
    "founder_lineages": (12, "int", 1, 500, "number of ancestral genomes the population is seeded from"),
    # --- evolution ---------------------------------------------------------
    "strategy": ("mutation_selection", "choice:" + "|".join(STRATEGIES), None, None, "evolutionary algorithm"),
    "mutation_rate": (0.02, "float", 0.0, 1.0, "per-gene mutation probability"),
    "mutation_strength": (0.25, "float", 0.0, 3.0, "std-dev of weight mutations"),
    "evolvable_mutation": (False, "bool", None, None, "allow the mutability gene to evolve"),
    "structural_mutation": (0.0, "float", 0.0, 2.0, "network connection add/remove rate (x mutation rate)"),
    "reproduction_mode": ("mixed", "choice:asexual|sexual|mixed", None, None, "reproduction style"),
    "crossover_mode": ("single_point", "choice:single_point|uniform", None, None, "crossover operator"),
    "reproduction_threshold": (0.70, "float", 0.2, 0.99, "energy fraction required to reproduce"),
    "min_reproduction_age": (8.0, "float", 0.0, 1000.0, "minimum age (s) to reproduce"),
    "reproduction_cooldown": (10.0, "float", 0.0, 1000.0, "seconds between births"),
    "birth_cost": (30.0, "float", 0.0, 500.0, "energy cost of a birth"),
    "max_age": (180.0, "float", 5.0, 100_000.0, "maximum lifespan (s)"),
    "species_threshold": (0.30, "float", 0.01, 3.0, "genetic distance threshold for species"),
    "species_interval": (50, "int", 1, 5000, "ticks between species updates"),
    # --- neural ------------------------------------------------------------
    "hidden_layers": ([12, 8], "intlist", 1, 128, "hidden layer sizes"),
    "recurrent": (False, "bool", None, None, "advanced mode: recurrent memory"),
    "sensor_groups": (["basic", "social", "env", "memory"], "groups", None, None, "enabled sensor groups"),
    # --- physiology constants ----------------------------------------------
    "basal_cost": (0.8, "float", 0.0, 50.0, "basal energy use per second"),
    "move_cost": (0.0003, "float", 0.0, 1.0, "movement energy coefficient"),
    "bite_rate": (35.0, "float", 0.1, 1000.0, "max energy eaten per second"),
    "attack_damage": (12.0, "float", 0.0, 500.0, "base damage per strike"),
    "attack_cost": (1.2, "float", 0.0, 100.0, "energy cost per strike per unit attack strength"),
    # --- experiments -------------------------------------------------------
    "replay_keyframe_interval": (400.0, "float", 10.0, 1e6, "simulated seconds between replay keyframes"),
}


def _coerce(key: str, value: Any, spec: tuple) -> Any:
    default, kind, lo, hi, _ = spec
    if kind == "seed":
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError("seed must be an integer")
        v = int(value)
        if v < 0 or v > 2**63 - 1:
            raise ValueError("seed must be in [0, 2^63)")
        return v
    if kind == "int":
        if isinstance(value, bool):
            raise ValueError("expected integer")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("expected integer")
        v = int(value)
    elif kind == "float":
        if isinstance(value, bool):
            raise ValueError("expected number")
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("must be finite")
    elif kind == "bool":
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes", "on"):
                return True
            if low in ("false", "0", "no", "off"):
                return False
            raise ValueError("expected boolean")
        return bool(value)
    elif kind.startswith("choice:"):
        opts = kind.split(":", 1)[1].split("|")
        if value not in opts:
            raise ValueError(f"must be one of {opts}")
        return value
    elif kind == "intlist":
        if not isinstance(value, (list, tuple)) or len(value) < 1 or len(value) > 4:
            raise ValueError("expected list of 1-4 integers")
        out = []
        for x in value:
            if isinstance(x, bool) or int(x) != x or not (lo <= int(x) <= hi):
                raise ValueError(f"layer sizes must be integers in [{lo},{hi}]")
            out.append(int(x))
        return out
    elif kind == "groups":
        if not isinstance(value, (list, tuple)):
            raise ValueError("expected list of sensor groups")
        bad = [g for g in value if g not in SENSOR_GROUPS]
        if bad:
            raise ValueError(f"unknown sensor groups {bad}; valid: {list(SENSOR_GROUPS)}")
        if "basic" not in value:
            raise ValueError("'basic' sensor group is mandatory")
        return [g for g in SENSOR_GROUPS if g in value]
    else:  # str
        return str(value)
    if lo is not None and v < lo:
        raise ValueError(f"must be >= {lo}")
    if hi is not None and v > hi:
        raise ValueError(f"must be <= {hi}")
    return v


class Config:
    """Dictionary-backed configuration with attribute access and validation."""

    def __init__(self, values: Optional[Dict[str, Any]] = None, **overrides: Any):
        object.__setattr__(self, "_d", {k: copy.deepcopy(v[0]) for k, v in SCHEMA.items()})
        merged = dict(values or {})
        merged.update(overrides)
        self.update(merged)

    # -- access ----------------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        d = object.__getattribute__(self, "_d")
        if key in d:
            return d[key]
        raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    # -- mutation --------------------------------------------------------
    def set(self, key: str, value: Any) -> None:
        self.update({key: value})

    def update(self, values: Dict[str, Any], strict: bool = True) -> None:
        errors = []
        staged = {}
        for k, v in values.items():
            if k.startswith("_") or k in ("description", "comment"):
                continue
            if k not in SCHEMA:
                if strict:
                    errors.append(f"unknown config key '{k}'")
                continue
            try:
                staged[k] = _coerce(k, v, SCHEMA[k])
            except (ValueError, TypeError) as e:
                errors.append(f"'{k}': {e} (got {v!r})")
        if errors:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
        trial = dict(self._d)
        trial.update(staged)
        self._cross_validate(trial)
        self._d.update(staged)

    @staticmethod
    def _cross_validate(d: Dict[str, Any]) -> None:
        errs = []
        if d["population"] > d["population_cap"]:
            errs.append(f"population ({d['population']}) exceeds population_cap ({d['population_cap']})")
        if d["birth_cost"] >= 100 * 5:
            errs.append("birth_cost is unreasonably large")
        if d["world_type"] != "maze" and d["world_width"] / d["terrain_cell"] > 800:
            errs.append("terrain grid too fine (>800 cells wide); increase terrain_cell")
        if errs:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errs))

    # -- io --------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._d)

    def copy(self) -> "Config":
        c = Config()
        object.__setattr__(c, "_d", copy.deepcopy(self._d))
        return c

    @classmethod
    def from_file(cls, path: str, overrides: Optional[Dict[str, Any]] = None) -> "Config":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ConfigError(f"config file not found: {path}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"config file {path} is not valid JSON: {e}")
        if not isinstance(data, dict):
            raise ConfigError(f"config file {path} must contain a JSON object")
        data.update(overrides or {})
        return cls(data)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._d, f, indent=2)

    def __repr__(self) -> str:
        return f"Config({self._d})"


def default_config_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "config", "default_config.json"))


def load_default() -> Config:
    p = default_config_path()
    return Config.from_file(p) if os.path.exists(p) else Config()
