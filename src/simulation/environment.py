"""Environmental conditions, disasters and scheduled pressure.

Effective conditions (temperature offset, food availability, water level, ...)
are computed every tick from *base* settings (user / timeline controlled) plus
all currently active disasters, which fade out smoothly.  Disasters never target
organisms individually: they change the world and organisms are harmed
according to their own traits (tolerance, size, speed, metabolism, ...).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .organism import C_DISASTER

DISASTERS: Dict[str, dict] = {
    "drought": dict(label="Drought", duration=90.0),
    "flood": dict(label="Flood", duration=60.0),
    "heatwave": dict(label="Heatwave", duration=80.0),
    "ice_age": dict(label="Ice age", duration=140.0),
    "meteor": dict(label="Meteor impact", duration=60.0),
    "toxic": dict(label="Toxic contamination", duration=70.0),
    "food_collapse": dict(label="Food collapse", duration=100.0),
}
RANDOM_DISASTER_WEIGHTS = {"drought": 2, "flood": 2, "heatwave": 2, "ice_age": 1, "meteor": 1, "toxic": 2, "food_collapse": 2}


class Environment:
    def __init__(self, eco):
        self.eco = eco
        cfg = eco.cfg
        self.base_temp_offset = float(cfg.temperature_offset)
        self.base_food_mult = 1.0
        self.base_water_offset = 0
        self.gravity = np.array([cfg.gravity_x, cfg.gravity_y], dtype=np.float32)
        self.disasters: List[dict] = []
        self.timeline: List[dict] = []
        self.history: List[dict] = []
        # effective values (recomputed each update)
        self.temp_offset = self.base_temp_offset
        self.food_mult = 1.0
        self.fert_mult = 1.0
        self.season_temp = 0.0
        self.flood_level = 0.0
        self.disasters_enabled = True

    # ------------------------------------------------------------------
    def add_timeline(self, trigger: dict, action: str, params: Optional[dict] = None, label: str = "") -> None:
        self.timeline.append({"trigger": dict(trigger), "action": action, "params": dict(params or {}),
                              "label": label or action, "done": False})

    def trigger(self, kind: str, strength: float = 1.0) -> dict:
        eco = self.eco
        if kind not in DISASTERS:
            raise ValueError(f"unknown disaster '{kind}'")
        spec = DISASTERS[kind]
        d = {"kind": kind, "t_left": spec["duration"], "duration": spec["duration"], "strength": float(strength),
             "started": eco.time}
        rng = eco.rng.events
        if kind == "meteor":
            w = eco.world
            d["x"] = float(rng.uniform(0.1, 0.9) * w.width)
            d["y"] = float(rng.uniform(0.1, 0.9) * w.height)
            d["radius"] = float(min(w.width, w.height) * 0.11 * strength)
            self._meteor(d)
        elif kind == "toxic":
            self._toxic(d)
        elif kind == "food_collapse":
            self._remove_food_fraction(0.85 * min(1.0, strength))
        elif kind == "drought":
            self._remove_food_fraction(0.35 * min(1.0, strength))
        self.disasters.append(d)
        rec = {"kind": kind, "label": spec["label"], "time": eco.time, "generation": eco.mean_generation,
               "pop_before": eco.org.n}
        self.history.append(rec)
        eco.log_event("disaster", f"{spec['label']} struck the ecosystem", severity=2,
                      data={"kind": kind, "pop_before": eco.org.n})
        eco.fx.append(("disaster", kind, d.get("x", 0.0), d.get("y", 0.0), d.get("radius", 0.0)))
        return d

    # ------------------------------------------------------------------
    def _remove_food_fraction(self, frac: float) -> None:
        eco = self.eco
        f = eco.food
        if f.n == 0:
            return
        mask = eco.rng.events.random(f.n) < frac
        lost = f.remove(mask)
        eco.ledger.add("out_removed", lost)

    def _meteor(self, d: dict) -> None:
        eco = self.eco
        s = eco.org
        n = s.n
        cx, cy, R = d["x"], d["y"], d["radius"]
        if n:
            dist = np.hypot(s.pos[:n, 0] - cx, s.pos[:n, 1] - cy)
            size = s.traits[:n, 5]
            dmg = np.zeros(n, dtype=np.float32)
            inner = dist < 0.5 * R
            outer = (dist < R) & ~inner
            dmg[inner] = 1000.0
            dmg[outer] = 120.0 * (1.0 - dist[outer] / R) * np.sqrt(size[outer] / 5.0)
            s.health[:n] -= dmg
            s.cause_dmg[:n, C_DISASTER] += dmg
        f = eco.food
        if f.n:
            m = np.hypot(f.pos[: f.n, 0] - cx, f.pos[: f.n, 1] - cy) < R
            eco.ledger.add("out_removed", f.remove(m))
        w = eco.world
        ys, xs = np.mgrid[0:w.gh, 0:w.gw]
        hot = (((xs + 0.5) * w.cell - cx) ** 2 + ((ys + 0.5) * w.cell - cy) ** 2) < (1.15 * R) ** 2
        w.danger_extra[hot] += 2.5

    def _toxic(self, d: dict) -> None:
        eco = self.eco
        w = eco.world
        from .world import fbm
        blob = fbm(eco.rng.events, w.gh, w.gw, 3, 3)
        mask = blob > np.quantile(blob, 1.0 - 0.35 * min(1.0, d["strength"]))
        w.danger_extra[mask] += 3.0 * d["strength"]

    # ------------------------------------------------------------------
    def update(self, dt: float) -> None:
        eco = self.eco
        w = eco.world
        # random disasters
        cfg = eco.cfg
        if self.disasters_enabled and cfg.disaster_rate > 0 and eco.world.world_type != "maze":
            if eco.rng.events.random() < cfg.disaster_rate * dt:
                kinds = list(RANDOM_DISASTER_WEIGHTS)
                p = np.array([RANDOM_DISASTER_WEIGHTS[k] for k in kinds], dtype=float)
                kind = kinds[int(eco.rng.events.choice(len(kinds), p=p / p.sum()))]
                self.trigger(kind, float(eco.rng.events.uniform(0.7, 1.2)))
        # timeline
        for e in self.timeline:
            if e["done"]:
                continue
            trig = e["trigger"]
            hit = ("generation" in trig and eco.mean_generation >= trig["generation"]) or \
                  ("time" in trig and eco.time >= trig["time"])
            if hit:
                e["done"] = True
                eco.apply_action(e["action"], **e["params"])
                eco.log_event("pressure", e["label"], severity=2, data={"action": e["action"], **e["params"]})
        # effective modifiers
        temp = self.base_temp_offset
        food = self.base_food_mult
        fert = 1.0
        water = self.base_water_offset
        flood_dmg_level = 0.0
        for d in self.disasters:
            d["t_left"] -= dt
            f = min(1.0, max(d["t_left"], 0.0) / (0.3 * d["duration"])) * d["strength"]
            k = d["kind"]
            if k == "drought":
                fert *= 1 - 0.65 * min(f, 1)
                temp += 0.10 * f
                water -= int(round(4 * min(f, 1)))
            elif k == "flood":
                water += int(round(3 * min(f, 1)))
                flood_dmg_level = max(flood_dmg_level, f)
            elif k == "heatwave":
                temp += 0.35 * f
            elif k == "ice_age":
                temp -= 0.38 * f
                fert *= 1 - 0.5 * min(f, 1)
            elif k == "food_collapse":
                food *= 1 - 0.85 * min(f, 1)
        self.disasters = [d for d in self.disasters if d["t_left"] > 0]
        self.season_temp = cfg.season_amplitude * math.sin(2 * math.pi * eco.time / 600.0)
        self.temp_offset = temp + self.season_temp
        self.food_mult = max(0.0, food)
        self.fert_mult = fert
        self.flood_level = flood_dmg_level
        w.set_fertility_mult(fert)
        w.set_water_offset(water)
        if w.danger_extra.any():
            w.danger_extra *= 0.985 ** (dt * 10) if dt else 1.0
            w.danger_extra[w.danger_extra < 0.02] = 0.0

    # ------------------------------------------------------------------
    def active_labels(self) -> List[str]:
        return [DISASTERS[d["kind"]]["label"] for d in self.disasters]

    def state_dict(self) -> dict:
        return {"base_temp_offset": self.base_temp_offset, "base_food_mult": self.base_food_mult,
                "base_water_offset": self.base_water_offset, "gravity": self.gravity.copy(),
                "disasters": [dict(d) for d in self.disasters], "timeline": [dict(e) for e in self.timeline],
                "history": [dict(h) for h in self.history], "disasters_enabled": self.disasters_enabled}

    def load_state_dict(self, d: dict) -> None:
        self.base_temp_offset = float(d["base_temp_offset"])
        self.base_food_mult = float(d["base_food_mult"])
        self.base_water_offset = int(d["base_water_offset"])
        self.gravity = np.asarray(d["gravity"], dtype=np.float32)
        self.disasters = [dict(x) for x in d["disasters"]]
        self.timeline = [dict(x) for x in d["timeline"]]
        self.history = [dict(x) for x in d["history"]]
        self.disasters_enabled = bool(d.get("disasters_enabled", True))
