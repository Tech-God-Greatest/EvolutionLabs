"""Evolution Events and Automatic Discovery System.

Detects milestone evolutionary events and ecological anomalies:
- New species emergence
- Species extinction
- Dominant species shifts
- Population explosions and crashes
- Emergence of predatory or pack behaviors
- Genetic shifts in average traits over generations
- Disaster strikes and bottleneck survivals

Maintains a timeline for display in the UI and experiment reports.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from ..evolution.genome import G_VIS, G_SPEED, G_SIZE, G_ATTACK, G_DIET, TRAIT_NAMES


class Event:
    __slots__ = ("id", "tick", "time", "generation", "kind", "text", "severity", "data")

    def __init__(
        self,
        event_id: int,
        tick: int,
        time: float,
        generation: float,
        kind: str,
        text: str,
        severity: int = 1,
        data: Optional[dict] = None,
    ):
        self.id = event_id
        self.tick = tick
        self.time = time
        self.generation = generation
        self.kind = kind
        self.text = text
        self.severity = severity
        self.data = data or {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tick": self.tick,
            "time": self.time,
            "generation": self.generation,
            "kind": self.kind,
            "text": self.text,
            "severity": self.severity,
            "data": self.data,
        }

    @staticmethod
    def from_dict(d: dict) -> "Event":
        return Event(
            event_id=d["id"],
            tick=d["tick"],
            time=d["time"],
            generation=d["generation"],
            kind=d["kind"],
            text=d["text"],
            severity=d.get("severity", 1),
            data=d.get("data", {}),
        )


class EventDetector:
    def __init__(self, eco, max_events: int = 500):
        self.eco = eco
        self.max_events = max_events
        self.events: List[Event] = []
        self.next_id = 1
        # Milestones tracking
        self.first_predator_detected = False
        self.first_pack_detected = False
        self.last_dominant_species: Optional[int] = None
        self.prev_pop = 0
        self.last_check_time = 0.0
        # Historical trait baselines for evolutionary drift
        self.trait_baselines: Dict[int, float] = {}
        self.last_trait_report_gen = 0

    def log(self, kind: str, text: str, severity: int = 1, data: Optional[dict] = None) -> Event:
        ev = Event(
            event_id=self.next_id,
            tick=self.eco.tick,
            time=self.eco.time,
            generation=float(self.eco.mean_generation),
            kind=kind,
            text=text,
            severity=severity,
            data=data,
        )
        self.next_id += 1
        self.events.append(ev)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        return ev

    def update(self) -> None:
        """Periodic automatic discovery hook invoked by the simulation/stats."""
        eco = self.eco
        s = eco.org
        n = s.n
        cur_gen = int(eco.mean_generation)

        # 1. Speciation events
        for sid in eco.species.new_this_epoch:
            sp = eco.species.species.get(sid)
            if sp:
                parent_info = ""
                if sp.parent is not None and sp.parent in eco.species.species:
                    parent_info = f" (diverged from {eco.species.species[sp.parent].name})"
                self.log(
                    "species_emerged",
                    f"Species '{sp.name}' emerged{parent_info}",
                    severity=1,
                    data={"species_id": sid, "name": sp.name, "parent": sp.parent},
                )

        for sid in eco.species.extinct_this_epoch:
            sp = eco.species.species.get(sid)
            name = sp.name if sp else f"#{sid}"
            self.log(
                "species_extinct",
                f"Species '{name}' went extinct after reaching peak population {sp.peak_population if sp else 0}",
                severity=2,
                data={"species_id": sid, "peak_pop": sp.peak_population if sp else 0},
            )

        if n == 0:
            return

        # 2. Emergence of predatory behavior
        if not self.first_predator_detected:
            if s.food_meat[:n].sum() > 5.0 or s.dmg_dealt[:n].sum() > 20.0:
                self.first_predator_detected = True
                self.log(
                    "behavior_discovery",
                    f"First active predator behavior observed in generation {cur_gen}",
                    severity=3,
                    data={"generation": cur_gen},
                )

        # 3. Emergence of pack / social behavior
        if not self.first_pack_detected:
            if (s.social_t[:n] > 15.0).sum() >= 5:
                self.first_pack_detected = True
                self.log(
                    "behavior_discovery",
                    f"Coordinated group/pack social behavior detected in generation {cur_gen}",
                    severity=2,
                    data={"generation": cur_gen},
                )

        # 4. Dominant species shift
        active_species = eco.species.top(k=1)
        if active_species:
            top_sp = active_species[0]
            if top_sp.population >= 10 and top_sp.id != self.last_dominant_species:
                if self.last_dominant_species is not None:
                    old_name = eco.species.species[self.last_dominant_species].name if self.last_dominant_species in eco.species.species else "previous"
                    self.log(
                        "species_dominant",
                        f"Species '{top_sp.name}' surpassed {old_name} as dominant with {top_sp.population} individuals",
                        severity=2,
                        data={"new_dominant": top_sp.id, "old_dominant": self.last_dominant_species},
                    )
                self.last_dominant_species = top_sp.id

        # 5. Population crash / explosion monitoring
        if self.prev_pop > 50:
            if n <= self.prev_pop * 0.45:
                self.log(
                    "population_crash",
                    f"Severe population collapse: {self.prev_pop} down to {n} organisms",
                    severity=3,
                    data={"before": self.prev_pop, "after": n},
                )
            elif n >= self.prev_pop * 2.2 and n > 200:
                self.log(
                    "population_explosion",
                    f"Rapid population bloom: reached {n} organisms",
                    severity=2,
                    data={"before": self.prev_pop, "after": n},
                )
        self.prev_pop = n

        # 6. Evolutionary trait drift milestones (every 25 generations)
        if cur_gen >= self.last_trait_report_gen + 25 and n >= 20:
            self.last_trait_report_gen = cur_gen
            for trait_idx, label in [(G_VIS, "vision range"), (G_SPEED, "speed"), (G_SIZE, "body size")]:
                cur_val = float(s.traits[:n, trait_idx].mean())
                if trait_idx not in self.trait_baselines:
                    self.trait_baselines[trait_idx] = cur_val
                else:
                    base = self.trait_baselines[trait_idx]
                    if base > 1e-4:
                        pct = (cur_val - base) / base * 100.0
                        if abs(pct) >= 20.0:
                            direction = "increased" if pct > 0 else "decreased"
                            self.log(
                                "trait_shift",
                                f"Average {label} {direction} by {abs(pct):.1f}% since start ({base:.1f} -> {cur_val:.1f})",
                                severity=2,
                                data={"trait": TRAIT_NAMES[trait_idx], "delta_pct": pct},
                            )
                            # update baseline so we don't spam duplicate events
                            self.trait_baselines[trait_idx] = cur_val

    def recent(self, count: int = 20) -> List[Event]:
        return self.events[-count:]

    def timeline(self) -> List[dict]:
        """Format events as structured evolution timeline milestones."""
        return [
            {
                "generation": int(round(e.generation)),
                "time": f"{e.time:.1f}s",
                "text": e.text,
                "kind": e.kind,
                "severity": e.severity,
            }
            for e in self.events
        ]

    def state_dict(self) -> dict:
        return {
            "events": [e.to_dict() for e in self.events],
            "next_id": self.next_id,
            "first_predator_detected": self.first_predator_detected,
            "first_pack_detected": self.first_pack_detected,
            "last_dominant_species": self.last_dominant_species,
            "prev_pop": self.prev_pop,
            "trait_baselines": {str(k): float(v) for k, v in self.trait_baselines.items()},
            "last_trait_report_gen": self.last_trait_report_gen,
        }

    def load_state_dict(self, d: dict) -> None:
        self.events = [Event.from_dict(x) for x in d.get("events", [])]
        self.next_id = int(d.get("next_id", len(self.events) + 1))
        self.first_predator_detected = bool(d.get("first_predator_detected", False))
        self.first_pack_detected = bool(d.get("first_pack_detected", False))
        self.last_dominant_species = d.get("last_dominant_species")
        self.prev_pop = int(d.get("prev_pop", 0))
        self.trait_baselines = {int(k): float(v) for k, v in d.get("trait_baselines", {}).items()}
        self.last_trait_report_gen = int(d.get("last_trait_report_gen", 0))
