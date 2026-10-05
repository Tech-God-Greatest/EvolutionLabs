"""Genome representation.

A genome = *trait genes* (physical/physiological parameters) + *neural genes*
(weights, biases and a connection mask).  Populations store genomes as dense
arrays (`traits` (N,G), `weights` (N,P), `masks` (N,P)); :class:`Genome` is the
convenient single-individual object used for import/export/surgery.
"""
from __future__ import annotations

import json
import math
from typing import Dict, Optional

import numpy as np

from ..core.errors import GenomeError
from ..neural.network import NetSpec

GENOME_FORMAT = "digital_evolution_genome"
GENOME_VERSION = 1

# name, lo, hi, default, description
TRAIT_SPECS = [
    ("max_speed", 15.0, 120.0, 55.0, "top speed (units/s)"),
    ("acceleration", 20.0, 220.0, 90.0, "acceleration (units/s^2)"),
    ("vision_range", 25.0, 200.0, 90.0, "vision range (units)"),
    ("vision_angle", 30.0, 340.0, 150.0, "vision cone angle (degrees)"),
    ("metabolism", 0.5, 2.0, 1.0, "metabolic rate multiplier"),
    ("body_size", 2.5, 10.0, 5.0, "body radius (units)"),
    ("attack_strength", 0.0, 3.0, 0.3, "attack strength"),
    ("temp_preference", 0.0, 1.0, 0.5, "preferred temperature"),
    ("temp_tolerance", 0.05, 0.6, 0.25, "temperature tolerance"),
    ("reproduction_threshold", 0.35, 0.98, 0.70, "energy fraction needed to reproduce"),
    ("energy_efficiency", 0.6, 1.4, 1.0, "digestive efficiency"),
    ("diet", 0.0, 1.0, 0.1, "0 = plants, 1 = meat"),
    ("mutability", 0.1, 5.0, 1.0, "per-organism mutation-rate multiplier"),
]
TRAIT_NAMES = [t[0] for t in TRAIT_SPECS]
N_TRAITS = len(TRAIT_SPECS)
TRAIT_LO = np.array([t[1] for t in TRAIT_SPECS], dtype=np.float32)
TRAIT_HI = np.array([t[2] for t in TRAIT_SPECS], dtype=np.float32)
TRAIT_RANGE = TRAIT_HI - TRAIT_LO
TRAIT_DEFAULT = np.array([t[3] for t in TRAIT_SPECS], dtype=np.float32)
TRAIT_DESC = {t[0]: t[4] for t in TRAIT_SPECS}
IDX: Dict[str, int] = {n: i for i, n in enumerate(TRAIT_NAMES)}
(G_SPEED, G_ACCEL, G_VIS, G_VANG, G_METAB, G_SIZE, G_ATTACK, G_TPREF, G_TTOL,
 G_REPRO, G_EFF, G_DIET, G_MUT) = range(N_TRAITS)


def clamp_traits(t: np.ndarray) -> np.ndarray:
    np.clip(t, TRAIT_LO, TRAIT_HI, out=t)
    return t


def carnivore_template(cfg) -> np.ndarray:
    """Founder trait template for carnivore lineages (seed genetics only)."""
    t = TRAIT_DEFAULT.copy()
    t[G_DIET] = 0.85
    t[G_ATTACK] = 1.3
    t[G_SIZE] = 6.0
    t[G_SPEED] = 68.0
    t[G_VIS] = 110.0
    t[G_VANG] = 110.0
    t[G_REPRO] = 0.78
    t[G_REPRO] = max(t[G_REPRO], cfg.reproduction_threshold)
    return t


def herbivore_template(cfg) -> np.ndarray:
    t = TRAIT_DEFAULT.copy()
    t[G_REPRO] = cfg.reproduction_threshold
    return t


def mean_normalized_trait_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean |a-b| over traits, each normalised by its range (broadcasts on leading dims)."""
    return np.mean(np.abs(a - b) / TRAIT_RANGE, axis=-1)


class Genome:
    """A single, serialisable genome."""

    def __init__(self, traits: np.ndarray, weights: np.ndarray, mask: np.ndarray, spec: NetSpec,
                 meta: Optional[dict] = None):
        self.traits = np.asarray(traits, dtype=np.float32).copy()
        self.weights = np.asarray(weights, dtype=np.float32).copy()
        self.mask = np.asarray(mask, dtype=np.uint8).copy()
        self.spec = spec
        self.meta = dict(meta or {})
        self.validate()

    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.traits.shape != (N_TRAITS,):
            raise GenomeError(f"expected {N_TRAITS} traits, got shape {self.traits.shape}")
        if self.weights.shape != (self.spec.n_params,) or self.mask.shape != (self.spec.n_params,):
            raise GenomeError("weight/mask vector does not match network specification")
        if not np.all(np.isfinite(self.traits)) or not np.all(np.isfinite(self.weights)):
            raise GenomeError("genome contains NaN or infinite values")
        if np.any(self.traits < TRAIT_LO - 1e-4) or np.any(self.traits > TRAIT_HI + 1e-4):
            bad = [TRAIT_NAMES[i] for i in range(N_TRAITS)
                   if not (TRAIT_LO[i] - 1e-4 <= self.traits[i] <= TRAIT_HI[i] + 1e-4)]
            raise GenomeError(f"trait(s) out of valid range: {bad}")

    def trait(self, name: str) -> float:
        return float(self.traits[IDX[name]])

    def set_trait(self, name: str, value: float) -> None:
        if name not in IDX:
            raise GenomeError(f"unknown trait '{name}'")
        if not math.isfinite(value):
            raise GenomeError("trait value must be finite")
        i = IDX[name]
        self.traits[i] = float(np.clip(value, TRAIT_LO[i], TRAIT_HI[i]))

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": GENOME_FORMAT,
            "version": GENOME_VERSION,
            "traits": {n: float(self.traits[i]) for i, n in enumerate(TRAIT_NAMES)},
            "network": {
                **self.spec.to_dict(),
                "weights": [float(x) for x in self.weights],
                "mask": [int(x) for x in self.mask],
            },
            "meta": self.meta,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(d: dict, expected_spec: Optional[NetSpec] = None) -> "Genome":
        try:
            if d.get("format") != GENOME_FORMAT:
                raise GenomeError("not a Digital Evolution Lab genome file")
            if int(d.get("version", 0)) > GENOME_VERSION:
                raise GenomeError(f"genome version {d.get('version')} is newer than supported ({GENOME_VERSION})")
            tr = d["traits"]
            traits = np.array([float(tr[n]) for n in TRAIT_NAMES], dtype=np.float32)
            net = d["network"]
            spec = NetSpec.from_dict(net)
            weights = np.array(net["weights"], dtype=np.float32)
            mask = np.array(net.get("mask", [1] * len(weights)), dtype=np.uint8)
        except GenomeError:
            raise
        except (KeyError, TypeError, ValueError) as e:
            raise GenomeError(f"malformed genome: {e!r}")
        if expected_spec is not None and not spec.same_as(expected_spec):
            raise GenomeError(f"genome network {spec} is incompatible with this world's network {expected_spec}")
        return Genome(traits, weights, mask, spec, d.get("meta"))

    @staticmethod
    def from_json(text: str, expected_spec: Optional[NetSpec] = None) -> "Genome":
        try:
            d = json.loads(text)
        except json.JSONDecodeError as e:
            raise GenomeError(f"genome is not valid JSON: {e}")
        if not isinstance(d, dict):
            raise GenomeError("genome JSON must be an object")
        return Genome.from_dict(d, expected_spec)

    def copy(self) -> "Genome":
        return Genome(self.traits, self.weights, self.mask, self.spec, self.meta)
