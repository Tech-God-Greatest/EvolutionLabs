"""Deterministic Simulation Replay and Keyframe Scrubber.

Captures periodic keyframes of ecosystem state along with logged user actions
and environmental events, allowing the user to scrub backward and forward
through evolutionary history (§32).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

from ..simulation.ecosystem import Ecosystem


class ReplayManager:
    def __init__(self, eco: Ecosystem, max_keyframes: int = 50):
        self.eco = eco
        self.max_keyframes = max_keyframes
        self.keyframe_interval = float(eco.cfg.replay_keyframe_interval)
        self.last_keyframe_time = -1e9
        # List of (time, generation, state_dict)
        self.keyframes: List[Tuple[float, float, dict]] = []
        # Record initial keyframe at t=0
        self.capture_keyframe(force=True)

    def capture_keyframe(self, force: bool = False) -> bool:
        if not force and (self.eco.time - self.last_keyframe_time < self.keyframe_interval):
            return False

        self.last_keyframe_time = self.eco.time
        state = copy.deepcopy(self.eco.state_dict())
        self.keyframes.append((self.eco.time, self.eco.mean_generation, state))

        if len(self.keyframes) > self.max_keyframes:
            # Retain first (origin), last, and decimate middle
            self.keyframes = [self.keyframes[0]] + self.keyframes[-(self.max_keyframes - 1):]
        return True

    def available_times(self) -> List[float]:
        return [kf[0] for kf in self.keyframes]

    def available_generations(self) -> List[float]:
        return [kf[1] for kf in self.keyframes]

    def scrub_to_generation(self, target_gen: float) -> bool:
        """Find the nearest preceding keyframe and replay forward to target_gen."""
        if not self.keyframes:
            return False

        # Find best preceding keyframe
        best_kf = None
        for kf in self.keyframes:
            if kf[1] <= target_gen:
                best_kf = kf
            else:
                break

        if best_kf is None:
            best_kf = self.keyframes[0]

        # Restore keyframe
        self.eco.load_state_dict(copy.deepcopy(best_kf[2]))

        # Re-simulate forward deterministically to target_gen
        while self.eco.mean_generation < target_gen and self.eco.org.n > 0:
            self.eco.step()

        return True

    def scrub_to_time(self, target_time: float) -> bool:
        """Find the nearest preceding keyframe and replay forward to target_time."""
        if not self.keyframes:
            return False

        best_kf = None
        for kf in self.keyframes:
            if kf[0] <= target_time:
                best_kf = kf
            else:
                break

        if best_kf is None:
            best_kf = self.keyframes[0]

        self.eco.load_state_dict(copy.deepcopy(best_kf[2]))

        while self.eco.time < target_time and self.eco.org.n > 0:
            self.eco.step()

        return True
