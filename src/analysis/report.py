"""Experiment Report Generator and Export System.

Produces structured markdown reports, summary metrics, CSV tables, and
matplotlib PNG charts from empirical simulation data (§46, §47).
All observations are strictly derived from measured time-series deltas.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional
import numpy as np


class ReportGenerator:
    def __init__(self, eco):
        self.eco = eco

    def generate_observations(self) -> List[str]:
        """Compute honest, evidence-based observations from statistical history."""
        eco = self.eco
        h = eco.stats.history
        obs = []
        if len(h["time"]) < 2:
            return ["Simulation ran for too few intervals to extract long-term evolutionary trends."]

        times = np.array(h["time"])
        gens = np.array(h["generation"])
        pops = np.array(h["population"])
        preds = np.array(h["predators"])
        herbs = np.array(h["herbivores"])
        food = np.array(h["food_count"])
        vis = np.array(h["avg_vision"])
        spd = np.array(h["avg_speed"])
        div = np.array(h["genetic_diversity"])

        # 1. Population & Carrying Capacity
        init_pop = pops[0]
        final_pop = pops[-1]
        max_pop = pops.max()
        min_pop = pops.min()
        if max_pop > init_pop * 1.5 and min_pop < init_pop * 0.5:
            obs.append(f"Population underwent high amplitude fluctuations (min: {min_pop}, max: {max_pop}).")
        else:
            obs.append(f"Population remained stabilized within [{min_pop}, {max_pop}] across the run.")

        # 2. Vision and Speed evolutionary trends
        if len(vis) >= 5 and vis[0] > 0:
            vis_delta = (vis[-1] - vis[0]) / vis[0] * 100.0
            if abs(vis_delta) >= 10.0:
                direction = "increased" if vis_delta > 0 else "decreased"
                obs.append(f"Average vision range {direction} by {abs(vis_delta):.1f}% over the course of evolution ({vis[0]:.1f} -> {vis[-1]:.1f} units).")

        if len(spd) >= 5 and spd[0] > 0:
            spd_delta = (spd[-1] - spd[0]) / spd[0] * 100.0
            if abs(spd_delta) >= 10.0:
                direction = "increased" if spd_delta > 0 else "decreased"
                obs.append(f"Average organism movement speed {direction} by {abs(spd_delta):.1f}% ({spd[0]:.1f} -> {spd[-1]:.1f} units/s).")

        # 3. Predator-Prey / Herbivore dynamics
        if preds.max() > 5 and herbs.max() > 10:
            # Check for correlation or cycles
            p_mean = preds.mean()
            h_mean = herbs.mean()
            if preds[-1] > 0 and herbs[-1] > 0:
                obs.append(f"Sustained trophic coexistence achieved: mean herbivore population {h_mean:.1f} vs predator population {p_mean:.1f}.")
            elif preds[-1] == 0 and preds[0] > 0:
                obs.append("Predator lineages suffered extinction due to starvation or prey scarcity.")

        # 4. Speciation
        n_created = eco.species.total_created
        n_extinct = eco.species.total_extinct
        n_alive = len(eco.species.active_ids())
        obs.append(f"Speciation dynamics: {n_created} total distinct species arose through divergence; {n_extinct} went extinct, leaving {n_alive} extant species.")

        # 5. Genetic Diversity
        if len(div) >= 5 and div[0] > 0:
            div_delta = (div[-1] - div[0]) / div[0] * 100.0
            if div_delta < -25.0:
                obs.append(f"Genetic diversity contracted by {abs(div_delta):.1f}%, indicating convergence around successful specialist lineages.")
            elif div_delta > 25.0:
                obs.append(f"Genetic diversity expanded by {div_delta:.1f}%, reflecting niche differentiation.")

        # 6. Extinction and Disasters
        if eco.env.history:
            dis_counts = len(eco.env.history)
            obs.append(f"Ecosystem survived {dis_counts} environmental perturbation / disaster event(s).")

        return obs

    def generate_markdown(self) -> str:
        eco = self.eco
        h = eco.stats.history
        init_pop = h["population"][0] if h["population"] else 0
        final_pop = h["population"][-1] if h["population"] else 0
        avg_life = h["avg_lifespan"][-1] if h["avg_lifespan"] else 0.0
        final_div = h["genetic_diversity"][-1] if h["genetic_diversity"] else 0.0
        dur_gens = eco.mean_generation
        dur_time = eco.time

        observations = self.generate_observations()
        obs_md = "\n".join(f"- {o}" for o in observations)

        events_md = ""
        recent_evs = eco.events.recent(10)
        if recent_evs:
            events_md = "\n".join(f"- **Gen {int(round(e.generation))}** ({e.time:.1f}s): {e.text}" for e in recent_evs)
        else:
            events_md = "- None recorded."

        return f"""# Digital Evolution Lab — Experiment Report

## Executive Summary
- **Simulated Duration:** {dur_gens:.1f} generations ({dur_time:.1f} simulated seconds)
- **Initial Population:** {init_pop:,}
- **Final Population:** {final_pop:,}
- **Total Species Discovered:** {eco.species.total_created}
- **Species Extinct:** {eco.species.total_extinct}
- **Extant Active Species:** {len(eco.species.active_ids())}
- **Average Lifespan:** {avg_life:.1f} s
- **Genetic Diversity Index:** {final_div:.4f}
- **Energy Conservation Residual:** {eco.energy_balance_error():.4e}

---

## Empirical Observations
{obs_md}

---

## Key Milestone Timeline
{events_md}

---

*Report automatically compiled by Digital Evolution Lab Analysis Engine.*
"""

    def export_full_report(self, target_dir: str) -> None:
        """Export comprehensive experiment bundle (§46)."""
        os.makedirs(target_dir, exist_ok=True)
        graphs_dir = os.path.join(target_dir, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)

        # 1. config.json
        self.eco.cfg.save(os.path.join(target_dir, "config.json"))

        # 2. population.csv
        self.eco.stats.export_csv(os.path.join(target_dir, "population.csv"))

        # 3. species.csv
        species_path = os.path.join(target_dir, "species.csv")
        with open(species_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "parent", "founded_gen", "extinct_time", "population", "peak_population", "avg_lifespan"])
            for sp in self.eco.species.species.values():
                writer.writerow([sp.id, sp.name, sp.parent or "", f"{sp.founded_gen}", f"{sp.extinct_time or ''}", sp.population, sp.peak_population, f"{sp.avg_lifespan:.1f}"])

        # 4. events.json
        events_path = os.path.join(target_dir, "events.json")
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.eco.events.events], f, indent=2)

        # 5. summary.md
        summary_path = os.path.join(target_dir, "summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown())

        # 6. graphs/ (PNG charts using matplotlib)
        self.render_charts(graphs_dir)

    def render_charts(self, output_dir: str) -> None:
        """Render high-resolution scientific charts using matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.style.use("dark_background")
            h = self.eco.stats.history
            if not h["time"]:
                return

            t = h["time"]

            # Chart 1: Population & Resources (Oscillations)
            fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
            ax1.set_facecolor("#0e131b")
            fig.patch.set_facecolor("#0a0d14")
            ax1.plot(t, h["population"], label="Total Population", color="#00ffcc", lw=2)
            ax1.plot(t, h["herbivores"], label="Herbivores", color="#44ff88", lw=1.5, ls="--")
            ax1.plot(t, h["predators"], label="Predators", color="#ff4466", lw=1.5)
            ax1.set_xlabel("Simulated Time (seconds)", color="#8ba2be")
            ax1.set_ylabel("Organism Count", color="#8ba2be")
            ax1.grid(True, color="#1c2432", ls=":")

            ax2 = ax1.twinx()
            ax2.plot(t, h["food_count"], label="Food Items", color="#ffcc00", lw=1.2, alpha=0.7)
            ax2.set_ylabel("Food Items", color="#ffcc00")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", facecolor="#141c26")
            plt.title("Ecosystem Population & Resource Cycles", color="#e2e8f0", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "population_cycles.png"))
            plt.close()

            # Chart 2: Evolutionary Trait Trends
            fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
            ax.set_facecolor("#0e131b")
            fig.patch.set_facecolor("#0a0d14")
            ax.plot(t, h["avg_speed"], label="Avg Speed", color="#38bdf8", lw=2)
            ax.plot(t, h["avg_vision"], label="Avg Vision Range", color="#a855f7", lw=2)
            ax.plot(t, h["avg_size"], label="Avg Size", color="#fb923c", lw=1.5)
            ax.set_xlabel("Simulated Time (seconds)", color="#8ba2be")
            ax.set_ylabel("Trait Value", color="#8ba2be")
            ax.grid(True, color="#1c2432", ls=":")
            ax.legend(loc="upper left", facecolor="#141c26")
            plt.title("Evolutionary Adaptation of Physical Traits", color="#e2e8f0", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "trait_evolution.png"))
            plt.close()

            # Chart 3: Genetic Diversity & Species Count
            fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
            ax1.set_facecolor("#0e131b")
            fig.patch.set_facecolor("#0a0d14")
            ax1.plot(t, h["genetic_diversity"], label="Genetic Diversity", color="#ec4899", lw=2)
            ax1.set_xlabel("Simulated Time (seconds)", color="#8ba2be")
            ax1.set_ylabel("Diversity Index", color="#ec4899")
            ax1.grid(True, color="#1c2432", ls=":")

            ax2 = ax1.twinx()
            ax2.plot(t, h["active_species"], label="Active Species", color="#22d3ee", lw=1.5, ls="--")
            ax2.set_ylabel("Species Count", color="#22d3ee")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", facecolor="#141c26")
            plt.title("Speciation & Genetic Diversity Over Time", color="#e2e8f0", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "genetic_diversity.png"))
            plt.close()

        except Exception as e:
            # Non-fatal if plotting environment has issues
            pass
