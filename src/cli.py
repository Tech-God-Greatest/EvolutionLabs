"""Command-line interface and main entry loop for Digital Evolution Lab."""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional
import pygame

from .core.config import Config
from .analysis.experiments import EXPERIMENT_PRESETS, ExperimentRunner
from .analysis.report import ReportGenerator
from .persistence.load import load_ecosystem as load_simulation
from .persistence.save import save_ecosystem as save_simulation
from .rendering.renderer import MasterRenderer
from .simulation.ecosystem import Ecosystem


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Digital Evolution Lab — Artificial Life & Neural Evolution Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless compute mode (no window)")
    parser.add_argument("--experiment", type=str, default=None, choices=list(EXPERIMENT_PRESETS.keys()),
                        help="Run a curated evolutionary experiment preset")
    parser.add_argument("--generations", type=float, default=None, help="Stop after reaching this mean generation")
    parser.add_argument("--ticks", type=int, default=None, help="Stop after reaching this number of simulation ticks")
    parser.add_argument("--seed", type=int, default=42, help="Master RNG seed for deterministic reproduction")
    parser.add_argument("--population", type=int, default=None, help="Initial population override")
    parser.add_argument("--config", type=str, default=None, help="Path to custom JSON configuration file")
    parser.add_argument("--load", type=str, default=None, help="Load simulation state from saved file (.json or .json.gz)")
    parser.add_argument("--save-on-exit", type=str, default=None, help="Save final simulation state to path on exit")
    parser.add_argument("--export-stats", type=str, default=None, help="Export time-series telemetry data to CSV file")
    parser.add_argument("--report", type=str, default=None, help="Directory to export scientific markdown report & charts")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark across 1k, 3k, 5k organisms")
    parser.add_argument("--screenshot", type=str, default=None, help="Capture screenshot of app window and exit")
    parser.add_argument("--screenshot-frames", type=int, default=40, help="Number of frames before capturing screenshot")
    parser.add_argument("--demo-screenshots", action="store_true", help="Capture full suite of app window screenshots for documentation")
    parser.add_argument("--width", type=int, default=1440, help="Initial window width")
    parser.add_argument("--height", type=int, default=860, help="Initial window height")
    return parser.parse_args(args)


def run_benchmark(seed: int = 42):
    print("=" * 65)
    print(" DIGITAL EVOLUTION LAB — PERFORMANCE BENCHMARK (SoA NUMPY)")
    print("=" * 65)
    test_sizes = [500, 1000, 2500, 5000]

    for pop in test_sizes:
        cfg = Config()
        cfg.update({"population_cap": pop * 2, "population": pop, "seed": seed, "dt": 0.1})
        eco = Ecosystem(config=cfg, seed=seed)

        # Warmup
        for _ in range(10):
            eco.step()

        # Timed steps
        n_steps = 100
        t0 = time.perf_counter()
        for _ in range(n_steps):
            eco.step()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        tps = n_steps / elapsed
        sim_sec_per_sec = (n_steps * cfg.dt) / elapsed

        print(f" Population: {pop:5d} | Steps: {n_steps:3d} | Elapsed: {elapsed:6.3f}s | Speed: {tps:7.1f} steps/s ({sim_sec_per_sec:5.1f}x sim speed)")
    print("=" * 65)


def run_headless(args, cfg: Config):
    target_gen = args.generations or 50.0
    print("=" * 65)
    print(" DIGITAL EVOLUTION LAB — HEADLESS SIMULATION ENGINE")
    print(f" Seed: {cfg.seed} | Population: {cfg.population} | Target Gen: {target_gen}")
    print("=" * 65)

    eco = None
    if args.load:
        print(f"Loading simulation state from: {args.load}")
        eco = load_simulation(args.load)
    else:
        eco = Ecosystem(config=cfg, seed=cfg.seed)
        if args.experiment and args.experiment in EXPERIMENT_PRESETS:
            preset = EXPERIMENT_PRESETS[args.experiment]
            print(f"Applying preset: {preset.title}")
            if preset.setup_fn:
                preset.setup_fn(eco)

    t0 = time.time()
    last_print = t0
    step = 0

    try:
        while True:
            eco.step()
            step += 1

            now = time.time()
            if now - last_print >= 2.0 or eco.mean_generation >= target_gen or (args.ticks and step >= args.ticks):
                last_print = now
                sp_count = len(eco.species.active_ids())
                print(f" [Tick {eco.tick:6d}] Gen: {eco.mean_generation:5.2f}/{target_gen:.1f} | Pop: {eco.org.n:4d} | Food: {eco.food.n:5d} | Species: {sp_count:3d}")

            if eco.mean_generation >= target_gen:
                print(f"Reached target generation {target_gen:.1f}!")
                break
            if args.ticks and step >= args.ticks:
                print(f"Reached target ticks {args.ticks}!")
                break
            if eco.org.n == 0:
                print("Total extinction reached. Ending simulation.")
                break
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")

    elapsed = time.time() - t0
    print("=" * 65)
    print(f" Completed in {elapsed:.2f}s ({step / max(0.001, elapsed):.1f} ticks/s)")
    print(f" Final Pop: {eco.org.n} | Total Births: {eco.births_total} | Total Deaths: {eco.deaths_total}")
    print("=" * 65)

    if args.export_stats:
        print(f"Exporting stats CSV to: {args.export_stats}")
        eco.stats.export_csv(args.export_stats)

    if args.report:
        print(f"Generating full research report in: {args.report}")
        rep = ReportGenerator(eco)
        rep.export_full_report(args.report)

    if args.save_on_exit:
        print(f"Saving simulation state to: {args.save_on_exit}")
        save_simulation(eco, args.save_on_exit)


def run_gui(args, cfg: Config):
    eco = None
    if args.load:
        print(f"Loading simulation state from: {args.load}")
        eco = load_simulation(args.load)
    else:
        eco = Ecosystem(config=cfg, seed=cfg.seed)
        if args.experiment and args.experiment in EXPERIMENT_PRESETS:
            preset = EXPERIMENT_PRESETS[args.experiment]
            print(f"Applying preset: {preset.title}")
            if preset.setup_fn:
                preset.setup_fn(eco)

    renderer = MasterRenderer(screen_w=args.width, screen_h=args.height)
    renderer.init_simulation(eco)

    clock = pygame.time.Clock()
    running = True
    frame_count = 0
    t_prev = time.perf_counter()
    sim_tps_accum = 0
    sim_tps_display = 0.0
    sim_tps_time = t_prev

    while running:
        t_now = time.perf_counter()
        dt_frame = min(0.1, t_now - t_prev)
        t_prev = t_now
        frame_count += 1

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            renderer.handle_event(event, eco)

        # Simulation stepping
        if not renderer.paused and eco.org.n > 0:
            speed = renderer.speed_multiplier
            # Compute steps needed this frame
            target_steps = max(1, int(round(speed)))
            for _ in range(target_steps):
                eco.step()
                sim_tps_accum += 1

        # Calculate TPS
        if t_now - sim_tps_time >= 1.0:
            sim_tps_display = sim_tps_accum / (t_now - sim_tps_time)
            sim_tps_accum = 0
            sim_tps_time = t_now

        # Render frame
        renderer.render(eco, dt=dt_frame, sim_fps=sim_tps_display)

        # Demo screenshots suite
        if args.demo_screenshots:
            if frame_count == 12:
                renderer.intro.dismiss()
            elif frame_count == 24:
                renderer.save_screenshot("screenshots/simulation_overview.png")
            elif frame_count == 28:
                renderer.dashboard.visible = True
            elif frame_count == 40:
                renderer.save_screenshot("screenshots/dashboard_telemetry.png")
            elif frame_count == 44:
                renderer.dashboard.visible = False
                if eco.org.n > 0:
                    renderer.selected_org_id = int(eco.org.id[0])
                    renderer.inspector.set_organism(renderer.selected_org_id, eco)
            elif frame_count == 56:
                renderer.save_screenshot("screenshots/neural_inspector.png")
            elif frame_count >= 60:
                running = False
                break

        # Single screenshot and exit check
        if args.screenshot and frame_count >= args.screenshot_frames:
            print(f"Capturing application window screenshot: {args.screenshot}")
            renderer.save_screenshot(args.screenshot)
            running = False
            break

        clock.tick(60)

    pygame.quit()

    if args.export_stats:
        eco.stats.export_csv(args.export_stats)
    if args.report:
        rep = ReportGenerator(eco)
        rep.export_full_report(args.report)
    if args.save_on_exit:
        save_simulation(eco, args.save_on_exit)


def main():
    args = parse_args()

    if args.benchmark:
        run_benchmark(seed=args.seed)
        return

    # Load configuration
    if args.config:
        cfg = Config.from_json(args.config)
    else:
        cfg = Config()

    if args.seed is not None:
        cfg.seed = args.seed
    if args.population is not None:
        cfg.update({
            "population_cap": max(cfg.population_cap, args.population * 2),
            "population": args.population,
        })

    if args.experiment and args.experiment in EXPERIMENT_PRESETS:
        preset = EXPERIMENT_PRESETS[args.experiment]
        cfg.update(preset.config_overrides)

    if args.headless:
        run_headless(args, cfg)
    else:
        run_gui(args, cfg)


if __name__ == "__main__":
    main()
