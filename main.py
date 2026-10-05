"""Digital Evolution Lab — Main Entry Point.

Run interactively:
    python main.py

Run headless simulation:
    python main.py --headless --generations 100 --seed 42

Run curated experiment preset:
    python main.py --experiment predator_prey

Run performance benchmark:
    python main.py --benchmark
"""
from __future__ import annotations

import sys
from src.cli import main

if __name__ == "__main__":
    main()
