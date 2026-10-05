"""Analysis, behavior discovery, statistics recording, experiments and reporting."""
from .behavior import classify_population, classify_organism_stats, explain_organism, BEHAVIOR_NAMES, BEHAVIOR_COLORS
from .events import EventDetector, Event
from .statistics import StatsRecorder
from .report import ReportGenerator
from .experiments import EXPERIMENT_PRESETS, ExperimentPreset, ExperimentRunner, WhatIfComparison, MazeExperimentRunner
