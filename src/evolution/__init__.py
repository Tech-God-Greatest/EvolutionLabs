"""Evolutionary mechanisms: genomes, mutation, crossover, selection, speciation."""
from .genome import Genome, TRAIT_NAMES, N_TRAITS
from .mutation import mutate, burst_mutation
from .crossover import crossover
from .selection import tournament, truncation, roulette, rank_select
from .species import Species, SpeciesManager
from .evolutionary_algorithms import make_strategy, make_generational
