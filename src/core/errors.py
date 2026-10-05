"""Typed exceptions so failures are explicit (never silent data corruption)."""


class DELError(Exception):
    """Base class for all Digital Evolution Lab errors."""


class ConfigError(DELError):
    """Invalid or inconsistent configuration."""


class SaveError(DELError):
    """Corrupted, incompatible or unreadable save file."""


class GenomeError(DELError):
    """Corrupted or incompatible genome."""
