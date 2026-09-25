"""panini — a small, exact, traceable model of Pāṇini's Aṣṭādhyāyī.

Nothing here is statistical: every output is a derivation, every step cites
the sūtra that licensed it, and optional rules are explicit choices.
"""

from .prakriya import decline, derive, derive_all, format_trace, forms, sandhi
from .translit import detect_and_parse, slp1_to_devanagari, slp1_to_iast

__all__ = ["derive", "derive_all", "forms", "sandhi", "decline", "format_trace",
           "detect_and_parse", "slp1_to_iast", "slp1_to_devanagari"]
