"""Manuscript definitions collected from subjects.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

UNREWARDED_MICE = ("A1", "A2")

GROUP_ORDER = ("C", "U", "I", "P")

DEFAULT_GROUP_COLORS = {
    "C": "#7ED3FC",
    "U": "#F2B84B",
    "I": "#B39DCA",
    "P": "#9BCD88",
}

PV_PC_GROUP_MICE = {
    "C": ("A8", "A7", "A9", "A4", "A5", "A10", "A11"),
    "U": UNREWARDED_MICE,
    "I": ("B1", "B2", "B5", "B6", "B7", "B10", "B11"),
    "P": ("H1", "H2", "H3", "H5", "H6"),
}
