"""Adaptive bandwidth policy - T4 initial selection.  Formal model §1.5/§3.

v0.1 policy: conservative default (minimal sufficient information), wider
only when the task features signal high information-processing demand
(Galbraith: uncertainty/complexity raise information requirements).
P3 treats bandwidth as an experimental variable across all four levels;
this policy is the 'adaptive' arm.
"""
from __future__ import annotations

from ..org.context import Bandwidth


def choose_bandwidth(features: dict[str, str]) -> Bandwidth:
    if features.get("uncertainty") == "high":
        return Bandwidth.WIDE
    if features.get("complexity") == "high":
        return Bandwidth.WIDE
    return Bandwidth.NARROW
