"""Per-task sandbox images. The fallback image is only used when a track has none.

SWE-bench 4.1.0 remote instance keys are ``swebench/sweb.eval.x86_64.<id>:latest``
with ``__`` rewritten to ``_1776_`` (see ``TestSpec.instance_image_key``).
ProgramBench's pinned usage guide names ``programbench/<id>:task_cleanroom_v6``
with the same ``__`` rewrite.
"""
from __future__ import annotations

import re


SWEBENCH_NAMESPACE = "swebench"
SWEBENCH_ARCH = "x86_64"
PROGRAMBENCH_TAG = "task_cleanroom_v6"
_INSTANCE_ID = re.compile(r"[A-Za-z0-9_.-]{1,200}")


def _safe_instance_id(instance_id):
    if not isinstance(instance_id, str) or not _INSTANCE_ID.fullmatch(instance_id):
        raise ValueError("invalid_instance_id")
    return instance_id.replace("__", "_1776_").lower()


def swebench_instance_image(instance_id):
    """Official remote eval image for one SWE-bench instance."""
    safe = _safe_instance_id(instance_id)
    return f"{SWEBENCH_NAMESPACE}/sweb.eval.{SWEBENCH_ARCH}.{safe}:latest"


def programbench_instance_image(instance_id):
    """Official clean-room image: binary and docs, no network at runtime."""
    safe = _safe_instance_id(instance_id)
    return f"programbench/{safe}:{PROGRAMBENCH_TAG}"


def resolve_task_image(track, task_id, fallback=None):
    """Prefer the official per-task image. ``fallback`` is ``ALO_SANDBOX_IMAGE``."""
    if track == "SWE-bench_Lite":
        return swebench_instance_image(task_id)
    if track == "ProgramBench":
        return programbench_instance_image(task_id)
    if isinstance(fallback, str) and fallback and not fallback.startswith("-") and not any(ch.isspace() for ch in fallback):
        return fallback
    return None
