"""Online learner - contextual model selection over the belief store.  §4.1/§4.2.

Thompson Sampling is primary (conjugate to the Beta-Bernoulli belief);
LinUCB lands as the IMACS-comparable control arm in P3.
"""
from __future__ import annotations

import random

from ..org.belief import BeliefStore


def thompson_select(
    belief: BeliefStore,
    model_ids: list[str],
    cell: str,
    rng: random.Random | None = None,
) -> str:
    """Sample each model's Beta posterior at this cell; return the argmax draw."""
    rng = rng or random
    best_id, best_draw = model_ids[0], -1.0
    for mid in model_ids:
        p = belief.query(mid, cell)
        draw = rng.betavariate(p.alpha, p.beta)
        if draw > best_draw:
            best_id, best_draw = mid, draw
    return best_id
