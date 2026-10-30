"""Opt-in topic routing and separate objective/preference/cost utility.

TMS shortlists by topic outcome rates, retaining ties and unseen candidates.
Uniform exploration over the entire eligible pool prevents permanent lockout.
The utility uses one objective Beta draw, an empirical user preference score,
and estimated USD cost. Topic history is not added again to the utility.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random


@dataclass(frozen=True)
class AssignmentPolicy:
    objective_weight: float = 1.0
    preference_weight: float = 0.0
    cost_weight: float = 0.0  # utility penalty per estimated USD
    use_tms: bool = True
    use_preference: bool = True
    shortlist_size: int = 1
    exploration: float = 0.1
    estimated_input_tokens: int = 1000
    estimated_output_tokens: int = 1000

    def __post_init__(self):
        weights = (self.objective_weight, self.preference_weight, self.cost_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
            raise ValueError('utility weights must be finite, nonnegative, and not all zero')
        if not math.isfinite(self.exploration) or not 0 < self.exploration <= 1:
            raise ValueError('exploration must be in (0, 1] to prevent permanent exclusion')
        if not isinstance(self.shortlist_size, int) or self.shortlist_size < 1:
            raise ValueError('shortlist_size must be a positive integer')
        if any(not isinstance(n, int) or n < 0 for n in
               (self.estimated_input_tokens, self.estimated_output_tokens)):
            raise ValueError('estimated token counts must be nonnegative integers')

    def select(self, org, pool, worker_ids, cell, topics, user_id, rng=None) -> tuple[str, dict]:
        rng = rng or random
        eligible = list(dict.fromkeys(worker_ids))
        if not eligible or any(mid not in pool for mid in eligible):
            raise ValueError('assignment requires a nonempty eligible model pool')
        history = org.tms.model_history(eligible, topics) if self.use_tms else {}
        candidates = eligible[:]
        if history:
            ranked = sorted(eligible, key=lambda mid: history[mid]['score'], reverse=True)
            threshold = history[ranked[min(self.shortlist_size, len(ranked)) - 1]]['score']
            candidates = [mid for mid in eligible if history[mid]['cold'] or history[mid]['score'] >= threshold]

        # Consume the same number of random draws for on/off controls.
        explore_draw = rng.random()
        explore_model = rng.choice(eligible)
        scores = {}
        for mid in eligible:
            p = org.belief.query(mid, cell)
            objective = rng.betavariate(p.alpha, p.beta)
            observed = org.preference.approval_rate(user_id, cell, mid) if self.use_preference else None
            preference = observed if observed is not None else .5
            info = pool[mid].info
            cost = (self.estimated_input_tokens * info.cost_per_1k_input +
                    self.estimated_output_tokens * info.cost_per_1k_output) / 1000
            if not math.isfinite(cost) or cost < 0:
                raise ValueError('model cost estimate must be finite and nonnegative')
            utility = (self.objective_weight * objective +
                       (self.preference_weight * preference if self.use_preference else 0) -
                       self.cost_weight * cost)
            scores[mid] = {'alpha': p.alpha, 'beta': p.beta, 'objective_draw': objective,
                           'preference_observed': observed, 'preference_value': preference,
                           'estimated_cost_usd': cost, 'utility': utility}
        exploring = explore_draw < self.exploration
        best = max(scores[mid]['utility'] for mid in candidates)
        ties = [mid for mid in candidates if scores[mid]['utility'] == best]
        chosen = explore_model if exploring else rng.choice(ties)
        return chosen, {'policy': asdict(self), 'user_id': user_id, 'cell': cell,
                        'topics': list(dict.fromkeys(topics)), 'eligible': eligible,
                        'tms_history': history, 'candidates': candidates,
                        'exploring': exploring, 'scores': scores, 'selected': chosen}
