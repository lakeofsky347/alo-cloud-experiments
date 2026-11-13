"""Official CNY list prices and the conservative charge used before dispatch.

Rates were read from each vendor's Chinese pricing page on 2026-10-31.
A model missing from this table, or marked unverified, cannot be dispatched.
``charge_microcny`` rounds up to the next microcny. If a response has no
cache-hit split, every input token is billed at the cache-miss price.
DeepSeek publishes a peak rate and a lower off-peak rate for ``deepseek-flash``.
The charge uses the peak rate so a request during peak hours cannot settle
above the amount this ledger reserved.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
import json


RETRIEVED_ON = "2026-10-31"
MICROCNY_PER_CNY = 1_000_000


class UnpricedModel(ValueError):
    """No verified official list price. Refuse the dispatch."""


def _band(hit, miss, output):
    return {
        "cache_hit_input_cny_per_million": hit,
        "cache_miss_input_cny_per_million": miss,
        "output_cny_per_million": output,
    }


# Charge rates are CNY per 1,000,000 tokens. Off-peak DeepSeek numbers are
# recorded and are not the rate ``charge_microcny`` applies.
PRICE_TABLE = {
    "deepseek-flash": {
        "verified": True,
        "retrieved_on": RETRIEVED_ON,
        "source_url": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing",
        "product": "realtime chat completions",
        "charge_band": "peak",
        "charge": _band("0.04", "2", "8"),
        "off_peak_not_used_for_charge": _band("0.02", "1", "4"),
        "notes": (
            "Official pricing page lists model name deepseek-flash. "
            "Peak is Beijing time Monday-Friday 09:00-12:00 and 14:00-18:00, "
            "excluding Chinese public holidays; other hours are half price. "
            "This table charges the peak row."
        ),
    },
    "mimo-v2.6-flash": {
        "verified": True,
        "retrieved_on": RETRIEVED_ON,
        "source_url": "https://mimo.mi.com/docs/zh-CN/pricing",
        "model_card_url": "https://mimo.mi.com/models/zh-CN/mimo-v2.6-flash",
        "product": "realtime pay-as-you-go, domestic CNY",
        "charge_band": "realtime",
        "charge": _band("0.02", "1", "2"),
        "notes": (
            "Realtime row. Batch is half of realtime and is a different product. "
            "Token Plan is not this channel."
        ),
    },
    "mimo-v2.6-pro": {
        "verified": True,
        "retrieved_on": RETRIEVED_ON,
        "source_url": "https://mimo.mi.com/docs/zh-CN/pricing",
        "model_card_url": "https://mimo.mi.com/models/zh-CN/mimo-v2.6-pro",
        "product": "realtime pay-as-you-go, domestic CNY",
        "charge_band": "realtime",
        "charge": _band("0.025", "3", "6"),
        "notes": (
            "Realtime row for mimo-v2.6-pro. UltraSpeed and batch prices are "
            "different products and are not used."
        ),
    },
    "glm-5.3-flash": {
        "verified": True,
        "retrieved_on": RETRIEVED_ON,
        "source_url": "https://docs.bigmodel.cn/cn/guide/start/pricing",
        "product": "pay-as-you-go list price",
        "charge_band": "list",
        "charge": _band("0.23", "0.8", "2.8"),
        "cache_storage_cny_per_million_tokens_hour": "0",
        "notes": (
            "Pricing-guide list price: input 0.8, cache hit 0.23, output 2.8 CNY "
            "per million tokens. The same page says cache storage is currently free, "
            "so storage adds nothing. A temporary discount is not used; the higher "
            "listed unit price is the charge."
        ),
    },
}


def verified_models():
    return tuple(model for model, row in PRICE_TABLE.items() if row.get("verified"))


def charge_if_priced(model, usage):
    """Charge a verified list price, or return None when the model has none.

    Foxrouter models are not in ``PRICE_TABLE``. Callers that must keep going
    without a CNY price use this. ``charge_microcny`` still refuses them.
    """
    row = PRICE_TABLE.get(model)
    if not row or not row.get("verified"):
        return None
    return charge_microcny(model, usage)


def _rates(model):
    row = PRICE_TABLE.get(model)
    if not row or not row.get("verified"):
        raise UnpricedModel(str(model))
    return row["charge"]


def _nonnegative_int(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid_{label}")
    return value


def charge_microcny(model, usage):
    """Ceil the official list-price charge for one completion.

    ``usage`` uses ``input_tokens`` and ``output_tokens``. ``cache_hit_tokens``
    may be an int. ``None`` or a missing key means the provider did not report
    a split, and the whole input is charged at the cache-miss price.
    ``reasoning_tokens`` is ignored because it is already inside ``output_tokens``.
    """
    if not isinstance(usage, dict):
        raise ValueError("usage_must_be_mapping")
    rates = _rates(model)
    incoming = _nonnegative_int(usage.get("input_tokens"), "input_tokens")
    outgoing = _nonnegative_int(usage.get("output_tokens"), "output_tokens")
    hit = usage.get("cache_hit_tokens", None)
    if hit is None:
        hit_tokens, miss_tokens = 0, incoming
    else:
        hit_tokens = _nonnegative_int(hit, "cache_hit_tokens")
        if hit_tokens > incoming:
            raise ValueError("cache_split_exceeds_input")
        miss_tokens = incoming - hit_tokens
    total = (
        Decimal(hit_tokens) * Decimal(rates["cache_hit_input_cny_per_million"])
        + Decimal(miss_tokens) * Decimal(rates["cache_miss_input_cny_per_million"])
        + Decimal(outgoing) * Decimal(rates["output_cny_per_million"])
    )
    return int(total.to_integral_value(rounding=ROUND_CEILING))


def request_ceiling_microcny(model, max_prompt_tokens, max_completion_tokens):
    """Worst-case hold: the whole prompt at cache-miss, plus the output cap."""
    if type(max_prompt_tokens) is not int or type(max_completion_tokens) is not int:
        raise ValueError("ceiling_token_counts_must_be_int")
    if max_prompt_tokens < 1 or max_completion_tokens < 1:
        raise ValueError("ceiling_token_counts_must_be_positive")
    amount = charge_microcny(model, {
        "input_tokens": max_prompt_tokens,
        "output_tokens": max_completion_tokens,
        "cache_hit_tokens": None,
    })
    if amount < 1:
        raise UnpricedModel(str(model))
    return amount


def conservative_prompt_tokens(messages):
    """Byte length of the JSON messages. BPE tokens are at most this many."""
    raw = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return max(1, len(raw))


def ceiling_for_messages(model, messages, max_completion_tokens):
    return request_ceiling_microcny(model, conservative_prompt_tokens(messages), max_completion_tokens)
