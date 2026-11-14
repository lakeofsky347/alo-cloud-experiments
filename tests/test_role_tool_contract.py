"""Planner-written role contracts: alias mapping, enforcement, and fallback."""
from types import SimpleNamespace

import pytest

from alo_cloud.adapters import (
    ROLE_CONTRACT_ENFORCED,
    ROLE_CONTRACT_UNRESTRICTED_EMPTY,
    ROLE_CONTRACT_UNRESTRICTED_UNKNOWN,
    ROLE_TOOL_ALIASES,
    resolve_role_tool_contract,
    role_tool_contract,
)
from alo_cloud.public_tools import TOOLS
from alo_cloud.review_isolation import role_is_review


def _role(tools):
    return SimpleNamespace(tools=tools)


# The ten role contracts written by the planner in the first fox-alo-learn-v1
# units on 5a981f3 (2026-11-15), with the tools each one should enforce.
FIRST_UNITS = [
    (("read_file", "search_code", "edit_file"), ("read_file", "search", "edit_file")),
    (("read_file", "run_tests", "execute_command"), ("read_file", "command")),
    (("shell", "file_editor", "test_runner"), ("command", "read_file", "edit_file", "write_file")),
    (("shell", "file_editor", "test_runner"), ("command", "read_file", "edit_file", "write_file")),
    (("shell", "file_editor", "test_runner"), ("command", "read_file", "edit_file", "write_file")),
    (("shell", "file_editor", "test_runner"), ("command", "read_file", "edit_file", "write_file")),
    (("search_code", "read_file", "edit_file"), ("search", "read_file", "edit_file")),
    (("search_code", "read_file", "run_tests"), ("search", "read_file", "command")),
    (("search_code", "read_file", "apply_patch", "run_tests"),
     ("search", "read_file", "edit_file", "write_file", "command")),
    (("read_file", "run_tests"), ("read_file", "command")),
]


@pytest.mark.parametrize("declared,expected", FIRST_UNITS)
def test_first_unit_mixed_contracts_are_mapped_and_enforced(declared, expected):
    contract = resolve_role_tool_contract(_role(declared))
    assert contract["mode"] == ROLE_CONTRACT_ENFORCED
    assert contract["mapped"] == expected
    assert contract["unknown"] == []
    assert contract["declared"] == list(declared)
    assert role_tool_contract(_role(declared)) == expected


def test_first_units_all_enforce_and_keep_review_semantics():
    modes = [resolve_role_tool_contract(_role(d))["mode"] for d, _ in FIRST_UNITS]
    assert modes.count(ROLE_CONTRACT_ENFORCED) == 10
    # A verifier limited to reading and tests stays a review role (#34).
    assert role_is_review(role_tool_contract(_role(("read_file", "run_tests"))))
    assert not role_is_review(role_tool_contract(_role(("read_file", "search_code", "edit_file"))))


def test_placeholder_maps_to_the_full_public_set():
    contract = resolve_role_tool_contract(_role(("public_tools",)))
    assert contract["mode"] == ROLE_CONTRACT_ENFORCED
    assert set(contract["mapped"]) == set(TOOLS)
    assert contract["restricts"] is False


def test_all_invented_names_fall_back_to_unrestricted():
    contract = resolve_role_tool_contract(_role(("magic_wand", "telepathy")))
    assert contract["mode"] == ROLE_CONTRACT_UNRESTRICTED_UNKNOWN
    assert contract["mapped"] is None
    assert contract["unknown"] == ["magic_wand", "telepathy"]
    assert role_tool_contract(_role(("magic_wand",))) is None


def test_one_unknown_name_falls_back_instead_of_narrowing():
    contract = resolve_role_tool_contract(_role(("read_file", "edit_file", "deploy_service")))
    assert contract["mode"] == ROLE_CONTRACT_UNRESTRICTED_UNKNOWN
    assert contract["mapped"] is None
    assert contract["unknown"] == ["deploy_service"]


def test_exact_public_tool_contracts_are_enforced_unchanged():
    assert role_tool_contract(_role(("command",))) == ("command",)
    assert role_tool_contract(_role(("read_file", "edit_file"))) == ("read_file", "edit_file")
    contract = resolve_role_tool_contract(_role(TOOLS))
    assert contract["mapped"] == tuple(TOOLS)
    assert contract["restricts"] is False


def test_names_are_normalized():
    assert role_tool_contract(_role(("Run-Tests", " Search Code "))) == ("command", "search")


def test_empty_contract_is_still_unrestricted():
    for role in (_role(()), _role(None), SimpleNamespace()):
        contract = resolve_role_tool_contract(role)
        assert contract["mode"] == ROLE_CONTRACT_UNRESTRICTED_EMPTY
        assert contract["mapped"] is None


def test_alias_table_only_targets_public_tools():
    for name, tools in ROLE_TOOL_ALIASES.items():
        assert name == name.lower() and name not in TOOLS
        assert tools and set(tools) <= set(TOOLS)


def test_dry_chain_records_each_role_contract(tmp_path):
    import json

    from alo_cloud.fox_learn import build_stream, dry_gateway, run_chain
    from alo_cloud.fox_pool import resolve_alo_pool

    built = build_stream(split="dev", task_ids=("sqlfluff__sqlfluff-1625",), turn_cap=8)
    chain = next(c for c in built["chains"] if c.chain_id == "persist-forward")
    run_chain(chain, tmp_path / "persist-forward", pool=resolve_alo_pool("fox"), live=False,
              sender=dry_gateway, sleep=lambda _seconds: None)
    rows = [json.loads(line) for line in
            (tmp_path / "persist-forward" / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    contracts = rows[0]["role_tool_contracts"]
    assert {row["node_id"] for row in contracts} == {"s1", "s2"}
    for row in contracts:
        assert row["declared"] == ["list_files"]
        assert row["mapped"] == ["list_files"]
        assert row["mode"] == ROLE_CONTRACT_ENFORCED
        assert row["restricts"] is True
        assert row["unknown"] == []
