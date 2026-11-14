"""Planner-written role contracts: placeholder tool names are not a restriction."""
from types import SimpleNamespace

from alo_cloud.adapters import role_tool_contract
from alo_cloud.public_tools import TOOLS


def _role(tools):
    return SimpleNamespace(tools=tools)


def test_placeholder_only_contract_means_full_public_set():
    assert role_tool_contract(_role(("public_tools",))) is None
    assert role_tool_contract(_role(["public_tools", "all_tools"])) is None


def test_placeholder_is_dropped_and_known_tools_are_enforced():
    assert role_tool_contract(_role(("public_tools", "read_file", "search"))) == ("read_file", "search")


def test_known_contracts_are_unchanged():
    assert role_tool_contract(_role(("command",))) == ("command",)
    assert role_tool_contract(_role(TOOLS)) == tuple(TOOLS)


def test_empty_contract_is_still_unrestricted():
    assert role_tool_contract(_role(())) is None
    assert role_tool_contract(_role(None)) is None
    assert role_tool_contract(SimpleNamespace()) is None
