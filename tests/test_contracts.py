"""Ordering and missing production boundaries, without benchmark tasks."""
import pytest

from alo_cloud.contracts import TaskIdentity, m1_units, m2_arm_order, m2_stage_order
from alo_cloud.errors import IntegrationNotConfigured
from alo_cloud.interfaces import BenchmarkEvaluator, ProviderTransport, SandboxBackend


def test_m1_order_uses_track_position_and_all_initial_rounds_first():
    tasks = [TaskIdentity("SWE-bench_Lite", "alpha"), TaskIdentity("ProgramBench", "beta"),
             TaskIdentity("SWE-bench_Lite", "gamma"), TaskIdentity("ProgramBench", "delta")]
    units = m1_units(tasks, ("G1", "G2", "G3"))
    for index, task in enumerate(tasks):
        block = units[index * 6:(index + 1) * 6]
        expected = ("G1", "G2", "G3") if index < 2 else ("G2", "G3", "G1")
        assert [x.phase for x in block] == ["B0"] * 3 + ["B1-self"] * 3
        assert tuple(x.condition for x in block[:3]) == tuple(x.condition for x in block[3:]) == expected
        assert all(x.task == task and x.order == expected for x in block)


def test_duplicate_task_does_not_create_extra_independent_units():
    task = TaskIdentity("ProgramBench", "duplicate")
    with pytest.raises(ValueError, match="unique"):
        m1_units([task, task], ("G1", "G2", "G3"))


def test_m2_paired_order_and_global_stages():
    assert m2_arm_order("persist", 0) == m2_arm_order("reset", 1) == ("persist", "reset")
    assert m2_stage_order() == ("all_learning_streams", "stateless_references", "terminal_no_update_probes", "independent_scoring")
    with pytest.raises(ValueError):
        m2_arm_order("persist", True)


@pytest.mark.parametrize("boundary,args", [(ProviderTransport().complete_for_model, ("model", [])),
    (SandboxBackend().command, (["anything"],)), (BenchmarkEvaluator().score_sealed, (None, None))])
def test_production_boundaries_cannot_silently_execute(boundary, args):
    with pytest.raises(IntegrationNotConfigured):
        boundary(*args)
