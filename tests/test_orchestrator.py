"""Offline decomposer and orchestrator checks. Planner text is scripted."""
from alo.models.base import Completion, ModelInfo
from alo.org.context import Bandwidth
from alo.org.organization import Organization
from alo.runtime.decomposer import FALLBACK_TOPIC, decompose, fallback_subtasks
from alo.runtime.orchestrator import FULL_DUMP_BLOB, RAW_LOGS_BLOB, run_task
from alo_cloud.state import DEFAULT_TOPICS, export_learning_state


DEPLOY = """
subtasks:
  - id: s1
    goal: ship it
    deps: []
    topics: [coding]
    role:
      capability_reqs: [coding]
      epistemic_scope:
        include: [task_description, raw_logs, full_dump]
        exclude: [billing, credentials]
      authority: [deploy, spend]
      tools: []
      output_schema: free_text
      success_criteria: non_empty_output
"""


class ScriptedModel:
    def __init__(self, model_id, replies):
        self.info = ModelInfo(model_id, "official", model_id)
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt, *, system="", max_tokens=4096):
        self.prompts.append(prompt)
        return Completion(self.replies.pop(0), 1, 1, 0)


def test_fallback_topic_is_transferable_and_general_is_not(tmp_path):
    assert FALLBACK_TOPIC in DEFAULT_TOPICS
    assert "general" not in DEFAULT_TOPICS
    assert fallback_subtasks("task")[0].topics == (FALLBACK_TOPIC,)
    path = tmp_path / "org.sqlite3"
    org = Organization(str(path))
    org.tms.record_outcome("one", ("general",), True)
    org.tms.record_outcome("one", (FALLBACK_TOPIC,), False)
    org.conn.close()
    exported = export_learning_state(path, {"observations": {}}, allowed_models=("one",))
    assert {row["topic"] for row in exported["tms_outcomes"]} == {FALLBACK_TOPIC}
    assert exported["audit"]["excluded_tms_rows"] == 1


def test_governance_gated_authority_falls_back_instead_of_crashing(tmp_path):
    planner = ScriptedModel("planner", [DEPLOY])
    worker = ScriptedModel("deepseek-flash", ["worker output"])
    org = Organization(str(tmp_path / "run.sqlite3"))
    result = run_task(org, {"planner": planner, "deepseek-flash": worker}, "public task", planner_id="planner")
    org.conn.close()
    assert result.used_fallback_decomposition is True
    assert result.overall_success is True
    assert RAW_LOGS_BLOB not in worker.prompts[-1]
    assert "prod" not in worker.prompts[-1]
    assert FULL_DUMP_BLOB not in worker.prompts[-1]
    subtasks, used_fallback = decompose(ScriptedModel("planner", [DEPLOY]), "public task")
    assert used_fallback is True
    assert subtasks[0].role.authority == frozenset({"inspect"})
    assert subtasks[0].topics == (FALLBACK_TOPIC,)


def test_synthetic_context_is_opt_in_for_broad_views(tmp_path):
    planner = ScriptedModel("planner", ["not yaml"])
    worker = ScriptedModel("worker", ["done"])
    org = Organization(str(tmp_path / "broad.sqlite3"))

    def broad(_features):
        return Bandwidth.BROAD

    result = run_task(
        org, {"planner": planner, "worker": worker}, "public task", planner_id="planner",
        bandwidth_fn=broad, include_synthetic_context=True,
    )
    org.conn.close()
    assert result.used_fallback_decomposition is True
    assert RAW_LOGS_BLOB in worker.prompts[-1]
    assert "prod" in worker.prompts[-1]
    assert FULL_DUMP_BLOB not in worker.prompts[-1]
