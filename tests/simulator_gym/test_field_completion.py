"""Field completion of simulator-predicted states (`env.complete_fields`).

These cases are the failure the 2026-07-29 runs actually hit: an
LLM-synthesized checker subscripts `note['tags']` across every note, the
simulator emits a newly created note without a `tags` key, and the resulting
KeyError scores a passing trajectory as failed.
"""

from qwen_agentworld.core.schemas import ToolCall
from qwen_agentworld.simulator_gym.env import ACTION_LOG_KEY, complete_fields, simulate_next_state
from qwen_agentworld.teacher.safe_predicate import evaluate_predicate


def test_new_object_missing_field_gets_sibling_neutral_value():
    prior = {"notes": [{"title": "Dinner Ideas", "content": "pasta", "tags": ["food"]}]}
    predicted = {
        "notes": [
            {"title": "Dinner Ideas", "content": "pasta", "tags": ["food"]},
            {"title": "Brainstormed Meals", "content": "Dinner Ideas"},  # no tags
        ]
    }
    completed = complete_fields(prior, predicted)
    assert completed["notes"][1]["tags"] == []


def test_existing_object_keeps_its_prior_value_not_a_blank():
    prior = {"notes": [{"title": "Dinner Ideas", "content": "pasta", "tags": ["food"]}]}
    predicted = {"notes": [{"title": "Dinner Ideas", "content": "pasta and salad"}]}
    completed = complete_fields(prior, predicted)
    assert completed["notes"][0]["tags"] == ["food"]


def test_deleted_object_stays_deleted():
    prior = {"notes": [{"title": "A", "tags": []}, {"title": "B", "tags": []}]}
    predicted = {"notes": [{"title": "A", "tags": []}]}
    completed = complete_fields(prior, predicted)
    assert [n["title"] for n in completed["notes"]] == ["A"]


def test_emptied_collection_is_not_repopulated():
    prior = {"notes": [{"title": "A", "tags": []}]}
    completed = complete_fields(prior, {"notes": []})
    assert completed["notes"] == []


def test_dropped_top_level_key_is_restored():
    prior = {"notes": [{"title": "A", "tags": []}], "folders": ["inbox"]}
    completed = complete_fields(prior, {"notes": [{"title": "A", "tags": []}]})
    assert completed["folders"] == ["inbox"]


def test_action_log_is_not_restored_here():
    """`record_action` owns the log; completing it too would append a stale
    copy before the new entry is written."""
    prior = {"notes": [], ACTION_LOG_KEY: [{"tool": "create_note", "arguments": {}}]}
    completed = complete_fields(prior, {"notes": []})
    assert ACTION_LOG_KEY not in completed


def test_nested_dict_state_is_completed_recursively():
    prior = {"workspace": {"notes": [{"title": "A", "tags": ["x"]}], "owner": "u1"}}
    predicted = {"workspace": {"notes": [{"title": "A"}]}}
    completed = complete_fields(prior, predicted)
    assert completed["workspace"]["notes"][0]["tags"] == ["x"]
    assert completed["workspace"]["owner"] == "u1"


def test_non_dict_prediction_passes_through():
    assert complete_fields({"notes": []}, ["not", "a", "state"]) == ["not", "a", "state"]


def test_checker_predicate_no_longer_raises_keyerror():
    """The end-to-end point of the fix, stated as the checker sees it."""
    prior = {"notes": [{"title": "Dinner Ideas", "content": "pasta", "tags": []}]}
    predicted = {
        "notes": [
            {"title": "Dinner Ideas", "content": "pasta", "tags": ["brainstorm"]},
            {"title": "Brainstormed Meals", "content": "Dinner Ideas"},
        ]
    }
    predicate = (
        "any(n['title'] == 'Dinner Ideas' and 'brainstorm' in n['tags'] for n in state['notes']) "
        "and any(n['title'] == 'Brainstormed Meals' and n['tags'] == [] for n in state['notes'])"
    )
    try:
        evaluate_predicate(predicate, predicted)
    except KeyError:
        pass
    else:
        raise AssertionError("expected the uncompleted state to raise, or this test proves nothing")

    assert evaluate_predicate(predicate, complete_fields(prior, predicted)) is True


class _StubSimulator:
    def __init__(self, content):
        self._content = content

    def chat(self, messages, max_tokens=None, tools=None):
        class _R:
            content = self._content
            tool_calls = None

        _R.content = self._content
        return _R()


def test_simulate_next_state_completes_before_returning():
    simulator = _StubSimulator('{"next_state": {"notes": [{"title": "A"}]}}')
    state = {"notes": [{"title": "A", "tags": ["x"]}]}
    result = simulate_next_state(simulator, state, ToolCall(tool_name="list_notes", arguments={}))
    assert result["notes"][0]["tags"] == ["x"]
