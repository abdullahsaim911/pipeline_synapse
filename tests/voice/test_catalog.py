"""Module 1: the action catalog, its example sentences and the held-out set."""

import json
import os
import shutil
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from voice.catalog import ActionCatalog, CatalogError, normalize_sentence  # noqa: E402
from voice.catalog.catalog import DATA_DIR, MIN_EXAMPLES_PER_ACTION  # noqa: E402

MIN_EVAL_PER_ACTION = 3


@pytest.fixture(scope="module")
def catalog():
    return ActionCatalog.load()


@pytest.fixture
def data_copy(tmp_path):
    """A writable copy of the real catalog data, for tests that break it on purpose."""
    target = tmp_path / "catalog"
    shutil.copytree(DATA_DIR, target, ignore=shutil.ignore_patterns("__pycache__", "*.py"))
    return target


def edit_json(path, change):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    change(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ── The real data ────────────────────────────────────────────────────────────

def test_real_catalog_loads(catalog):
    assert len(catalog.ids()) >= 30
    assert "en" in catalog.languages


def test_every_action_has_enough_examples(catalog):
    examples = catalog.examples("en")
    for action_id in catalog.ids():
        assert len(examples.get(action_id, [])) >= MIN_EXAMPLES_PER_ACTION, action_id


def test_every_action_is_covered_by_the_held_out_set(catalog):
    counts = {}
    for item in catalog.eval_items("en"):
        counts[item.expect] = counts.get(item.expect, 0) + 1
    for action_id in catalog.ids():
        assert counts.get(action_id, 0) >= MIN_EVAL_PER_ACTION, action_id


def test_held_out_set_includes_things_the_app_cannot_do(catalog):
    assert sum(1 for i in catalog.eval_items("en") if i.expect == "unsupported") >= 5


def test_held_out_sentences_are_not_examples(catalog):
    example_keys = {normalize_sentence(s) for v in catalog.examples("en").values() for s in v}
    for item in catalog.eval_items("en"):
        assert normalize_sentence(item.text) not in example_keys, item.text


def test_every_action_with_a_handler_names_it(catalog):
    for action_id in catalog.ids():
        spec = catalog.get(action_id)
        assert spec.binding.strip(), action_id
        assert spec.binding_status in ("existing", "wrapper", "voice")


def test_voice_layer_actions_are_not_bound_to_the_app(catalog):
    for action_id in ("stop_speaking", "repeat_last_response", "cancel", "help", "confirm_yes", "confirm_no"):
        assert catalog.get(action_id).binding_status == "voice"


def test_destructive_or_costly_actions_require_confirmation(catalog):
    assert catalog.get("remove_item").risk == "confirm"


def test_actions_available_on_a_screen(catalog):
    library = {a.id for a in catalog.for_screen("library")}
    assert {"navigate_home", "play_item_audio", "open_item", "remove_item", "stop_speaking"} <= library
    assert not library & {"play_lecture", "next_moment", "search_lectures"}
    lecture = {a.id for a in catalog.for_screen("explanation")}
    assert {"play_lecture", "pause_explanation", "ask_question", "seek_lecture"} <= lecture


def test_eval_items_use_context_where_the_action_depends_on_it(catalog):
    pauses = [i for i in catalog.eval_items("en") if i.text == "pause"]
    assert {i.expect for i in pauses} == {"pause_lecture", "pause_explanation"}
    assert {i.playing for i in pauses} == {"lecture", "explanation"}


# ── Argument validation ──────────────────────────────────────────────────────

def test_valid_arguments_are_returned_clean(catalog):
    assert catalog.validate_args("seek_lecture", {"direction": "forward", "seconds": 30}) == {
        "direction": "forward", "seconds": 30,
    }
    assert catalog.validate_args("ask_question", {"question": "  why?  "}) == {"question": "why?"}
    assert catalog.validate_args("next_moment", {}) == {}


@pytest.mark.parametrize("action,args", [
    ("seek_lecture", {"direction": "forward"}),                                  # missing seconds
    ("seek_lecture", {"direction": "sideways", "seconds": 5}),                   # bad enum
    ("seek_lecture", {"direction": "forward", "seconds": 0}),                    # below min
    ("seek_lecture", {"direction": "forward", "seconds": 601}),                  # above max
    ("seek_lecture", {"direction": "forward", "seconds": True}),                 # bool is not an int
    ("seek_lecture", {"direction": "forward", "seconds": "30"}),                 # string is not an int
    ("next_moment", {"index": 1}),                                               # unexpected argument
    ("ask_question", {"question": "   "}),                                       # blank
    ("ask_question", {"question": "x" * 501}),                                   # too long
    ("set_theme", {"theme": "neon"}),                                            # bad enum
    ("go_to_moment", {"index": None}),                                           # null for a required argument
])
def test_invalid_arguments_are_rejected(catalog, action, args):
    with pytest.raises(CatalogError):
        catalog.validate_args(action, args)


def test_unknown_action_is_rejected(catalog):
    with pytest.raises(CatalogError):
        catalog.get("launch_missiles")
    assert catalog.unknown_actions(["next_moment", "nope"]) == ["nope"]


def test_args_must_be_an_object(catalog):
    with pytest.raises(CatalogError):
        catalog.validate_args("next_moment", ["x"])


# ── The loader refuses inconsistent data ─────────────────────────────────────

def test_loader_rejects_a_sentence_used_for_two_actions(data_copy):
    def change(data):
        data["examples"]["navigate_home"].append("Open Bookmarks!")  # same as navigate_bookmarks
    edit_json(data_copy / "examples" / "en.json", change)
    with pytest.raises(CatalogError, match="appears under both"):
        ActionCatalog.load(str(data_copy))


def test_loader_rejects_examples_for_an_unknown_action(data_copy):
    edit_json(data_copy / "examples" / "en.json", lambda d: d["examples"].update({"fly_away": ["fly away"]}))
    with pytest.raises(CatalogError, match="unknown actions"):
        ActionCatalog.load(str(data_copy))


def test_loader_rejects_a_complete_language_with_thin_actions(data_copy):
    edit_json(data_copy / "examples" / "en.json", lambda d: d["examples"].update({"navigate_home": ["go home"]}))
    with pytest.raises(CatalogError, match="fewer than"):
        ActionCatalog.load(str(data_copy))


def test_loader_allows_a_partial_language(data_copy):
    # A language that is not marked complete may cover only some actions.
    (data_copy / "examples" / "xx.json").write_text(
        json.dumps({"language": "xx", "complete": False, "examples": {"navigate_home": ["casa"]}}),
        encoding="utf-8",
    )
    assert "xx" in ActionCatalog.load(str(data_copy)).languages


def test_loader_rejects_an_eval_sentence_that_is_also_an_example(data_copy):
    def change(data):
        data["items"].append({"text": "go to the home screen", "expect": "navigate_home", "screen": "home"})
    edit_json(data_copy / "eval" / "en.json", change)
    with pytest.raises(CatalogError, match="also an example"):
        ActionCatalog.load(str(data_copy))


def test_loader_rejects_an_eval_item_on_a_screen_where_the_action_is_unavailable(data_copy):
    def change(data):
        data["items"].append({"text": "make it play now", "expect": "play_lecture", "screen": "library"})
    edit_json(data_copy / "eval" / "en.json", change)
    with pytest.raises(CatalogError, match="not available"):
        ActionCatalog.load(str(data_copy))


def test_loader_rejects_an_eval_item_with_bad_arguments(data_copy):
    def change(data):
        data["items"].append({
            "text": "jump a very long way", "expect": "seek_lecture",
            "args": {"direction": "forward", "seconds": 9999}, "screen": "explanation",
        })
    edit_json(data_copy / "eval" / "en.json", change)
    with pytest.raises(CatalogError, match="between"):
        ActionCatalog.load(str(data_copy))


def test_loader_rejects_an_eval_item_for_an_unknown_action(data_copy):
    def change(data):
        data["items"].append({"text": "do a barrel roll", "expect": "barrel_roll", "screen": "home"})
    edit_json(data_copy / "eval" / "en.json", change)
    with pytest.raises(CatalogError, match="unknown action"):
        ActionCatalog.load(str(data_copy))


def test_loader_rejects_duplicate_action_ids(data_copy):
    def change(data):
        data["actions"].append(dict(data["actions"][0]))
    edit_json(data_copy / "actions.json", change)
    with pytest.raises(CatalogError, match="duplicate action id"):
        ActionCatalog.load(str(data_copy))


def test_loader_rejects_a_bad_parameter_spec(data_copy):
    def change(data):
        for action in data["actions"]:
            if action["id"] == "set_theme":
                action["params"]["theme"]["values"] = []
    edit_json(data_copy / "actions.json", change)
    with pytest.raises(CatalogError, match="needs values"):
        ActionCatalog.load(str(data_copy))


def test_loader_rejects_an_unknown_screen(data_copy):
    def change(data):
        data["actions"][0]["screens"] = ["settings"]
    edit_json(data_copy / "actions.json", change)
    with pytest.raises(CatalogError, match="unknown screen"):
        ActionCatalog.load(str(data_copy))


def test_loader_reports_broken_json(data_copy):
    (data_copy / "actions.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CatalogError, match="not valid JSON"):
        ActionCatalog.load(str(data_copy))


# ── Frontend export ──────────────────────────────────────────────────────────

def test_frontend_module_describes_every_action(catalog):
    source = catalog.to_frontend_js()
    assert source.startswith("// GENERATED")
    start = source.index("export const ACTIONS = ") + len("export const ACTIONS = ")
    actions = json.loads(source[start:source.rindex(";")])
    assert set(actions) == set(catalog.ids())
    assert actions["seek_lecture"]["params"]["seconds"] == {"type": "int", "required": True, "min": 1, "max": 600}
    assert actions["remove_item"]["risk"] == "confirm"
    assert actions["pause_lecture"]["bindingStatus"] == "wrapper"


def test_frontend_copy_matches_the_catalog(catalog):
    """Set SYNAPSE_WEB_PATH to the frontend checkout to verify its generated copy is current."""
    web = os.environ.get("SYNAPSE_WEB_PATH")
    if not web:
        pytest.skip("SYNAPSE_WEB_PATH is not set")
    generated = os.path.join(web, "src", "voice", "actions.generated.js")
    with open(generated, encoding="utf-8", newline="") as f:
        assert f.read() == catalog.to_frontend_js()
