import copy
import json

from PIL import Image, ImageDraw
import pytest

from tests.test_task5_synthesis import _evidence, _element
from vlm_engine.evidence_refinement import refine_critical_details
from vlm_interface import VLMResponse
from visual_evidence.notation import equivalent_math_spelling


MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"
INWARD = {"pattern": "radial", "arrowhead_location": "near_center", "direction": "toward_center",
          "center_description": "blue-green globe", "arrow_color": "green",
          "visible_evidence": "The pointed ends are next to the central globe."}


class Reader:
    def __init__(self, payload=INWARD, reason="stop"):
        self.calls = []
        self.payload = payload
        self.reason = reason

    def analyze(self, image, prompt, generation_params):
        self.calls.append((image.size, prompt, generation_params))
        return VLMResponse(json.dumps(self.payload), MODEL, 1, 200, 80, self.reason)


def snapshot():
    rays = _element("rays", "Field lines", "CT-03", kind="line_family")
    rays.update(description="Green radial arrows pointing outward from the globe.", bbox_normalized=[0.1, 0.2, 0.4, 0.7])
    globe = _element("globe", "Globe", "CT-03")
    globe["bbox_normalized"] = [0.2, 0.4, 0.3, 0.5]
    evidence = _evidence("CT-03", elements=[rays, globe], relations=[{
        "id": "edge", "source_id": "globe", "target_id": "rays", "relation": "points_to",
        "directed": True, "evidence_status": "supported",
    }])
    evidence["regions"][0]["bbox_normalized"] = [0.05, 0.1, 0.45, 0.8]
    return {"_error": False, "visual_evidence": evidence}


def image_path(tmp_path):
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (1280, 720), "black").save(path)
    return path


def test_focused_direction_replaces_claim_but_preserves_conflict_and_original(tmp_path):
    source = snapshot()
    original = copy.deepcopy(source)
    reader = Reader()
    result = refine_critical_details(source, image_path(tmp_path), reader)
    node = result["visual_evidence"]["regions"][0]["elements"][0]
    assert node["attributes"]["direction"] == "toward_center"
    assert "toward the center" in node["description"] and "outward" not in node["description"]
    assert node["evidence_status"] == "partially_supported"
    assert result["visual_evidence"]["relations"][0]["evidence_status"] == "ambiguous_or_conflicting"
    assert result["_focused_checks"][0]["changed"]
    assert result["_focused_checks"][0]["original_description"] == original["visual_evidence"]["regions"][0]["elements"][0]["description"]
    assert source == original
    assert len(reader.calls) == 1 and max(reader.calls[0][0]) == 1344
    assert reader.calls[0][2]["_response_schema"]["properties"]["pattern"]
    assert result["_focused_checks"][0]["model"] == MODEL


@pytest.mark.parametrize("payload,reason", [
    ({**INWARD, "direction": "away_from_center"}, "stop"),
])
def test_unresolved_or_incomplete_check_never_promotes_an_overview_direction(tmp_path, payload, reason):
    result = refine_critical_details(snapshot(), image_path(tmp_path), Reader(payload, reason))
    node = result["visual_evidence"]["regions"][0]["elements"][0]
    assert node["evidence_status"] == "ambiguous_or_conflicting"
    assert "could not be verified" in node["description"]
    assert not result["_focused_checks"][0]["usable"]


def test_verification_timeout_or_truncation_preserves_unverified_overview(tmp_path):
    source = snapshot()
    original = copy.deepcopy(source)
    result = refine_critical_details(source, image_path(tmp_path), Reader(INWARD, "length"))
    node = result["visual_evidence"]["regions"][0]["elements"][0]
    assert node["description"] == original["visual_evidence"]["regions"][0]["elements"][0]["description"]
    assert node["evidence_status"] == "partially_supported"
    assert result["_focused_checks"][0]["verification_failed"]
    assert source == original


def test_exact_read_does_not_receive_the_overview_candidate(tmp_path):
    old = "x = 43791"
    node = _element("eq", "Equation", "CT-02", kind="equation", exact_text=old)
    node["bbox_normalized"] = [0.1, 0.1, 0.8, 0.8]
    evidence = _evidence("CT-02", elements=[node], exact=[{"text": old, "notation": "math", "status": "supported", "source_ref": "eq"}])
    reader = Reader({"readable": True, "text": old, "uncertainty": ""})
    refine_critical_details({"visual_evidence": evidence}, image_path(tmp_path), reader)
    assert old not in reader.calls[0][1]


@pytest.mark.parametrize("arrowhead,status", [("target", "supported"), ("source", "ambiguous_or_conflicting"), ("unclear", "ambiguous_or_conflicting")])
def test_connector_direction_checks_preserve_both_anchors_without_reversing_edge(tmp_path, arrowhead, status):
    first = _element("first", "A", "CT-04")
    second = _element("second", "B", "CT-04")
    first["bbox_normalized"] = [0.1, 0.2, 0.2, 0.4]
    second["bbox_normalized"] = [0.7, 0.2, 0.8, 0.4]
    evidence = _evidence("CT-04", elements=[first, second], relations=[{
        "id": "arrow", "source_id": "first", "target_id": "second", "relation": "leads_to",
        "directed": True, "evidence_status": "supported",
    }])
    result = refine_critical_details({"visual_evidence": evidence}, image_path(tmp_path),
                                    Reader({"connected": True, "arrowhead_at": arrowhead, "visible_evidence": "arrow tip"}))
    edge = result["visual_evidence"]["relations"][0]
    assert edge["evidence_status"] == status
    assert edge["source_id"] == "first" and edge["target_id"] == "second"
    bounds = result["_focused_checks"][0]["crop_bounds"]
    assert bounds[0] <= 0.1 and bounds[2] >= 0.8


def test_code_reading_retains_literal_indentation_and_records_a_changed_operator(tmp_path):
    old = "if x > 2:\n    print(x)"
    corrected = "if x >= 2:\n    print(x)"
    node = _element("code", "Code block", "CT-08", kind="code", exact_text=old)
    node["bbox_normalized"] = [0.1, 0.1, 0.8, 0.8]
    evidence = _evidence("CT-08", elements=[node], exact=[{
        "text": old, "notation": "code", "status": "supported", "source_ref": "code",
    }])
    result = refine_critical_details({"visual_evidence": evidence}, image_path(tmp_path),
                                     Reader({"readable": True, "text": corrected, "uncertainty": ""}))
    assert result["visual_evidence"]["exact_content"][0]["text"] == old
    assert result["visual_evidence"]["exact_content"][0]["status"] == "ambiguous_or_conflicting"
    assert result["_focused_checks"][0]["original_text"] == old
    assert json.loads(result["_focused_checks"][0]["raw_response"])["text"] == corrected
    assert "exact_text" not in result["visual_evidence"]["regions"][0]["elements"][0]


def test_ownership_loss_between_checks_discards_local_updates(tmp_path):
    source = snapshot()
    region = copy.deepcopy(source["visual_evidence"]["regions"][0])
    region["id"] = "second-region"
    region["elements"] = [copy.deepcopy(region["elements"][0])]
    region["elements"][0].update(id="second-rays", bbox_normalized=[0.6, 0.2, 0.9, 0.7])
    region["bbox_normalized"] = [0.55, 0.1, 0.95, 0.8]
    source["visual_evidence"]["regions"].append(region)
    original = copy.deepcopy(source)
    calls = 0

    def ownership():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("lease lost")

    reader = Reader()
    with pytest.raises(RuntimeError, match="lease lost"):
        refine_critical_details(source, image_path(tmp_path), reader, ownership_check=ownership)
    assert len(reader.calls) == 1 and source == original


@pytest.mark.parametrize("overlap", [False, True])
def test_center_appearance_is_updated_only_for_a_unique_grounded_object(tmp_path, overlap):
    source = snapshot()
    objects = source["visual_evidence"]["regions"][0]["elements"]
    objects[1].update(kind="object", description="Stale overview colors")
    if overlap:
        second = copy.deepcopy(objects[1])
        second["id"] = "overlapping-object"
        objects.append(second)
    result = refine_critical_details(source, image_path(tmp_path), Reader())
    center = result["visual_evidence"]["regions"][0]["elements"][1]
    if overlap:
        assert center["description"] == "Stale overview colors"
        assert "center_ref" not in result["_focused_checks"][0]
    else:
        assert "blue-green globe" in center["description"]
        assert center["evidence_status"] == "partially_supported"
        assert result["_focused_checks"][0]["original_center_description"] == "Stale overview colors"
    assert source["visual_evidence"]["regions"][0]["elements"][1]["description"] == "Stale overview colors"


@pytest.mark.parametrize("first,second", [
    ("∫ 2x sin x dx", r"\int 2x \sin x dx"),
    ("∫ 2x sin x dx", r"\\int 2x \\sin x \\, dx"),
    ("x y", r"x\quad y"),
    ("∫ u dv = uv - ∫ v du", r"\\int u dv = uv - \\int v du"),
    ("I_1 ω_1 + I_2 ω_2", r"I_1 \omega_1 + I_2 \omega_2"),
])
def test_only_conventional_math_spellings_are_equivalent(first, second):
    assert equivalent_math_spelling(first, second)


@pytest.mark.parametrize("first,second", [
    ("x < 2", "x <= 2"), ("x - 2", "x + 2"), ("x_1", "x1"),
    ("(x+2)/(y-1)", "x+2/y-1"), ("sin x", "sinx"),
    (r"\vec{x}", "x"), ("10 cm", "10 m"), ("x^2", "x2"),
    ("x,y", "x y"),
])
def test_math_comparison_preserves_meaningful_source_differences(first, second):
    assert not equivalent_math_spelling(first, second)


def divided_image():
    image = Image.new("RGB", (400, 200), "black")
    draw = ImageDraw.Draw(image)
    draw.line((200, 0, 200, 199), fill="white", width=2)
    for y in range(20, 180, 20):
        draw.rectangle((20, y, 170, y + 3), fill=(110, 110, 110))
        draw.rectangle((230, y, 370, y + 3), fill=(110, 110, 110))
    return image


def test_visible_divider_recovers_only_the_uncovered_panel():
    from vlm_engine.region_grounding import uncovered_panels
    image = divided_image()
    evidence = {"regions": [{"bbox_normalized": [0.5, 0, 1, 1]}]}
    boxes = uncovered_panels(image, evidence)
    assert len(boxes) == 1 and boxes[0][0] == 0 and abs(boxes[0][2] - 0.5) < 0.02
    assert not uncovered_panels(image, {"regions": [{"bbox_normalized": [0, 0, 1, 1]}]})
    assert not uncovered_panels(image, {"regions": [
        {"bbox_normalized": [0, 0, 0.5, 1]},
        {"bbox_normalized": [0.5, 0, 1, 0.5]},
        {"bbox_normalized": [0.5, 0.5, 1, 1]},
    ]})
    assert not uncovered_panels(Image.new("RGB", (400, 200), "white"), evidence)
    assert not uncovered_panels(Image.new("RGB", (400, 200), "black"), evidence)


def test_new_panel_reading_is_partial_audited_and_does_not_invent_exact_text(tmp_path):
    image = divided_image()
    path = tmp_path / "divided.png"
    image.save(path)
    node = _element("part", "Existing right panel", "CT-06")
    evidence = _evidence("CT-06", elements=[node])
    evidence["regions"][0]["bbox_normalized"] = [0.5, 0, 1, 1]
    source = {"visual_evidence": evidence}
    original = copy.deepcopy(source)
    result = refine_critical_details(source, path, Reader({
        "description": "A repeated grid of small diagram blocks separated by horizontal lines.",
        "categories": ["CT-11"], "uncertainty": "Small labels cannot be read.",
    }))
    assert source == original
    recovered = result["visual_evidence"]["regions"][-1]
    assert recovered["position"] == "left panel" and recovered["evidence_status"] == "partially_supported"
    assert recovered["elements"][0].get("exact_text") is None
    assert result["_focused_checks"][0]["kind"] == "region"
    assert "CT-11" in result["visual_evidence"]["category_ids"]
    from visual_evidence import validate_visual_evidence
    assert validate_visual_evidence(result["visual_evidence"]).structurally_valid


def test_region_reader_gets_category_meanings_and_cannot_expand_to_all_categories(tmp_path):
    path = tmp_path / "divided.png"
    divided_image().save(path)
    evidence = _evidence("CT-06", elements=[_element("part", "Right panel", "CT-06")])
    evidence["regions"][0]["bbox_normalized"] = [0.5, 0, 1, 1]
    reader = Reader({"description": "Repeated small blocks.", "categories": [f"CT-{i:02d}" for i in range(1, 13)], "uncertainty": ""})
    result = refine_critical_details({"visual_evidence": evidence}, path, reader)
    assert len(result["visual_evidence"]["regions"]) == 1
    assert not result["_focused_checks"][0]["usable"]
    assert "CT-11 = observational image" in reader.calls[0][1]
