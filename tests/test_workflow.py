from __future__ import annotations

import json
from pathlib import Path

WORKFLOW = Path(__file__).parents[1] / "example_workflows" / "H3_Edit_Mixed_References.json"
CHARACTER_WORKFLOW = Path(__file__).parents[1] / "example_workflows" / "H3_Character_Sheet_6_Panel.json"
CLOTHING_WORKFLOW = Path(__file__).parents[1] / "example_workflows" / "H3_Character_Clothing_6_View.json"
EIGHT_VIEW_WORKFLOW = Path(__file__).parents[1] / "example_workflows" / "H3_Character_Sheet_8_View.json"
EIGHT_VIEW_PROMPT = Path(__file__).parents[1] / "prompts" / "H3_Character_Sheet_8_View.txt"
DIRECTED_WORKFLOW = Path(__file__).parents[1] / "example_workflows" / "H3_Directed_Transformations.json"


def _load_validated_workflow(path: Path):
    graph = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}

    assert graph["last_node_id"] == max(nodes)
    assert graph["last_link_id"] == max(links)
    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in links.values():
        assert origin_id in nodes, link_id
        assert target_id in nodes, link_id
        assert nodes[origin_id]["outputs"][origin_slot]["type"] == link_type
        assert nodes[target_id]["inputs"][target_slot]["type"] == link_type
    return graph, nodes


def test_example_workflow_links_are_internally_consistent():
    _graph, nodes = _load_validated_workflow(WORKFLOW)

    assert len(nodes) == 18
    assert nodes[6]["type"] == "TextEncodeH3Edit"
    assert nodes[13]["type"] == "DecodeH3SingleFrame"
    assert nodes[16]["type"] == "AddH3EditReference"
    assert nodes[18]["type"] == "AddH3EditReference"

    edit = nodes[6]
    assert edit["widgets_values"][1] == "edit | strong scene anchor (FL2VA)"
    assert edit["widgets_values"][2] == "none (source only)"
    assert edit["widgets_values"][9] == "recommended | 5-frame context -> 1 image"
    assert edit["inputs"][2]["link"] is not None
    assert edit["inputs"][4]["name"] == "primary_image_role"
    assert edit["inputs"][12]["link"] is None
    assert edit["inputs"][13]["name"] == "quality_profile"
    assert edit["inputs"][14]["name"] == "reference_stack"
    assert edit["inputs"][14]["link"] is not None

    semantic = nodes[16]
    native = nodes[18]
    assert semantic["widgets_values"][0] == "semantic (Qwen only)"
    assert semantic["inputs"][4]["link"] is None
    assert native["widgets_values"][0] == "native (Qwen + VAE ref)"
    assert native["inputs"][4]["link"] is not None
    assert "<Picture 2>" in edit["widgets_values"][0]
    assert "<Picture 3>" in edit["widgets_values"][0]


def test_character_sheet_workflow_uses_ref2va_orbit_and_pack_decoder():
    _graph, nodes = _load_validated_workflow(CHARACTER_WORKFLOW)

    assert len(nodes) == 20
    assert "ref2va" in nodes[1]["widgets_values"][0]
    assert nodes[7]["type"] == "AddH3EditReference"
    assert nodes[8]["type"] == "AddH3EditReference"
    assert nodes[9]["type"] == "TextEncodeH3Edit"
    assert nodes[15]["type"] == "DecodeH3CharacterSheet"

    encoder = nodes[9]
    assert encoder["widgets_values"][1] == "generate | native Picture 1 (REF2VA)"
    assert encoder["widgets_values"][9] == "character sheet | 6 panels / 124-frame orbit"
    assert encoder["inputs"][14]["link"] is not None
    assert all(f"<Picture {ordinal}>" in encoder["widgets_values"][0] for ordinal in (1, 2, 3))
    assert nodes[7]["widgets_values"][0] == "native (Qwen + VAE ref)"
    assert nodes[8]["widgets_values"][0] == "native (Qwen + VAE ref)"

    decoder = nodes[15]
    assert decoder["widgets_values"] == ["auto from encoded profile", 6, "black"]
    assert decoder["outputs"][0]["name"] == "sheet"
    assert decoder["outputs"][1]["name"] == "selected_views"
    assert decoder["outputs"][2]["name"] == "all_frames"
    assert nodes[18]["mode"] == 4
    assert nodes[19]["mode"] == 4

    credit = nodes[20]
    assert credit["type"] == "MarkdownNote"
    assert "C_Nugget" in credit["widgets_values"][0]
    assert "https://huggingface.co/PoopMan333/H3_Character_Sheet_Generator" in credit["widgets_values"][0]


def test_clothing_character_workflow_keeps_fixed_head_to_knee_framing():
    _graph, nodes = _load_validated_workflow(CLOTHING_WORKFLOW)

    assert len(nodes) == 21
    assert "ref2va" in nodes[1]["widgets_values"][0]
    assert nodes[21]["type"] == "H3EditOptions"
    assert nodes[21]["widgets_values"][0] == "character sheet | clothing 6 views"
    assert nodes[21]["widgets_values"][1] is False

    encoder = nodes[9]
    assert encoder["widgets_values"][1] == "generate | native Picture 1 (REF2VA)"
    assert encoder["widgets_values"][6] == "character sheet | clothing head-to-knee"
    assert encoder["widgets_values"][9] == "character sheet | 6 panels / 124-frame orbit"
    assert encoder["inputs"][14]["link"] is not None
    assert encoder["inputs"][20]["name"] == "options"
    assert encoder["inputs"][20]["link"] is not None
    assert "rear construction" in encoder["widgets_values"][0]

    assert nodes[7]["widgets_values"][0] == "native (Qwen + VAE ref)"
    assert nodes[8]["widgets_values"][0] == "native (Qwen + VAE ref)"
    decoder = nodes[15]
    assert decoder["widgets_values"] == ["auto from encoded profile", 6, "black"]
    assert "head-to-knee" in decoder["title"]

    instructions = nodes[20]["widgets_values"][0]
    assert "never zooms" in instructions
    assert "frames **2, 21, 42, 63, 84, and 113**" in instructions
    assert "without numbered shot tags" in instructions
    assert "C_Nugget" in instructions


def test_eight_view_character_workflow_uses_long_body_detail_face_sequence():
    _graph, nodes = _load_validated_workflow(EIGHT_VIEW_WORKFLOW)

    assert len(nodes) == 21
    assert "ref2va" in nodes[1]["widgets_values"][0]
    assert nodes[21]["type"] == "H3EditOptions"
    assert nodes[21]["widgets_values"][0] == "character sheet | canonical 8 views"
    assert nodes[21]["widgets_values"][1] is False

    encoder = nodes[9]
    assert encoder["widgets_values"][1] == "generate | native Picture 1 (REF2VA)"
    assert encoder["widgets_values"][6] == "character sheet | 8-view body detail"
    assert encoder["widgets_values"][9] == "character sheet | 8 panels / 171-frame sequence"
    assert encoder["inputs"][14]["link"] is not None
    assert encoder["inputs"][20]["name"] == "options"
    assert encoder["inputs"][20]["link"] is not None
    assert "adult character" in encoder["widgets_values"][0]

    decoder = nodes[15]
    assert decoder["widgets_values"] == ["auto from encoded profile", 6, "black"]
    assert "eight calibrated" in decoder["title"]

    instructions = nodes[20]["widgets_values"][0]
    assert "171-frame" in instructions
    assert "frames **2, 21, 42, 63, 84, 108, 131, and 160**" in instructions
    assert "waist-to-knee" in instructions
    assert "never stops" in instructions
    assert "without numbered shot tags" in instructions
    assert "C_Nugget" in instructions


def test_standalone_eight_view_prompt_matches_decoder_schedule():
    prompt = EIGHT_VIEW_PROMPT.read_text(encoding="utf-8")
    sections = [
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ]

    assert [prompt.index(section) for section in sections] == sorted(prompt.index(section) for section in sections)
    assert all(f"<Picture {ordinal}>" in prompt for ordinal in (1, 2, 3))
    assert "171-frame target" in prompt
    assert "decoder frame 84" in prompt
    assert "decoder frame 108" in prompt
    assert "camera never stops during this detail orbit" in prompt
    assert "00:05.458" in prompt
    assert "00:06.667" in prompt
    assert "[Shot" not in prompt


def test_directed_workflow_switches_three_tasks_and_selects_settled_tail():
    _graph, nodes = _load_validated_workflow(DIRECTED_WORKFLOW)

    assert len(nodes) == 16
    assert "fl2va" in nodes[1]["widgets_values"][0]
    assert nodes[6]["type"] == "AddH3EditReference"
    assert nodes[7]["type"] == "TextEncodeH3Edit"
    assert nodes[13]["type"] == "DecodeH3SingleFrame"

    encoder = nodes[7]
    assert encoder["widgets_values"][1] == "edit | strong scene anchor (FL2VA)"
    assert encoder["widgets_values"][6] == "directed | re-pose character"
    assert encoder["widgets_values"][9] == "directed change | 39-frame settle -> 1 image"
    assert encoder["inputs"][14]["link"] is not None
    assert "<Picture 2>" in encoder["widgets_values"][0]
    assert nodes[6]["widgets_values"][0] == "semantic (Qwen only)"
    assert nodes[8]["widgets_values"] == ["euler"]
    assert nodes[9]["widgets_values"] == ["linear_quadratic", 25, 1.0]

    instructions = nodes[16]["widgets_values"][0]
    assert "directed | re-pose character" in instructions
    assert "directed | character swap" in instructions
    assert "directed | new camera angle" in instructions
    assert "tail frames 34–38" in instructions
    assert "C_Nugget" in instructions
    assert "https://huggingface.co/PoopMan333/H3_Character_Sheet_Generator" in instructions
