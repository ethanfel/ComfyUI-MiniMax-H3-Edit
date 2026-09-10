from __future__ import annotations

import json
from pathlib import Path

WORKFLOW = Path(__file__).parents[1] / "example_workflows" / "H3_Semantic_Room_Object_Study.json"
PRECISE_CUT_WORKFLOW = Path(__file__).parents[1] / "example_workflows" / "H3_Art_Room_Precise_Hard_Cuts.json"
STUDIO_TRAVEL_WORKFLOW = (
    Path(__file__).parents[1] / "example_workflows" / "H3_Empty_Drawing_Studio_15s_Travel.json"
)
STUDIO_TRAVEL_PROMPT = Path(__file__).parents[1] / "prompts" / "H3_Empty_Drawing_Studio_15s_Travel.txt"
GAZE_WORKFLOW = Path(__file__).parents[1] / "example_workflows" / "H3_Portrait_Eye_Direction_8_View.json"
GAZE_PROMPT = Path(__file__).parents[1] / "prompts" / "H3_Portrait_Eye_Direction_8_View.txt"


def test_semantic_room_object_workflow_is_complete_and_self_documenting():
    graph = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    by_type = {}
    for node in nodes.values():
        by_type.setdefault(node["type"], []).append(node)

    assert len(by_type["LoadImage"]) == 7
    assert len(by_type["AddH3EditReference"]) == 6
    assert len(by_type["H3EditOptions"]) == 1
    assert len(by_type["DecodeH3SceneCoverage"]) == 1

    options = by_type["H3EditOptions"][0]
    assert options["widgets_values"][0] == "scene coverage | room + object study"
    assert options["widgets_values"][1] is False

    encoder = by_type["TextEncodeH3Edit"][0]
    assert encoder["widgets_values"][1] == "generate | semantic Picture 1 (FL2VA)"
    assert encoder["widgets_values"][3:5] == [1344, 768]
    assert "<Picture 1> through <Picture 7>" in encoder["widgets_values"][0]
    assert "equal semantic evidence" in encoder["widgets_values"][0]
    assert "No picture is a master reference" in encoder["widgets_values"][0]
    assert "cream/off-white leather or vinyl cube ottoman" in encoder["widgets_values"][0]
    assert "The cube never spins, yaws, turns, or pivots" in encoder["widgets_values"][0]
    assert "roughly 25 to 45 percent of the image" in encoder["widgets_values"][0]
    assert "at least two recognizable room anchors" in encoder["widgets_values"][0]

    note = by_type["MarkdownNote"][0]["widgets_values"][0]
    assert "6 generated room-establishing views + 10 generated contextual views" in note
    assert "https://huggingface.co/PoopMan333/H3_Character_Sheet_Generator" in note

    link_ids = set()
    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in graph["links"]:
        assert link_id not in link_ids
        link_ids.add(link_id)
        origin = nodes[origin_id]
        target = nodes[target_id]
        assert link_id in origin["outputs"][origin_slot]["links"]
        assert target["inputs"][target_slot]["link"] == link_id
        assert origin["outputs"][origin_slot]["type"] == link_type
        assert target["inputs"][target_slot]["type"] == link_type

    assert graph["last_node_id"] == max(nodes)
    assert graph["last_link_id"] == max(link_ids)


def test_precise_art_room_workflow_uses_exact_continuous_cut_timeline():
    graph = json.loads(PRECISE_CUT_WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    by_type = {}
    for node in nodes.values():
        by_type.setdefault(node["type"], []).append(node)

    assert [node["widgets_values"][0] for node in by_type["LoadImage"]] == [
        "location/art_room/raw/3_regen.png",
        "location/art_room/raw/5_regen.png",
        "location/art_room/raw/1_regen.png",
        "location/art_room/raw/4_regen.png",
        "location/art_room/raw/2_regen.png",
    ]
    assert len(by_type["AddH3EditReference"]) == 4
    assert all(node["widgets_values"][0] == "semantic (Qwen only)" for node in by_type["AddH3EditReference"])

    options = by_type["H3EditOptions"][0]
    assert options["widgets_values"][0] == "scene coverage | cinematic hard cuts"
    assert options["widgets_values"][1] is True
    assert options["widgets_values"][2] == "scene coverage | 243-frame camera path"
    assert options["widgets_values"][7:12] == [8, 360, "clockwise / camera right", 5, False]

    encoder = by_type["TextEncodeH3Edit"][0]
    assert encoder["widgets_values"][1] == "generate | semantic Picture 1 (FL2VA)"
    assert encoder["widgets_values"][6] == "directed | frozen cinematic cuts"
    assert encoder["widgets_values"][9] == "scene coverage | 243-frame camera path"
    assert encoder["inputs"][14]["link"] is not None
    compiled_input = next(item for item in encoder["inputs"] if item["name"] == "compiled_prompt")
    assert compiled_input["link"] is not None

    compiled = by_type["PrimitiveStringMultiline"][0]["widgets_values"][0]
    assert "[Shot" not in compiled
    assert "Camera A is due south of C" in compiled
    assert "Camera H southeast of C" in compiled
    assert "No camera may be farther than 2.4 meters from C" in compiled
    assert "use no lens wider than 45 mm" in compiled
    assert "At 00:08.208, cut instantly" in compiled

    link_ids = set()
    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in graph["links"]:
        assert link_id not in link_ids
        link_ids.add(link_id)
        origin = nodes[origin_id]
        target = nodes[target_id]
        assert link_id in origin["outputs"][origin_slot]["links"]
        assert target["inputs"][target_slot]["link"] == link_id
        assert origin["outputs"][origin_slot]["type"] == link_type
        assert target["inputs"][target_slot]["type"] == link_type

    assert graph["last_node_id"] == max(nodes)
    assert graph["last_link_id"] == max(link_ids)


def test_empty_drawing_studio_workflow_uses_exact_anchored_travel_timeline():
    graph = json.loads(STUDIO_TRAVEL_WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    by_type = {}
    for node in nodes.values():
        by_type.setdefault(node["type"], []).append(node)

    assert len(by_type["LoadImage"]) == 1
    assert by_type["LoadImage"][0]["widgets_values"][0] == "krea2_anypaint_00001_.png"

    options = by_type["H3EditOptions"][0]
    assert options["widgets_values"][:3] == [
        "scene coverage | canonical camera path",
        True,
        "scene coverage | 362-frame camera path",
    ]
    assert options["widgets_values"][7:12] == [12, 270, "clockwise / camera right", 5, False]

    encoder = by_type["TextEncodeH3Edit"][0]
    assert encoder["widgets_values"][1] == "edit | strong scene anchor (FL2VA)"
    assert encoder["widgets_values"][6] == "directed | frozen scene coverage"
    assert encoder["widgets_values"][9] == "scene coverage | 362-frame camera path"
    assert encoder["inputs"][14]["link"] is None
    compiled_input = next(item for item in encoder["inputs"] if item["name"] == "compiled_prompt")
    assert compiled_input["link"] is not None

    compiled = by_type["PrimitiveStringMultiline"][0]["widgets_values"][0]
    assert compiled == STUDIO_TRAVEL_PROMPT.read_text(encoding="utf-8")
    assert "[Shot" not in compiled
    assert "Trees are visible outside the windows" in compiled
    assert "There is no final-frame reference" in compiled
    assert "<Picture 2>" not in compiled
    assert "The white ottoman is the primary tracked visual subject" in compiled
    assert "camera never tracks or orbits around the easel" in compiled
    assert "center C as the floor point exactly halfway" in compiled
    assert "one continuous 270-degree clockwise arc" in compiled
    assert "Never move counterclockwise" in compiled
    assert "opening camera is the farthest point" in compiled
    assert "closest approach to the easel" in compiled
    assert "Every one of the 362 frames is a potential cinematic capture" in compiled
    assert "Capture 12 reaches 270.000 degrees clockwise" in compiled
    assert "Finish naturally on the west side" in compiled
    assert "plain smooth neutral off-white" in compiled
    assert "closed plain white double sliding door" in compiled
    assert "never inherits the gray-green color" in compiled

    decoder = by_type["DecodeH3SceneCoverage"][0]
    assert decoder["widgets_values"][0] == 4
    assert decoder["outputs"][2]["links"] is not None
    all_frame_saver = next(node for node in by_type["SaveImage"] if node["id"] == 30)
    assert all_frame_saver["mode"] == 0

    link_ids = set()
    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in graph["links"]:
        assert link_id not in link_ids
        link_ids.add(link_id)
        origin = nodes[origin_id]
        target = nodes[target_id]
        assert link_id in origin["outputs"][origin_slot]["links"]
        assert target["inputs"][target_slot]["link"] == link_id
        assert origin["outputs"][origin_slot]["type"] == link_type
        assert target["inputs"][target_slot]["type"] == link_type

    assert graph["last_node_id"] == max(nodes)
    assert graph["last_link_id"] == max(link_ids)


def test_portrait_gaze_workflow_locks_the_head_and_extracts_eight_edge_targets():
    graph = json.loads(GAZE_WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    by_type = {}
    for node in nodes.values():
        by_type.setdefault(node["type"], []).append(node)

    assert len(by_type["LoadImage"]) == 1
    assert by_type["LoadImage"][0]["widgets_values"][0] == "portrait.png"

    options = by_type["H3EditOptions"][0]
    assert options["widgets_values"][:3] == [
        "scene coverage | canonical camera path",
        True,
        "scene coverage | 243-frame camera path",
    ]
    assert options["widgets_values"][7:12] == [9, 15, "clockwise / camera right", 5, False]

    encoder = by_type["TextEncodeH3Edit"][0]
    assert encoder["widgets_values"][1] == "edit | strong scene anchor (FL2VA)"
    assert encoder["widgets_values"][3:5] == [768, 1024]
    compiled_input = next(item for item in encoder["inputs"] if item["name"] == "compiled_prompt")
    assert compiled_input["link"] is not None

    compiled = by_type["PrimitiveStringMultiline"][0]["widgets_values"][0]
    assert compiled == GAZE_PROMPT.read_text(encoding="utf-8").rstrip("\n")
    assert "[Shot" not in compiled
    assert "The head lock is the highest-priority constraint" in compiled
    assert "Only the two eyeballs move" in compiled
    assert "image-space from the viewer's perspective" in compiled
    assert "midpoint of the left edge of the camera image" in compiled
    assert "upper-left corner of the camera image" in compiled
    assert "midpoint of the top edge of the camera image" in compiled
    assert "upper-right corner of the camera image" in compiled
    assert "midpoint of the right edge of the camera image" in compiled
    assert "lower-right corner of the camera image" in compiled
    assert "midpoint of the bottom edge of the camera image" in compiled
    assert "lower-left corner of the camera image" in compiled

    decoder = by_type["DecodeH3SceneCoverage"][0]
    assert decoder["widgets_values"] == [3, 8, "black"]
    slicer = by_type["ImageFromBatch"][0]
    assert slicer["widgets_values"] == [1, 8]
    direction_saver = next(node for node in by_type["SaveImage"] if node["id"] == 27)
    assert direction_saver["inputs"][0]["link"] == slicer["outputs"][0]["links"][0]

    link_ids = set()
    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in graph["links"]:
        assert link_id not in link_ids
        link_ids.add(link_id)
        origin = nodes[origin_id]
        target = nodes[target_id]
        assert link_id in origin["outputs"][origin_slot]["links"]
        assert target["inputs"][target_slot]["link"] == link_id
        assert origin["outputs"][origin_slot]["type"] == link_type
        assert target["inputs"][target_slot]["type"] == link_type

    assert graph["last_node_id"] == max(nodes)
    assert graph["last_link_id"] == max(link_ids)
