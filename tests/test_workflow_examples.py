from __future__ import annotations

import json
from pathlib import Path

WORKFLOW = Path(__file__).parents[1] / "example_workflows" / "H3_Semantic_Room_Object_Study.json"


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
