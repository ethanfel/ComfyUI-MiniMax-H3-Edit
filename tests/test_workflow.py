from __future__ import annotations

import json
from pathlib import Path

WORKFLOW = Path(__file__).parents[1] / "example_workflows" / "H3_Edit_Mixed_References.json"


def test_example_workflow_links_are_internally_consistent():
    graph = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}

    assert graph["last_node_id"] == max(nodes)
    assert graph["last_link_id"] == max(links)
    assert len(nodes) == 18
    assert nodes[6]["type"] == "TextEncodeH3Edit"
    assert nodes[13]["type"] == "DecodeH3SingleFrame"
    assert nodes[16]["type"] == "AddH3EditReference"
    assert nodes[18]["type"] == "AddH3EditReference"

    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in links.values():
        assert origin_id in nodes, link_id
        assert target_id in nodes, link_id
        assert nodes[origin_id]["outputs"][origin_slot]["type"] == link_type
        assert nodes[target_id]["inputs"][target_slot]["type"] == link_type

    edit = nodes[6]
    assert edit["widgets_values"][1] == "none (source only)"
    assert edit["widgets_values"][8] == "recommended | 5-frame context -> 1 image"
    assert edit["inputs"][2]["link"] is not None
    assert edit["inputs"][11]["link"] is None
    assert edit["inputs"][12]["name"] == "quality_profile"
    assert edit["inputs"][13]["name"] == "reference_stack"
    assert edit["inputs"][13]["link"] is not None

    semantic = nodes[16]
    native = nodes[18]
    assert semantic["widgets_values"][0] == "semantic (Qwen only)"
    assert semantic["inputs"][4]["link"] is None
    assert native["widgets_values"][0] == "native (Qwen + VAE ref)"
    assert native["inputs"][4]["link"] is not None
    assert "<Picture 2>" in edit["widgets_values"][0]
    assert "<Picture 3>" in edit["widgets_values"][0]
