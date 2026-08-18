from __future__ import annotations

import sys
import types

import pytest
import torch

from h3edit.nodes import (
    CHARACTER_SHEET_AUTO,
    CHARACTER_SHEET_SIX,
    NATIVE_SIZE_MATCH,
    PRIMARY_EDIT_ANCHOR,
    PRIMARY_NATIVE_REFERENCE,
    PRIMARY_SEMANTIC_REFERENCE,
    PROMPT_EDIT,
    QUALITY_CHARACTER_FOUR,
    QUALITY_CHARACTER_SIX,
    QUALITY_EXPERIMENTAL,
    QUALITY_MAXIMUM,
    QUALITY_RECOMMENDED,
    REFERENCE_NATIVE,
    REFERENCE_NONE,
    REFERENCE_SEMANTIC,
    AddH3EditReference,
    DecodeH3CharacterSheet,
    DecodeH3SingleFrame,
    TextEncodeH3Edit,
    semantic_target_size,
)


class FakeNestedTensor:
    def __init__(self, tensors):
        self.tensors = tuple(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


class FakeClip:
    def __init__(self):
        self.tokenize_calls = []

    def tokenize(self, prompt, **kwargs):
        self.tokenize_calls.append((prompt, kwargs))
        return {"prompt": prompt, **kwargs}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.ones((1, 1, 4)), {"tokens": tokens}]]


class FakeVAE:
    def __init__(self):
        self.encoded = []
        self.decoded = []

    def encode(self, image):
        self.encoded.append(image)
        return torch.zeros((1, 24, 1, image.shape[1] // 16, image.shape[2] // 16))

    def decode(self, latent):
        self.decoded.append(latent)
        frame_per_token = (1, 4, 4, 4, 4)
        frame_count = (
            1
            if latent.shape[2] == 1
            else sum(frame_per_token[index % len(frame_per_token)] for index in range(latent.shape[2]))
        )
        values = torch.arange(frame_count, dtype=torch.float32).view(1, frame_count, 1, 1, 1)
        return values.expand(latent.shape[0], frame_count, latent.shape[3] * 16, latent.shape[4] * 16, 3)


@pytest.fixture(autouse=True)
def fake_comfy_modules(monkeypatch):
    comfy = types.ModuleType("comfy")
    comfy.model_management = types.SimpleNamespace(intermediate_device=lambda: torch.device("cpu"))
    comfy.nested_tensor = types.SimpleNamespace(NestedTensor=FakeNestedTensor)
    monkeypatch.setitem(sys.modules, "comfy", comfy)

    node_helpers = types.ModuleType("node_helpers")

    def conditioning_set_values(conditioning, values):
        result = []
        for tensor, metadata in conditioning:
            result.append([tensor, {**metadata, **values}])
        return result

    node_helpers.conditioning_set_values = conditioning_set_values
    monkeypatch.setitem(sys.modules, "node_helpers", node_helpers)


def _image(height=768, width=512):
    return torch.rand((1, height, width, 3))


def _encode(
    reference_mode,
    reference_image=None,
    semantic_resolution=1024,
    quality_profile=QUALITY_RECOMMENDED,
    reference_stack=None,
    primary_image_role=PRIMARY_EDIT_ANCHOR,
):
    clip = FakeClip()
    vae = FakeVAE()
    output = TextEncodeH3Edit().encode(
        clip=clip,
        vae=vae,
        source_image=_image(),
        prompt="Add the glasses from Picture 2.",
        reference_mode=reference_mode,
        width=768,
        height=1344,
        source_fit="crop center",
        prompt_mode=PROMPT_EDIT,
        semantic_resolution=semantic_resolution,
        native_reference_size=NATIVE_SIZE_MATCH,
        primary_image_role=primary_image_role,
        reference_image=reference_image,
        quality_profile=quality_profile,
        reference_stack=reference_stack,
    )
    return clip, vae, output


def test_semantic_target_size_preserves_ratio_and_area_budget():
    width, height = semantic_target_size(_image(height=2000, width=1000), 1024)

    assert width % 32 == 0
    assert height % 32 == 0
    assert width * height <= 1024**2
    assert width / height == pytest.approx(0.5, abs=0.03)


def test_semantic_reference_is_qwen_only():
    clip, vae, (conditioning, latent, fitted, encoded_prompt, info) = _encode(
        REFERENCE_SEMANTIC,
        _image(height=400, width=800),
    )

    assert len(vae.encoded) == 1
    assert fitted.shape == (1, 1344, 768, 3)
    assert len(clip.tokenize_calls) == 1
    prompt, kwargs = clip.tokenize_calls[0]
    assert "<Picture 2> is a semantic visual guide only" in prompt
    assert prompt == encoded_prompt
    assert len(kwargs["minimax_ref_items"]) == 2
    semantic_image = kwargs["minimax_ref_items"][1]["data"]
    expected_width, expected_height = semantic_target_size(_image(height=400, width=800), 1024)
    assert semantic_image.shape == (1, expected_height, expected_width, 3)
    metadata = conditioning[0][1]
    assert metadata["minimax_frame_count"] == 5
    assert len(metadata["minimax_keyframes"]) == 1
    assert "minimax_refs" not in metadata
    video, audio = latent["samples"].unbind()
    assert video.shape == (1, 24, 2, 84, 48)
    assert audio.shape == (1, 32, 2, 8)
    assert latent["h3edit_requested_frames"] == 5
    assert latent["h3edit_natural_frames"] == 5
    assert "VAE skipped" in info
    assert "requested context 5 frames" in info
    assert "natural packet 5 frames" in info


def test_native_reference_gets_qwen_and_vae_transport():
    clip, vae, (conditioning, _latent, _fitted, _prompt, info) = _encode(
        REFERENCE_NATIVE,
        _image(height=400, width=800),
    )

    assert len(vae.encoded) == 2
    assert len(clip.tokenize_calls[0][1]["minimax_ref_items"]) == 2
    metadata = conditioning[0][1]
    assert len(metadata["minimax_keyframes"]) == 1
    assert len(metadata["minimax_refs"]) == 1
    ref = metadata["minimax_refs"][0]
    assert ref["kind"] == "image"
    assert ref["latent_h"] == vae.encoded[1].shape[1] // 16
    assert ref["latent_w"] == vae.encoded[1].shape[2] // 16
    assert "mixed with the frame-zero keyframe are experimental" in info


def test_semantic_generation_switch_removes_every_vae_anchor():
    clip, vae, (conditioning, _latent, prepared_primary, prompt, info) = _encode(
        REFERENCE_NONE,
        primary_image_role=PRIMARY_SEMANTIC_REFERENCE,
    )

    assert len(vae.encoded) == 0
    assert len(clip.tokenize_calls[0][1]["minimax_ref_items"]) == 1
    expected_width, expected_height = semantic_target_size(_image(), 1024)
    assert prepared_primary.shape == (1, expected_height, expected_width, 3)
    metadata = conditioning[0][1]
    assert "minimax_keyframes" not in metadata
    assert "minimax_refs" not in metadata
    assert "Create a completely new still image" in prompt
    assert "<Picture 1> is a semantic Qwen-only reference" in prompt
    assert "no frame-zero keyframe" in info
    assert "expected route FL2VA" in info


def test_native_generation_switch_uses_ref2va_without_a_keyframe():
    clip, vae, (conditioning, _latent, _prepared_primary, prompt, info) = _encode(
        REFERENCE_NONE,
        primary_image_role=PRIMARY_NATIVE_REFERENCE,
    )

    assert len(vae.encoded) == 1
    assert len(clip.tokenize_calls[0][1]["minimax_ref_items"]) == 1
    metadata = conditioning[0][1]
    assert "minimax_keyframes" not in metadata
    assert len(metadata["minimax_refs"]) == 1
    assert "<Picture 1> is a native Qwen+VAE reference" in prompt
    assert "no frame-zero keyframe" in info
    assert "expected route REF2VA" in info


def test_generation_switch_retains_mixed_ordered_reference_transport():
    native_stack, _info = AddH3EditReference().add(
        image=_image(height=900, width=600),
        transport=REFERENCE_NATIVE,
        semantic_resolution=1024,
        native_reference_size=NATIVE_SIZE_MATCH,
    )
    clip, vae, (conditioning, _latent, _prepared_primary, prompt, info) = _encode(
        REFERENCE_NONE,
        reference_stack=native_stack,
        primary_image_role=PRIMARY_SEMANTIC_REFERENCE,
    )

    assert len(clip.tokenize_calls[0][1]["minimax_ref_items"]) == 2
    assert len(vae.encoded) == 1
    assert "<Picture 1> is a semantic Qwen-only reference" in prompt
    assert "<Picture 2> is a native Qwen+VAE reference" in prompt
    assert "minimax_keyframes" not in conditioning[0][1]
    assert len(conditioning[0][1]["minimax_refs"]) == 1
    assert "Mixed semantic/native REF2VA generation is experimental" in info


def test_four_panel_character_profile_builds_semantic_fl2va_orbit():
    clip, vae, (conditioning, latent, _prepared_primary, prompt, info) = _encode(
        REFERENCE_NONE,
        quality_profile=QUALITY_CHARACTER_FOUR,
        primary_image_role=PRIMARY_SEMANTIC_REFERENCE,
    )

    video, audio = latent["samples"].unbind()
    assert video.shape == (1, 24, 22, 84, 48)
    assert audio.shape == (1, 32, 2, 122)
    assert latent["h3edit_requested_frames"] == 73
    assert latent["h3edit_natural_frames"] == 73
    assert len(vae.encoded) == 0
    assert "minimax_keyframes" not in conditioning[0][1]
    assert "subject_definitions:" in prompt
    assert "summary:\n[reference generation]" in prompt
    assert "retention_analysis:" in prompt
    assert "detailed_description:" in prompt
    assert "180-degree orbit" in prompt
    assert "overall_soundscape:" in prompt
    assert "non_diegetic_music:" in prompt
    assert "Decode H3 Character Sheet" in info
    assert len(clip.tokenize_calls[0][1]["minimax_ref_items"]) == 1


def test_six_panel_character_profile_builds_native_ref2va_orbit():
    _clip, vae, (conditioning, latent, _prepared_primary, prompt, info) = _encode(
        REFERENCE_NONE,
        quality_profile=QUALITY_CHARACTER_SIX,
        primary_image_role=PRIMARY_NATIVE_REFERENCE,
    )

    video, audio = latent["samples"].unbind()
    assert video.shape == (1, 24, 37, 84, 48)
    assert audio.shape == (1, 32, 2, 207)
    assert latent["h3edit_requested_frames"] == 124
    assert latent["h3edit_natural_frames"] == 124
    assert len(vae.encoded) == 1
    assert "minimax_keyframes" not in conditioning[0][1]
    assert len(conditioning[0][1]["minimax_refs"]) == 1
    assert "360-degree orbit" in prompt
    assert "expected route REF2VA" in info


def test_character_profile_rejects_strong_source_anchor():
    with pytest.raises(ValueError, match="cannot be a frame anchor"):
        _encode(
            REFERENCE_NONE,
            quality_profile=QUALITY_CHARACTER_FOUR,
            primary_image_role=PRIMARY_EDIT_ANCHOR,
        )


def test_none_mode_can_leave_reference_socket_connected():
    clip, vae, (conditioning, _latent, _fitted, _prompt, info) = _encode(
        REFERENCE_NONE,
        _image(height=400, width=800),
    )

    assert len(vae.encoded) == 1
    assert len(clip.tokenize_calls[0][1]["minimax_ref_items"]) == 1
    assert "minimax_refs" not in conditioning[0][1]
    assert "intentionally ignored" in info


def test_enabled_direct_mode_without_picture_two_runs_source_only():
    clip, vae, (conditioning, _latent, _fitted, prompt, info) = _encode(REFERENCE_SEMANTIC)

    assert len(vae.encoded) == 1
    assert len(clip.tokenize_calls[0][1]["minimax_ref_items"]) == 1
    assert "<Picture 2>" not in prompt
    assert "minimax_refs" not in conditioning[0][1]
    assert "No guide references" in info


def test_chainable_reference_stack_preserves_picture_order_and_per_ref_transport():
    builder = AddH3EditReference()
    semantic_stack, semantic_info = builder.add(
        image=_image(height=400, width=800),
        transport=REFERENCE_SEMANTIC,
        semantic_resolution=768,
        native_reference_size=NATIVE_SIZE_MATCH,
    )
    mixed_stack, native_info = builder.add(
        image=_image(height=900, width=600),
        transport=REFERENCE_NATIVE,
        semantic_resolution=1024,
        native_reference_size=NATIVE_SIZE_MATCH,
        previous_references=semantic_stack,
    )

    assert len(mixed_stack) == 2
    assert [item["transport"] for item in mixed_stack] == [REFERENCE_SEMANTIC, REFERENCE_NATIVE]
    assert "stack position 1" in semantic_info
    assert "stack position 2" in native_info

    clip, vae, (conditioning, _latent, _fitted, prompt, info) = _encode(
        REFERENCE_NONE,
        reference_stack=mixed_stack,
    )

    assert len(clip.tokenize_calls[0][1]["minimax_ref_items"]) == 3
    assert len(vae.encoded) == 2
    assert "<Picture 2> is a semantic visual guide only" in prompt
    assert "<Picture 3> is a native visual reference" in prompt
    assert len(conditioning[0][1]["minimax_refs"]) == 1
    assert "Semantic=1; native=1" in info


def test_reference_builder_expands_an_image_batch_in_order():
    image_batch = torch.rand((3, 128, 256, 3))
    references, info = AddH3EditReference().add(
        image=image_batch,
        transport=REFERENCE_SEMANTIC,
        semantic_resolution=512,
        native_reference_size=NATIVE_SIZE_MATCH,
    )

    assert len(references) == 3
    assert all(item["image"].shape == (1, 128, 256, 3) for item in references)
    assert "stack positions 1 through 3" in info


def test_experimental_profile_retains_true_one_frame_latent():
    _clip, _vae, (conditioning, latent, _fitted, prompt, info) = _encode(
        REFERENCE_NONE,
        quality_profile=QUALITY_EXPERIMENTAL,
    )

    video, audio = latent["samples"].unbind()
    assert video.shape == (1, 24, 1, 84, 48)
    assert audio.shape == (1, 32, 2, 2)
    assert conditioning[0][1]["minimax_frame_count"] == 1
    assert "Produce exactly one finished still image" in prompt
    assert "natural packet 1 frames" in info


def test_maximum_profile_keeps_twenty_candidates_from_natural_twenty_two():
    _clip, _vae, (conditioning, latent, _fitted, _prompt, info) = _encode(
        REFERENCE_NONE,
        quality_profile=QUALITY_MAXIMUM,
    )

    video, audio = latent["samples"].unbind()
    assert video.shape == (1, 24, 7, 84, 48)
    assert audio.shape == (1, 32, 2, 37)
    assert latent["h3edit_requested_frames"] == 20
    assert latent["h3edit_natural_frames"] == 22
    assert conditioning[0][1]["minimax_frame_count"] == 22
    assert "requested context 20 frames" in info
    assert "natural packet 22 frames" in info


def test_invalid_quality_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown quality_profile"):
        _encode(REFERENCE_NONE, quality_profile="imaginary")


def test_true_one_frame_decode_extracts_video():
    vae = FakeVAE()
    video = torch.zeros((1, 24, 1, 48, 84))
    audio = torch.zeros((1, 32, 2, 2))
    samples = {"samples": FakeNestedTensor((video, audio))}

    image, info = DecodeH3SingleFrame().decode(samples, vae)

    assert vae.decoded == [video]
    assert image.shape == (1, 768, 1344, 3)
    assert "to 1 frame(s)" in info
    assert "stable-quality frame 0" in info


def test_packet_decode_scores_context_and_returns_one_frame():
    vae = FakeVAE()
    video = torch.zeros((1, 24, 2, 48, 84))
    audio = torch.zeros((1, 32, 2, 8))
    samples = {
        "samples": FakeNestedTensor((video, audio)),
        "h3edit_requested_frames": 5,
        "h3edit_natural_frames": 5,
    }

    image, info = DecodeH3SingleFrame().decode(samples, vae)

    assert vae.decoded == [video]
    assert image.shape == (1, 768, 1344, 3)
    selected_value = int(image[0, 0, 0, 0].item())
    assert 0 <= selected_value < 5
    assert "to 5 frame(s)" in info
    assert "scored 5 requested candidate(s)" in info
    assert f"stable-quality frame {selected_value}" in info


def test_character_sheet_decoder_extracts_and_stitches_four_calibrated_views():
    vae = FakeVAE()
    video = torch.zeros((1, 24, 22, 4, 3))
    audio = torch.zeros((1, 32, 2, 122))
    samples = {
        "samples": FakeNestedTensor((video, audio)),
        "h3edit_requested_frames": 73,
        "h3edit_natural_frames": 73,
    }

    sheet, selected, all_frames, info = DecodeH3CharacterSheet().decode(
        samples,
        vae,
        CHARACTER_SHEET_AUTO,
        6,
        "black",
    )

    assert sheet.shape == (1, 134, 102, 3)
    assert selected.shape == (4, 64, 48, 3)
    assert all_frames.shape == (73, 64, 48, 3)
    assert selected[:, 0, 0, 0].tolist() == [2.0, 24.0, 45.0, 68.0]
    assert "[2, 24, 45, 68]" in info
    assert "2x2 sheet" in info


def test_character_sheet_decoder_can_force_six_panel_layout():
    vae = FakeVAE()
    video = torch.zeros((1, 24, 37, 4, 3))
    audio = torch.zeros((1, 32, 2, 207))
    samples = {
        "samples": FakeNestedTensor((video, audio)),
        "h3edit_requested_frames": 124,
        "h3edit_natural_frames": 124,
    }

    sheet, selected, all_frames, info = DecodeH3CharacterSheet().decode(
        samples,
        vae,
        CHARACTER_SHEET_SIX,
        6,
        "neutral gray",
    )

    assert sheet.shape == (1, 134, 156, 3)
    assert selected.shape == (6, 64, 48, 3)
    assert all_frames.shape == (124, 64, 48, 3)
    assert selected[:, 0, 0, 0].tolist() == [2.0, 21.0, 42.0, 63.0, 84.0, 113.0]
    assert "3x2 sheet" in info
