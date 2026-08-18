from __future__ import annotations

import sys
import types

import pytest
import torch

from h3edit.nodes import (
    NATIVE_SIZE_MATCH,
    PROMPT_EDIT,
    QUALITY_EXPERIMENTAL,
    QUALITY_RECOMMENDED,
    REFERENCE_NATIVE,
    REFERENCE_NONE,
    REFERENCE_SEMANTIC,
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
        reference_image=reference_image,
        quality_profile=quality_profile,
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
    assert metadata["minimax_frame_count"] == 22
    assert len(metadata["minimax_keyframes"]) == 1
    assert "minimax_refs" not in metadata
    video, audio = latent["samples"].unbind()
    assert video.shape == (1, 24, 7, 84, 48)
    assert audio.shape == (1, 32, 2, 37)
    assert latent["h3edit_output_frame_index"] == 21
    assert "guide VAE encode skipped" in info
    assert "internal packet 22 frames" in info


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
    assert "mixed keyframe+REF path is experimental" in info


def test_none_mode_can_leave_reference_socket_connected():
    clip, vae, (conditioning, _latent, _fitted, _prompt, info) = _encode(
        REFERENCE_NONE,
        _image(height=400, width=800),
    )

    assert len(vae.encoded) == 1
    assert len(clip.tokenize_calls[0][1]["minimax_ref_items"]) == 1
    assert "minimax_refs" not in conditioning[0][1]
    assert "intentionally ignored" in info


def test_enabled_reference_requires_picture_two():
    with pytest.raises(ValueError, match="reference_image is not connected"):
        _encode(REFERENCE_SEMANTIC)


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
    assert "internal packet 1 frames" in info


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
    assert "returned completed frame 0" in info


def test_packet_decode_returns_only_the_completed_final_frame():
    vae = FakeVAE()
    video = torch.zeros((1, 24, 7, 48, 84))
    audio = torch.zeros((1, 32, 2, 37))
    samples = {
        "samples": FakeNestedTensor((video, audio)),
        "h3edit_natural_frames": 22,
        "h3edit_output_frame_index": 21,
    }

    image, info = DecodeH3SingleFrame().decode(samples, vae)

    assert vae.decoded == [video]
    assert image.shape == (1, 768, 1344, 3)
    assert torch.all(image == 21)
    assert "to 22 frame(s)" in info
    assert "returned completed frame 21" in info
