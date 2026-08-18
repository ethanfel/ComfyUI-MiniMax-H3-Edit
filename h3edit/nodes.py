"""Standalone one-frame MiniMax H3 edit conditioning and decode nodes."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

CATEGORY = "MiniMax H3/Edit"
CANVAS_MULTIPLE = 32
MAX_RESOLUTION = 16384
MAX_SEMANTIC_RESOLUTION = 3584
H3_VIDEO_CHANNELS = 24
H3_AUDIO_CHANNELS = 32
H3_AUDIO_STEREO = 2
H3_AUDIO_FRAMES_FOR_ONE_IMAGE = 2

REFERENCE_NONE = "none (source only)"
REFERENCE_SEMANTIC = "semantic (Qwen only)"
REFERENCE_NATIVE = "native (Qwen + VAE ref)"
REFERENCE_MODES = [REFERENCE_SEMANTIC, REFERENCE_NATIVE, REFERENCE_NONE]

PROMPT_EDIT = "edit instruction"
PROMPT_VERBATIM = "use prompt verbatim"
PROMPT_MODES = [PROMPT_EDIT, PROMPT_VERBATIM]

SOURCE_FIT_MODES = ["crop center", "contain / pad", "stretch"]
NATIVE_SIZE_MATCH = "match output area"
NATIVE_SIZE_MAX = "up to 2048px short edge"
NATIVE_SIZE_MODES = [NATIVE_SIZE_MATCH, NATIVE_SIZE_MAX]


def _validate_image(image: Any, name: str) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1:
        raise ValueError(f"{name} must be a non-empty ComfyUI IMAGE tensor [B,H,W,C].")
    if image.shape[1] < 1 or image.shape[2] < 1 or image.shape[3] < 3:
        raise ValueError(f"{name} has invalid image dimensions {tuple(image.shape)}.")
    return image[:1, ..., :3]


def _round_dimension(value: int | float, multiple: int = CANVAS_MULTIPLE) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _fit_grid_area(source_width: int, source_height: int, target_area: int) -> tuple[int, int]:
    """Preserve aspect ratio on Qwen/H3's 32-pixel grid without exceeding an area cap."""
    source_width = max(1, int(source_width))
    source_height = max(1, int(source_height))
    target_area = max(CANVAS_MULTIPLE**2, int(target_area))
    ratio = source_width / source_height
    ideal_width = math.sqrt(target_area * ratio)
    ideal_height = math.sqrt(target_area / ratio)
    center_width = max(1, round(ideal_width / CANVAS_MULTIPLE))
    center_height = max(1, round(ideal_height / CANVAS_MULTIPLE))

    candidates: list[tuple[float, int, int, int]] = []
    for width_cells in range(max(1, center_width - 8), center_width + 9):
        max_height_cells = max(1, target_area // (CANVAS_MULTIPLE**2 * width_cells))
        for height_cells in {
            max(1, center_height - 2),
            max(1, center_height - 1),
            center_height,
            center_height + 1,
            center_height + 2,
            max_height_cells,
        }:
            width = width_cells * CANVAS_MULTIPLE
            height = height_cells * CANVAS_MULTIPLE
            area = width * height
            if area > target_area:
                continue
            aspect_error = abs(math.log((width / height) / ratio))
            area_error = (target_area - area) / target_area
            candidates.append((4.0 * aspect_error + area_error, -area, width, height))

    if not candidates:
        return CANVAS_MULTIPLE, CANVAS_MULTIPLE
    _, _, width, height = min(candidates)
    return width, height


def semantic_target_size(image: torch.Tensor, equivalent_square_resolution: int) -> tuple[int, int]:
    """Return an aspect-preserving Qwen-only size with an equivalent-square pixel budget."""
    source = _validate_image(image, "reference_image")
    resolution = _round_dimension(equivalent_square_resolution)
    resolution = max(256, min(MAX_SEMANTIC_RESOLUTION, resolution))
    return _fit_grid_area(int(source.shape[2]), int(source.shape[1]), resolution * resolution)


def _resize(image: torch.Tensor, width: int, height: int, fit_mode: str) -> torch.Tensor:
    source = _validate_image(image, "image")
    samples = source.movedim(-1, 1)

    if fit_mode == "stretch":
        resized = F.interpolate(samples, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
        return resized.movedim(1, -1).clamp(0.0, 1.0)

    source_height, source_width = int(samples.shape[-2]), int(samples.shape[-1])
    if fit_mode == "crop center":
        scale = max(width / source_width, height / source_height)
    else:
        scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = F.interpolate(
        samples,
        size=(resized_height, resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    if fit_mode == "crop center":
        left = max(0, (resized_width - width) // 2)
        top = max(0, (resized_height - height) // 2)
        resized = resized[..., top : top + height, left : left + width]
    else:
        pad_left = (width - resized_width) // 2
        pad_right = width - resized_width - pad_left
        pad_top = (height - resized_height) // 2
        pad_bottom = height - resized_height - pad_top
        resized = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    return resized.movedim(1, -1).clamp(0.0, 1.0)


def _resize_exact(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    source = _validate_image(image, "reference_image")
    if (int(source.shape[2]), int(source.shape[1])) == (width, height):
        return source
    resized = F.interpolate(
        source.movedim(-1, 1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized.movedim(1, -1).clamp(0.0, 1.0)


def _native_reference_size(
    image: torch.Tensor,
    output_width: int,
    output_height: int,
    size_mode: str,
) -> tuple[int, int]:
    source = _validate_image(image, "reference_image")
    source_height, source_width = int(source.shape[1]), int(source.shape[2])
    if size_mode == NATIVE_SIZE_MAX:
        scale = min(1.0, 2048.0 / min(source_width, source_height))
        target_area = max(
            CANVAS_MULTIPLE**2,
            int(source_width * scale) * int(source_height * scale),
        )
    else:
        scale = min(1.0, math.sqrt((output_width * output_height) / (source_width * source_height)))
        target_area = max(
            CANVAS_MULTIPLE**2,
            int(source_width * scale) * int(source_height * scale),
        )
    return _fit_grid_area(source_width, source_height, target_area)


def _build_prompt(prompt: str, prompt_mode: str, reference_mode: str) -> str:
    prompt = (prompt or "").strip()
    if prompt_mode == PROMPT_VERBATIM:
        return prompt
    guide = ""
    if reference_mode == REFERENCE_SEMANTIC:
        guide = (
            " <Picture 2> is a semantic visual guide only. Transfer only the requested concept, object, material, "
            "or attribute from it; do not copy its scene, identity, pose, or composition."
        )
    elif reference_mode == REFERENCE_NATIVE:
        guide = (
            " <Picture 2> is a native visual reference. Use its detailed appearance only where the requested edit "
            "calls for it."
        )
    return (
        "Edit <Picture 1>, which is the source image and frame-zero anchor. Preserve the source identity, facial "
        "structure, pose, framing, lighting, background, and all unrequested details."
        f"{guide} Produce exactly one finished still image. Requested edit: {prompt}"
    )


def _empty_one_frame_h3_latent(width: int, height: int) -> dict[str, Any]:
    from comfy import model_management, nested_tensor

    device = model_management.intermediate_device()
    video = torch.zeros((1, H3_VIDEO_CHANNELS, 1, height // 16, width // 16), device=device)
    audio = torch.zeros(
        (1, H3_AUDIO_CHANNELS, H3_AUDIO_STEREO, H3_AUDIO_FRAMES_FOR_ONE_IMAGE),
        device=device,
    )
    return {
        "samples": nested_tensor.NestedTensor((video, audio)),
        "h3edit_frame_count": 1,
        "h3edit_width": width,
        "h3edit_height": height,
    }


class TextEncodeH3Edit:
    """Prepare a native source edit plus an optional native or semantic guide."""

    DESCRIPTION = (
        "One-node MiniMax H3 single-frame edit conditioning. Picture 1 is always the VAE keyframe source. "
        "Picture 2 can be disabled, Qwen-only semantic conditioning, or a native Qwen+VAE reference."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "MiniMax H3 Qwen3-VL text/vision encoder."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE used for the source anchor."}),
                "source_image": (
                    "IMAGE",
                    {"tooltip": "Picture 1: the photo to edit. It is always encoded as the frame-zero native anchor."},
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "Add the glasses from <Picture 2> to the woman.",
                        "tooltip": "The requested edit. Picture tags are optional in edit-instruction mode.",
                    },
                ),
                "reference_mode": (
                    REFERENCE_MODES,
                    {
                        "default": REFERENCE_SEMANTIC,
                        "tooltip": (
                            "Semantic sends Picture 2 only through Qwen. Native also VAE-encodes it into minimax_refs. "
                            "None disables Picture 2 even if its socket remains connected."
                        ),
                    },
                ),
                "width": (
                    "INT",
                    {"default": 768, "min": 32, "max": MAX_RESOLUTION, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 1344, "min": 32, "max": MAX_RESOLUTION, "step": 32},
                ),
                "source_fit": (
                    SOURCE_FIT_MODES,
                    {
                        "default": "crop center",
                        "tooltip": "How Picture 1 is fitted to the output canvas before native VAE anchoring.",
                    },
                ),
                "prompt_mode": (
                    PROMPT_MODES,
                    {
                        "default": PROMPT_EDIT,
                        "tooltip": "Edit instruction adds a concise preservation wrapper. Verbatim sends your text unchanged.",
                    },
                ),
                "semantic_resolution": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": MAX_SEMANTIC_RESOLUTION,
                        "step": 32,
                        "tooltip": (
                            "Equivalent-square Qwen pixel budget for a semantic Picture 2. Aspect ratio is preserved. "
                            "It does not allocate a VAE latent."
                        ),
                    },
                ),
                "native_reference_size": (
                    NATIVE_SIZE_MODES,
                    {
                        "default": NATIVE_SIZE_MATCH,
                        "tooltip": "Resize policy used only for native Picture 2 VAE conditioning.",
                    },
                ),
            },
            "optional": {
                "reference_image": (
                    "IMAGE",
                    {"tooltip": "Optional Picture 2 guide. Choose semantic or native transport above."},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("positive", "latent", "fitted_source", "encoded_prompt", "info")
    OUTPUT_TOOLTIPS = (
        "Positive conditioning with the native source keyframe and selected Picture 2 transport.",
        "A true one-frame MiniMax H3 audio/video latent.",
        "Picture 1 after fitting to the output canvas.",
        "The exact prompt sent after the visual token blocks.",
        "Reference transport, Qwen size, VAE usage, and checkpoint guidance.",
    )
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(
        self,
        clip: Any,
        vae: Any,
        source_image: torch.Tensor,
        prompt: str,
        reference_mode: str,
        width: int,
        height: int,
        source_fit: str,
        prompt_mode: str,
        semantic_resolution: int,
        native_reference_size: str,
        reference_image: torch.Tensor | None = None,
    ):
        import node_helpers

        if reference_mode not in REFERENCE_MODES:
            raise ValueError(f"Unknown reference_mode: {reference_mode}")
        if source_fit not in SOURCE_FIT_MODES:
            raise ValueError(f"Unknown source_fit: {source_fit}")
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode}")
        if reference_mode != REFERENCE_NONE and reference_image is None:
            raise ValueError(
                f"reference_mode is '{reference_mode}', but reference_image is not connected. "
                f"Connect Picture 2 or choose '{REFERENCE_NONE}'."
            )

        width = _round_dimension(width)
        height = _round_dimension(height)
        fitted_source = _resize(source_image, width, height, source_fit)
        source_latent = vae.encode(fitted_source)

        visual_items = [{"type": "image", "data": fitted_source}]
        ref_blocks: list[dict[str, Any]] = []
        reference_note = "Picture 2 disabled; only Picture 1 is presented to Qwen."

        if reference_mode == REFERENCE_SEMANTIC:
            semantic_width, semantic_height = semantic_target_size(reference_image, semantic_resolution)
            prepared_reference = _resize_exact(reference_image, semantic_width, semantic_height)
            visual_items.append({"type": "image", "data": prepared_reference})
            reference_note = (
                f"Picture 2 semantic Qwen-only at {semantic_width}x{semantic_height}; guide VAE encode skipped and "
                "no minimax_refs block attached. Use an FL2VA edit checkpoint."
            )
        elif reference_mode == REFERENCE_NATIVE:
            native_width, native_height = _native_reference_size(
                reference_image,
                width,
                height,
                native_reference_size,
            )
            prepared_reference = _resize_exact(reference_image, native_width, native_height)
            reference_latent = vae.encode(prepared_reference)
            visual_items.append({"type": "image", "data": prepared_reference})
            ref_blocks.append(
                {
                    "kind": "image",
                    "latent_h": native_height // 16,
                    "latent_w": native_width // 16,
                    "latent": reference_latent,
                }
            )
            reference_note = (
                f"Picture 2 native Qwen+VAE at {native_width}x{native_height}; one minimax_refs block attached. "
                "This mixed keyframe+REF path is experimental and needs checkpoint weights that respond to both transports."
            )

        encoded_prompt = _build_prompt(prompt, prompt_mode, reference_mode)
        tokens = clip.tokenize(encoded_prompt, minimax_ref_items=visual_items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning_values: dict[str, Any] = {
            "minimax_keyframes": [{"resolved_frame_index": 0, "latent": source_latent}],
            "minimax_frame_count": 1,
        }
        if ref_blocks:
            conditioning_values["minimax_refs"] = ref_blocks
        conditioning = node_helpers.conditioning_set_values(conditioning, conditioning_values)

        latent = _empty_one_frame_h3_latent(width, height)
        ignored_note = (
            " A connected reference_image is intentionally ignored."
            if reference_mode == REFERENCE_NONE and reference_image is not None
            else ""
        )
        info = (
            f"Single-frame H3 edit | output {width}x{height} | Picture 1 native frame-zero keyframe "
            f"({tuple(source_latent.shape)}) | {reference_note}{ignored_note} "
            "The latent contains one video token and decodes to one image on current ComfyUI."
        )
        return conditioning, latent, fitted_source, encoded_prompt, info


class DecodeH3SingleFrame:
    """Decode only a true one-token MiniMax H3 video latent."""

    DESCRIPTION = (
        "Decodes the video stream from a true one-frame H3 nested latent. Current ComfyUI's native MiniMax H3 VAE "
        "supports temporal length 1 directly; no frame duplication or selection is performed."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Sampled one-frame MiniMax H3 nested latent."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "decode"
    CATEGORY = CATEGORY

    def decode(self, samples: dict[str, Any], vae: Any):
        if not isinstance(samples, dict) or "samples" not in samples:
            raise ValueError("Decode H3 Single Frame expects a LATENT dictionary with a samples entry.")
        packed = samples["samples"]
        video = packed.unbind()[0] if getattr(packed, "is_nested", False) else packed
        if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[1] != H3_VIDEO_CHANNELS:
            raise ValueError("Decode H3 Single Frame expects H3 video latents shaped [B,24,T,H,W].")
        if int(video.shape[2]) != 1:
            raise ValueError(
                f"Decode H3 Single Frame received temporal latent length {int(video.shape[2])}; expected exactly 1."
            )

        decoded = vae.decode(video)
        if not isinstance(decoded, torch.Tensor):
            raise ValueError("The MiniMax H3 VAE returned a non-tensor result.")
        if decoded.ndim == 5:
            if int(decoded.shape[1]) != 1:
                raise ValueError(f"The H3 VAE decoded {int(decoded.shape[1])} frames; expected exactly 1.")
            image = decoded[:, 0]
        elif decoded.ndim == 4:
            image = decoded
        else:
            raise ValueError(f"Unexpected H3 VAE output shape {tuple(decoded.shape)}.")
        return image, f"Decoded one native H3 frame from latent shape {tuple(video.shape)} to {tuple(image.shape)}."


NODE_CLASS_MAPPINGS = {
    "TextEncodeH3Edit": TextEncodeH3Edit,
    "DecodeH3SingleFrame": DecodeH3SingleFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeH3Edit": "Text Encode H3 Edit",
    "DecodeH3SingleFrame": "Decode H3 Single Frame",
}
