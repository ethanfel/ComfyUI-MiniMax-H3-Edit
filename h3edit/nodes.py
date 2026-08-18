"""Standalone single-image MiniMax H3 edit conditioning and decode nodes."""

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
FPS = 24
AUDIO_LATENT_FPS = 40

QUALITY_RECOMMENDED = "recommended | 5-frame context -> 1 image"
QUALITY_EXTENDED = "extended | 9-frame context -> 1 image"
QUALITY_HIGH = "high | 13-frame context -> 1 image"
QUALITY_MAXIMUM = "maximum | 20-frame context -> 1 image"
QUALITY_EXPERIMENTAL = "experimental | true 1 frame (low quality)"
QUALITY_PROFILES = {
    QUALITY_RECOMMENDED: 5,
    QUALITY_EXTENDED: 9,
    QUALITY_HIGH: 13,
    QUALITY_MAXIMUM: 20,
    QUALITY_EXPERIMENTAL: 1,
}

REFERENCE_NONE = "none (source only)"
REFERENCE_SEMANTIC = "semantic (Qwen only)"
REFERENCE_NATIVE = "native (Qwen + VAE ref)"
REFERENCE_MODES = [REFERENCE_SEMANTIC, REFERENCE_NATIVE, REFERENCE_NONE]
REFERENCE_TRANSPORTS = [REFERENCE_SEMANTIC, REFERENCE_NATIVE]
REFERENCE_STACK_TYPE = "H3EDIT_REFERENCE_STACK"

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


def _build_prompt(prompt: str, prompt_mode: str, reference_modes: list[str], frame_count: int) -> str:
    prompt = (prompt or "").strip()
    if prompt_mode == PROMPT_VERBATIM:
        return prompt
    guide_parts = []
    for ordinal, reference_mode in enumerate(reference_modes, start=2):
        if reference_mode == REFERENCE_SEMANTIC:
            guide_parts.append(
                f" <Picture {ordinal}> is a semantic visual guide only. Transfer only the requested concept, object, "
                "material, or attribute from it; do not copy its scene, identity, pose, or composition."
            )
        elif reference_mode == REFERENCE_NATIVE:
            guide_parts.append(
                f" <Picture {ordinal}> is a native visual reference. Use its detailed appearance only where the "
                "requested edit calls for it."
            )
    guide = "".join(guide_parts)
    still_contract = (
        " Produce exactly one finished still image."
        if frame_count == 1
        else (
            " Apply the edit immediately after the source anchor, then hold the fully completed result unchanged "
            "across the short internal frame packet: locked camera, fixed composition, no subject motion, and no "
            "temporal progression. Every generated frame must read as a crisp finished still image."
        )
    )
    return (
        "Edit <Picture 1>, which is the source image and frame-zero anchor. Preserve the source identity, facial "
        "structure, pose, framing, lighting, background, and all unrequested details."
        f"{guide}{still_contract} Requested edit: {prompt}"
    )


def _decoded_frames_for_latent_t(latent_t: int) -> int:
    latent_t = max(1, int(latent_t))
    if latent_t == 1:
        return 1
    frame_per_token = (1, 4, 4, 4, 4)
    return sum(frame_per_token[index % len(frame_per_token)] for index in range(latent_t))


def _latent_t_for_frame_count(frame_count: int) -> tuple[int, int]:
    requested = max(1, int(frame_count))
    latent_t = 1
    while _decoded_frames_for_latent_t(latent_t) < requested:
        latent_t += 1
    return latent_t, _decoded_frames_for_latent_t(latent_t)


def _empty_h3_edit_latent(width: int, height: int, frame_count: int) -> tuple[dict[str, Any], int]:
    from comfy import model_management, nested_tensor

    requested_frames = max(1, int(frame_count))
    latent_t, natural_frames = _latent_t_for_frame_count(requested_frames)
    audio_t = max(1, round((natural_frames / FPS) * AUDIO_LATENT_FPS))
    device = model_management.intermediate_device()
    video = torch.zeros((1, H3_VIDEO_CHANNELS, latent_t, height // 16, width // 16), device=device)
    audio = torch.zeros(
        (1, H3_AUDIO_CHANNELS, H3_AUDIO_STEREO, audio_t),
        device=device,
    )
    return (
        {
            "samples": nested_tensor.NestedTensor((video, audio)),
            "h3edit_requested_frames": requested_frames,
            "h3edit_natural_frames": natural_frames,
            "h3edit_width": width,
            "h3edit_height": height,
        },
        natural_frames,
    )


def _minmax(values: torch.Tensor) -> torch.Tensor:
    low = values.min()
    high = values.max()
    if float((high - low).abs()) < 1e-8:
        return torch.ones_like(values)
    return (values - low) / (high - low)


def _stable_quality_frame(frames: torch.Tensor, max_side: int = 512) -> tuple[int, float]:
    """Choose one sharp, clean, temporally stable frame using Studio's lightweight metrics."""
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[0] < 1:
        raise ValueError("Stable frame selection expects a non-empty IMAGE batch [T,H,W,C].")
    if frames.shape[0] == 1:
        return 0, 1.0

    samples = frames[..., :3].movedim(-1, 1)
    height, width = samples.shape[-2:]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        target = (max(16, round(height * scale)), max(16, round(width * scale)))
        metric_chunks = []
        for chunk in samples.split(4, dim=0):
            metric_chunks.append(
                F.interpolate(
                    chunk.float(),
                    size=target,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            )
        x = torch.cat(metric_chunks, dim=0).clamp(0.0, 1.0)
    else:
        x = samples.float().clamp(0.0, 1.0)

    gray = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
    lap_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    laplacian = F.conv2d(gray, lap_kernel, padding=1)
    sharpness = torch.log1p(laplacian.var(dim=(1, 2, 3)) * 1000.0)
    contrast = gray.std(dim=(1, 2, 3))
    clipped = ((x < 0.01) | (x > 0.99)).float().mean(dim=(1, 2, 3))
    exposure = (1.0 - clipped * 3.0).clamp(0.0, 1.0)
    quality = 0.70 * _minmax(sharpness) + 0.20 * _minmax(contrast) + 0.10 * exposure

    temporal_delta = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    temporal_delta[0] = (x[0] - x[1]).abs().mean()
    temporal_delta[-1] = (x[-1] - x[-2]).abs().mean()
    if x.shape[0] > 2:
        temporal_delta[1:-1] = 0.5 * (x[1:-1] - x[:-2]).abs().mean(dim=(1, 2, 3))
        temporal_delta[1:-1] += 0.5 * (x[1:-1] - x[2:]).abs().mean(dim=(1, 2, 3))
    stability = 1.0 - _minmax(temporal_delta)
    scores = 0.80 * quality + 0.20 * stability
    selected = int(torch.argmax(scores).item())
    return selected, float(scores[selected].item())


class AddH3EditReference:
    """Build an ordered, chainable list of semantic and native edit references."""

    DESCRIPTION = (
        "Adds one image or IMAGE batch to an ordered H3 Edit reference stack. Chain as many nodes as needed; every "
        "entry independently declares semantic Qwen-only or native Qwen+VAE transport."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Reference image(s) appended in batch order as the next <Picture N> entries."},
                ),
                "transport": (
                    REFERENCE_TRANSPORTS,
                    {
                        "default": REFERENCE_SEMANTIC,
                        "tooltip": "Semantic skips the guide VAE; native also creates a minimax_refs latent.",
                    },
                ),
                "semantic_resolution": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": MAX_SEMANTIC_RESOLUTION,
                        "step": 32,
                        "tooltip": "Equivalent-square Qwen budget used when transport is semantic.",
                    },
                ),
                "native_reference_size": (
                    NATIVE_SIZE_MODES,
                    {
                        "default": NATIVE_SIZE_MATCH,
                        "tooltip": "VAE resize policy used when transport is native.",
                    },
                ),
            },
            "optional": {
                "previous_references": (
                    REFERENCE_STACK_TYPE,
                    {"tooltip": "Connect the previous Add H3 Edit Reference node to keep appending in order."},
                ),
            },
        }

    RETURN_TYPES = (REFERENCE_STACK_TYPE, "STRING")
    RETURN_NAMES = ("references", "info")
    FUNCTION = "add"
    CATEGORY = CATEGORY

    def add(
        self,
        image: torch.Tensor,
        transport: str,
        semantic_resolution: int,
        native_reference_size: str,
        previous_references: tuple[dict[str, Any], ...] | None = None,
    ):
        if transport not in REFERENCE_TRANSPORTS:
            raise ValueError(f"Unknown reference transport: {transport}")
        if native_reference_size not in NATIVE_SIZE_MODES:
            raise ValueError(f"Unknown native_reference_size: {native_reference_size}")
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1:
            raise ValueError("image must be a non-empty ComfyUI IMAGE tensor [B,H,W,C].")
        if image.shape[1] < 1 or image.shape[2] < 1 or image.shape[3] < 3:
            raise ValueError(f"image has invalid image dimensions {tuple(image.shape)}.")

        previous = tuple(previous_references or ())
        if any(not isinstance(item, dict) or "image" not in item or "transport" not in item for item in previous):
            raise ValueError("previous_references is not a valid H3 Edit reference stack.")
        additions = tuple(
            {
                "image": image[index : index + 1, ..., :3],
                "transport": transport,
                "semantic_resolution": int(semantic_resolution),
                "native_reference_size": native_reference_size,
            }
            for index in range(int(image.shape[0]))
        )
        result = previous + additions
        first_position = len(previous) + 1
        last_position = len(result)
        position_text = (
            f"stack position {first_position}"
            if first_position == last_position
            else f"stack positions {first_position} through {last_position}"
        )
        return (
            result,
            f"Added {len(additions)} {transport} reference(s) at {position_text}; stack now contains {len(result)} "
            "guide(s). Text Encode assigns final <Picture N> ordinals after its optional direct reference.",
        )


class TextEncodeH3Edit:
    """Prepare a native source edit plus an optional native or semantic guide."""

    DESCRIPTION = (
        "One-node MiniMax H3 single-image edit conditioning. Picture 1 is always the VAE keyframe source. "
        "The direct Picture 2 guide and an unlimited ordered reference stack can independently use Qwen-only semantic "
        "or native Qwen+VAE transport. Quality modes use a short hidden temporal packet but emit one final image."
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
                "quality_profile": (
                    list(QUALITY_PROFILES),
                    {
                        "default": QUALITY_RECOMMENDED,
                        "tooltip": (
                            "H3 is video-trained. Recommended matches Studio's short 5-frame context, then the decoder "
                            "automatically returns one stable high-quality frame. True 1-frame mode is often poor quality."
                        ),
                    },
                ),
                "reference_stack": (
                    REFERENCE_STACK_TYPE,
                    {
                        "tooltip": (
                            "Optional ordered guides from chained Add H3 Edit Reference nodes. These follow the direct "
                            "reference_image and become the next <Picture N> entries."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("positive", "latent", "fitted_source", "encoded_prompt", "info")
    OUTPUT_TOOLTIPS = (
        "Positive conditioning with the native source keyframe and every ordered guide transport.",
        "An H3 latent with the selected hidden quality context; downstream decode still emits one image.",
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
        quality_profile: str = QUALITY_RECOMMENDED,
        reference_stack: tuple[dict[str, Any], ...] | None = None,
    ):
        import node_helpers

        if reference_mode not in REFERENCE_MODES:
            raise ValueError(f"Unknown reference_mode: {reference_mode}")
        if source_fit not in SOURCE_FIT_MODES:
            raise ValueError(f"Unknown source_fit: {source_fit}")
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode}")
        if quality_profile not in QUALITY_PROFILES:
            raise ValueError(f"Unknown quality_profile: {quality_profile}")

        width = _round_dimension(width)
        height = _round_dimension(height)
        fitted_source = _resize(source_image, width, height, source_fit)
        source_latent = vae.encode(fitted_source)

        visual_items = [{"type": "image", "data": fitted_source}]
        ref_blocks: list[dict[str, Any]] = []
        reference_specs: list[dict[str, Any]] = []
        ignored_direct = reference_mode == REFERENCE_NONE and reference_image is not None
        if reference_image is not None and reference_mode != REFERENCE_NONE:
            reference_specs.append(
                {
                    "image": _validate_image(reference_image, "reference_image"),
                    "transport": reference_mode,
                    "semantic_resolution": int(semantic_resolution),
                    "native_reference_size": native_reference_size,
                }
            )

        if reference_stack is not None:
            if not isinstance(reference_stack, (tuple, list)):
                raise ValueError("reference_stack is not a valid H3 Edit reference stack.")
            for stack_index, item in enumerate(reference_stack):
                if not isinstance(item, dict):
                    raise ValueError(f"reference_stack item {stack_index} is invalid.")
                transport = item.get("transport")
                if transport not in REFERENCE_TRANSPORTS:
                    raise ValueError(f"reference_stack item {stack_index} has unknown transport: {transport}")
                size_mode = item.get("native_reference_size", NATIVE_SIZE_MATCH)
                if size_mode not in NATIVE_SIZE_MODES:
                    raise ValueError(f"reference_stack item {stack_index} has unknown native size mode: {size_mode}")
                reference_specs.append(
                    {
                        "image": _validate_image(item.get("image"), f"reference_stack[{stack_index}].image"),
                        "transport": transport,
                        "semantic_resolution": int(item.get("semantic_resolution", semantic_resolution)),
                        "native_reference_size": size_mode,
                    }
                )

        reference_notes = []
        semantic_count = 0
        native_count = 0
        for ordinal, item in enumerate(reference_specs, start=2):
            reference = item["image"]
            if item["transport"] == REFERENCE_SEMANTIC:
                semantic_width, semantic_height = semantic_target_size(reference, item["semantic_resolution"])
                prepared_reference = _resize_exact(reference, semantic_width, semantic_height)
                visual_items.append({"type": "image", "data": prepared_reference})
                semantic_count += 1
                reference_notes.append(
                    f"Picture {ordinal} semantic Qwen-only {semantic_width}x{semantic_height} (VAE skipped)"
                )
            else:
                native_width, native_height = _native_reference_size(
                    reference,
                    width,
                    height,
                    item["native_reference_size"],
                )
                prepared_reference = _resize_exact(reference, native_width, native_height)
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
                native_count += 1
                reference_notes.append(f"Picture {ordinal} native Qwen+VAE {native_width}x{native_height}")

        if reference_notes:
            reference_note = (
                f"{len(reference_specs)} ordered guide(s): " + "; ".join(reference_notes) + ". "
                f"Semantic={semantic_count}; native={native_count}. Use an FL2VA edit checkpoint."
            )
            if native_count:
                reference_note += (
                    " Native guide minimax_refs mixed with the frame-zero keyframe are experimental and require "
                    "weights that respond to both transports."
                )
        else:
            reference_note = "No guide references; only Picture 1 is presented to Qwen."

        requested_frames = QUALITY_PROFILES[quality_profile]
        latent, natural_frames = _empty_h3_edit_latent(width, height, requested_frames)
        encoded_prompt = _build_prompt(
            prompt,
            prompt_mode,
            [item["transport"] for item in reference_specs],
            requested_frames,
        )
        tokens = clip.tokenize(encoded_prompt, minimax_ref_items=visual_items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning_values: dict[str, Any] = {
            "minimax_keyframes": [{"resolved_frame_index": 0, "latent": source_latent}],
            "minimax_frame_count": natural_frames,
        }
        if ref_blocks:
            conditioning_values["minimax_refs"] = ref_blocks
        conditioning = node_helpers.conditioning_set_values(conditioning, conditioning_values)

        ignored_note = " The direct reference_image is intentionally ignored." if ignored_direct else ""
        info = (
            f"Single-image H3 edit | {quality_profile} | requested context {requested_frames} frames | "
            f"natural packet {natural_frames} frames | "
            f"output {width}x{height} | Picture 1 native frame-zero keyframe "
            f"({tuple(source_latent.shape)}) | {reference_note}{ignored_note} "
            "Decode H3 Edit to One Image scores the decoded context and returns one stable high-quality still."
        )
        return conditioning, latent, fitted_source, encoded_prompt, info


class DecodeH3SingleFrame:
    """Decode an H3 edit packet and return exactly one completed still."""

    DESCRIPTION = (
        "Decodes the complete video stream from an H3 edit latent, scores sharpness, exposure, contrast, and temporal "
        "stability, and returns exactly one image. Experimental true-one-frame input remains supported."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Sampled MiniMax H3 edit latent from Text Encode H3 Edit."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "decode"
    CATEGORY = CATEGORY

    def decode(self, samples: dict[str, Any], vae: Any):
        if not isinstance(samples, dict) or "samples" not in samples:
            raise ValueError("Decode H3 Edit to One Image expects a LATENT dictionary with a samples entry.")
        packed = samples["samples"]
        video = packed.unbind()[0] if getattr(packed, "is_nested", False) else packed
        if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[1] != H3_VIDEO_CHANNELS:
            raise ValueError("Decode H3 Edit to One Image expects H3 video latents shaped [B,24,T,H,W].")
        decoded = vae.decode(video)
        if not isinstance(decoded, torch.Tensor):
            raise ValueError("The MiniMax H3 VAE returned a non-tensor result.")
        if decoded.ndim == 5:
            decoded_frames = int(decoded.shape[1])
            frames = decoded[0]
        elif decoded.ndim == 4:
            expected_frames = max(1, int(samples.get("h3edit_natural_frames", 1)))
            if int(video.shape[0]) == 1 and expected_frames > 1 and int(decoded.shape[0]) >= expected_frames:
                decoded_frames = int(decoded.shape[0])
                frames = decoded
            else:
                decoded_frames = 1
                frames = decoded[:1]
        else:
            raise ValueError(f"Unexpected H3 VAE output shape {tuple(decoded.shape)}.")

        requested_frames = max(1, int(samples.get("h3edit_requested_frames", decoded_frames)))
        candidate_count = min(requested_frames, decoded_frames)
        candidates = frames[:candidate_count]
        frame_index, score = _stable_quality_frame(candidates)
        image = candidates[frame_index : frame_index + 1].clone()
        return (
            image,
            f"Decoded H3 latent {tuple(video.shape)} to {decoded_frames} frame(s); scored {candidate_count} requested "
            f"candidate(s) and returned stable-quality frame {frame_index} (score {score:.4f}) as one image "
            f"{tuple(image.shape)}.",
        )


NODE_CLASS_MAPPINGS = {
    "AddH3EditReference": AddH3EditReference,
    "TextEncodeH3Edit": TextEncodeH3Edit,
    "DecodeH3SingleFrame": DecodeH3SingleFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddH3EditReference": "Add H3 Edit Reference",
    "TextEncodeH3Edit": "Text Encode H3 Edit",
    "DecodeH3SingleFrame": "Decode H3 Edit to One Image",
}
