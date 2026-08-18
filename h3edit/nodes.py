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
QUALITY_CHARACTER_FOUR = "character sheet | 4 panels / 73-frame orbit"
QUALITY_CHARACTER_SIX = "character sheet | 6 panels / 124-frame orbit"
QUALITY_PROFILES = {
    QUALITY_RECOMMENDED: 5,
    QUALITY_EXTENDED: 9,
    QUALITY_HIGH: 13,
    QUALITY_MAXIMUM: 20,
    QUALITY_EXPERIMENTAL: 1,
    QUALITY_CHARACTER_FOUR: 73,
    QUALITY_CHARACTER_SIX: 124,
}
CHARACTER_SHEET_FRAME_INDICES = {
    QUALITY_CHARACTER_FOUR: (2, 24, 45, 68),
    QUALITY_CHARACTER_SIX: (2, 21, 42, 63, 84, 113),
}
CHARACTER_SHEET_AUTO = "auto from encoded profile"
CHARACTER_SHEET_FOUR = "4 panels | 2x2"
CHARACTER_SHEET_SIX = "6 panels | 3x2"
CHARACTER_SHEET_LAYOUTS = [CHARACTER_SHEET_AUTO, CHARACTER_SHEET_FOUR, CHARACTER_SHEET_SIX]
GUTTER_COLORS = ["black", "neutral gray", "white"]

REFERENCE_NONE = "none (source only)"
REFERENCE_SEMANTIC = "semantic (Qwen only)"
REFERENCE_NATIVE = "native (Qwen + VAE ref)"
REFERENCE_MODES = [REFERENCE_SEMANTIC, REFERENCE_NATIVE, REFERENCE_NONE]
REFERENCE_TRANSPORTS = [REFERENCE_SEMANTIC, REFERENCE_NATIVE]
REFERENCE_STACK_TYPE = "H3EDIT_REFERENCE_STACK"

PRIMARY_EDIT_ANCHOR = "edit | strong scene anchor (FL2VA)"
PRIMARY_SEMANTIC_REFERENCE = "generate | semantic Picture 1 (FL2VA)"
PRIMARY_NATIVE_REFERENCE = "generate | native Picture 1 (REF2VA)"
PRIMARY_IMAGE_ROLES = [PRIMARY_EDIT_ANCHOR, PRIMARY_SEMANTIC_REFERENCE, PRIMARY_NATIVE_REFERENCE]

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


def _guide_contract(ordinal: int, reference_mode: str, *, generation: bool) -> str:
    if reference_mode == REFERENCE_SEMANTIC:
        if generation:
            return (
                f" <Picture {ordinal}> is a semantic Qwen-only reference. Use its identity, concept, object, style, "
                "material, or composition only as requested; it is not a fixed source frame."
            )
        return (
            f" <Picture {ordinal}> is a semantic visual guide only. Transfer only the requested concept, object, "
            "material, or attribute from it; do not copy its scene, identity, pose, or composition."
        )
    if reference_mode == REFERENCE_NATIVE:
        if generation:
            return (
                f" <Picture {ordinal}> is a native Qwen+VAE reference. Use its detailed visual appearance for only "
                "the role requested; it is not a frame or composition anchor."
            )
        return (
            f" <Picture {ordinal}> is a native visual reference. Use its detailed appearance only where the "
            "requested edit calls for it."
        )
    return ""


def _build_character_sheet_prompt(
    prompt: str,
    reference_modes: list[str],
    primary_image_role: str,
    frame_count: int,
) -> str:
    primary_transport = (
        REFERENCE_SEMANTIC if primary_image_role == PRIMARY_SEMANTIC_REFERENCE else REFERENCE_NATIVE
    )
    picture_modes = [primary_transport, *reference_modes]
    transport_contract = " ".join(
        _guide_contract(ordinal, reference_mode, generation=True).strip()
        for ordinal, reference_mode in enumerate(picture_modes, start=1)
    )

    if frame_count == 73:
        summary = (
            "[reference generation] The target video presents <Subject 1> as one coherent character in a silent "
            "three-second studio turntable: a front view, left profile, back view, and final front facial close-up."
        )
        retention = (
            "<Subject 1> (appears in [Shot 1] and [Shot 2]): fully_preserved - preserve the requested identity, "
            "facial structure, anatomy, hairstyle, costume construction, materials, colors, accessories, and props "
            "without drift between angles; exclude every source background, source pose, and unrequested person."
        )
        detail = (
            "The target uses the requested visual style with sharp facial and costume detail, a solid uniform light-gray "
            "seamless backdrop, soft even form lighting, no cast or contact shadows, and a long telephoto near-orthographic "
            "view. [Shot 1] <Subject 1> stands centered in a relaxed neutral A-pose, feet shoulder-width apart, arms slightly "
            "clear of the torso, palms toward the thighs, head level, eyes open, and expression neutral. The character is "
            "rigidly motionless like a statue. Hair, cloth, armor, straps, jewelry, loose accessories, and props remain locked "
            "in exactly the same configuration. From 00:00.000 to 00:02.000 the camera performs one smooth constant-speed "
            "180-degree orbit from square front, through the character's left profile, to square back. Framing, scale, "
            "lighting, proportions, surface details, and wardrobe construction remain identical throughout; there is no "
            "breathing, blinking, wind, fabric motion, secondary motion, zoom, roll, shake, or motion blur. [Shot 2] At "
            "00:02.000, the camera rapidly returns to the front and pushes into a locked head-and-shoulders close-up while "
            "<Subject 1> remains unchanged, face square to camera and eyes toward the lens. The close-up settles sharply "
            "before 00:03.000."
        )
    else:
        summary = (
            "[reference generation] The target video presents <Subject 1> as one coherent character in a silent five-second "
            "studio turntable: front, left, back, and right full-body views followed by front and three-quarter facial views."
        )
        retention = (
            "<Subject 1> (appears in [Shot 1], [Shot 2], and [Shot 3]): fully_preserved - preserve the requested identity, "
            "facial structure, anatomy, hairstyle, costume construction, materials, colors, accessories, and props without "
            "drift across the complete orbit and both facial views; exclude every source background, source pose, and "
            "unrequested person."
        )
        detail = (
            "The target uses the requested visual style with sharp facial and costume detail, a solid uniform light-gray "
            "seamless backdrop, soft even form lighting, no cast or contact shadows, and a long telephoto near-orthographic "
            "view. [Shot 1] <Subject 1> stands centered in a relaxed neutral A-pose, feet shoulder-width apart, arms slightly "
            "clear of the torso, palms toward the thighs, head level, eyes open, and expression neutral. The character is "
            "rigidly motionless like a statue. Hair, cloth, armor, straps, jewelry, loose accessories, and props remain locked "
            "in exactly the same configuration. From 00:00.000 to 00:03.000 the camera performs one smooth constant-speed "
            "360-degree orbit, beginning square on the front, passing the left profile at one quarter turn, reaching the back "
            "at halfway, passing the right profile at three quarters, and returning to the front. Framing, scale, lighting, "
            "proportions, surface details, and wardrobe construction remain identical throughout; there is no breathing, "
            "blinking, wind, fabric motion, secondary motion, zoom, roll, shake, or motion blur. [Shot 2] At 00:03.000, the "
            "camera rapidly pushes into a locked head-and-shoulders close-up while <Subject 1> remains unchanged, face square "
            "to camera and eyes toward the lens. The front facial view settles sharply before 00:04.000. [Shot 3] At "
            "00:04.000, the camera rapidly repositions to a clean three-quarter head-and-shoulders angle. <Subject 1> keeps "
            "the same identity, neutral expression, hairstyle, costume neckline, and lighting while the final facial view "
            "settles sharply before 00:05.000."
        )

    return (
        "subject_definitions:\n"
        "<Subject 1> is the single coherent character assembled from the ordered reference pictures according to these "
        f"assignments: {prompt}\n{transport_contract}\n\n"
        f"summary:\n{summary}\n\n"
        f"retention_analysis:\n{retention}\n\n"
        f"detailed_description:\n{detail}\n\n"
        "overall_soundscape:\nSilence; no dialogue, ambience, Foley, room tone, or movement sounds.\n\n"
        "non_diegetic_music:\nN/A"
    )


def _build_prompt(
    prompt: str,
    prompt_mode: str,
    reference_modes: list[str],
    frame_count: int,
    primary_image_role: str,
) -> str:
    prompt = (prompt or "").strip()
    if prompt_mode == PROMPT_VERBATIM:
        return prompt

    if frame_count in {73, 124}:
        return _build_character_sheet_prompt(prompt, reference_modes, primary_image_role, frame_count)

    if primary_image_role == PRIMARY_EDIT_ANCHOR:
        guide = "".join(
            _guide_contract(ordinal, reference_mode, generation=False)
            for ordinal, reference_mode in enumerate(reference_modes, start=2)
        )
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

    primary_transport = (
        REFERENCE_SEMANTIC if primary_image_role == PRIMARY_SEMANTIC_REFERENCE else REFERENCE_NATIVE
    )
    guide = "".join(
        _guide_contract(ordinal, reference_mode, generation=True)
        for ordinal, reference_mode in enumerate([primary_transport, *reference_modes], start=1)
    )
    still_contract = (
        " Produce exactly one finished still image."
        if frame_count == 1
        else (
            " Generate the finished image immediately, then hold it unchanged across the short internal frame packet: "
            "locked camera, fixed composition, no subject motion, and no temporal progression. Every generated frame "
            "must read as the same crisp finished still image."
        )
    )
    return (
        "Create a completely new still image from the requested elements. No input picture is a source frame or scene "
        f"anchor.{guide}{still_contract} Requested image: {prompt}"
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
    """Prepare a source-anchored edit or reference-driven still generation."""

    DESCRIPTION = (
        "Switch Picture 1 between a strong FL2VA source anchor, a semantic FL2VA generation reference, or a native "
        "REF2VA generation reference. Additional ordered guides independently use Qwen-only semantic or native "
        "Qwen+VAE transport. Quality modes use a short hidden temporal packet but emit one final image."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "MiniMax H3 Qwen3-VL text/vision encoder."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE used by anchor and native-reference modes."}),
                "source_image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Picture 1: the photo to edit in anchor mode, or the first semantic/native reference in "
                            "generation mode."
                        )
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "Add the glasses from <Picture 2> to the woman.",
                        "tooltip": "The requested edit or new image. Use explicit <Picture N> roles for references.",
                    },
                ),
                "primary_image_role": (
                    PRIMARY_IMAGE_ROLES,
                    {
                        "default": PRIMARY_EDIT_ANCHOR,
                        "tooltip": (
                            "Edit creates a frame-zero VAE keyframe. Generate removes that keyframe and treats "
                            "Picture 1 as either a semantic FL2VA or native REF2VA reference."
                        ),
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
                            "Equivalent-square Qwen pixel budget for semantic direct or Picture 1 generation refs. "
                            "Aspect ratio is preserved and no VAE latent is allocated."
                        ),
                    },
                ),
                "native_reference_size": (
                    NATIVE_SIZE_MODES,
                    {
                        "default": NATIVE_SIZE_MATCH,
                        "tooltip": "Resize policy for native direct or Picture 1 generation-reference VAE conditioning.",
                    },
                ),
            },
            "optional": {
                "reference_image": (
                    "IMAGE",
                    {"tooltip": "Optional next Picture guide. Choose semantic or native transport above."},
                ),
                "quality_profile": (
                    list(QUALITY_PROFILES),
                    {
                        "default": QUALITY_RECOMMENDED,
                        "tooltip": (
                            "H3 is video-trained. Recommended matches Studio's short 5-frame context, then the decoder "
                            "returns one stable frame. Character-sheet profiles create calibrated 73/124-frame orbits for "
                            "the dedicated sheet decoder. True 1-frame mode is often poor quality."
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
        "Positive conditioning with the selected Picture 1 role and every ordered guide transport.",
        "An H3 latent with the selected still or character-sheet temporal profile.",
        "Picture 1 after preparation for its selected anchor/reference role.",
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
        primary_image_role: str = PRIMARY_EDIT_ANCHOR,
        reference_image: torch.Tensor | None = None,
        quality_profile: str = QUALITY_RECOMMENDED,
        reference_stack: tuple[dict[str, Any], ...] | None = None,
    ):
        import node_helpers

        if reference_mode not in REFERENCE_MODES:
            raise ValueError(f"Unknown reference_mode: {reference_mode}")
        if primary_image_role not in PRIMARY_IMAGE_ROLES:
            raise ValueError(f"Unknown primary_image_role: {primary_image_role}")
        if source_fit not in SOURCE_FIT_MODES:
            raise ValueError(f"Unknown source_fit: {source_fit}")
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode}")
        if quality_profile not in QUALITY_PROFILES:
            raise ValueError(f"Unknown quality_profile: {quality_profile}")
        if quality_profile in CHARACTER_SHEET_FRAME_INDICES and primary_image_role == PRIMARY_EDIT_ANCHOR:
            raise ValueError(
                "Character-sheet profiles require semantic or native generation mode; Picture 1 cannot be a frame anchor."
            )

        width = _round_dimension(width)
        height = _round_dimension(height)
        primary_image = _validate_image(source_image, "source_image")
        anchored_edit = primary_image_role == PRIMARY_EDIT_ANCHOR
        source_latent = None
        visual_items: list[dict[str, Any]] = []
        ref_blocks: list[dict[str, Any]] = []
        reference_specs: list[dict[str, Any]] = []
        if anchored_edit:
            fitted_source = _resize(primary_image, width, height, source_fit)
            source_latent = vae.encode(fitted_source)
            visual_items.append({"type": "image", "data": fitted_source})
            first_reference_ordinal = 2
        else:
            primary_transport = (
                REFERENCE_SEMANTIC
                if primary_image_role == PRIMARY_SEMANTIC_REFERENCE
                else REFERENCE_NATIVE
            )
            reference_specs.append(
                {
                    "image": primary_image,
                    "transport": primary_transport,
                    "semantic_resolution": int(semantic_resolution),
                    "native_reference_size": native_reference_size,
                }
            )
            fitted_source = primary_image
            first_reference_ordinal = 1

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
        for ordinal, item in enumerate(reference_specs, start=first_reference_ordinal):
            reference = item["image"]
            if item["transport"] == REFERENCE_SEMANTIC:
                semantic_width, semantic_height = semantic_target_size(reference, item["semantic_resolution"])
                prepared_reference = _resize_exact(reference, semantic_width, semantic_height)
                visual_items.append({"type": "image", "data": prepared_reference})
                if not anchored_edit and ordinal == 1:
                    fitted_source = prepared_reference
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
                if not anchored_edit and ordinal == 1:
                    fitted_source = prepared_reference
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

        if anchored_edit and not reference_notes:
            reference_note = "No guide references; only Picture 1 is presented to Qwen."
        else:
            reference_kind = "guide" if anchored_edit else "generation reference"
            checkpoint = "FL2VA edit" if anchored_edit else "REF2VA" if native_count else "FL2VA"
            reference_note = (
                f"{len(reference_specs)} ordered {reference_kind}(s): " + "; ".join(reference_notes) + ". "
                f"Semantic={semantic_count}; native={native_count}. Use a {checkpoint} checkpoint."
            )
            if anchored_edit and native_count:
                reference_note += (
                    " Native guide minimax_refs mixed with the frame-zero keyframe are experimental and require "
                    "weights that respond to both transports."
                )
            elif not anchored_edit and semantic_count and native_count:
                reference_note += (
                    " Mixed semantic/native REF2VA generation is experimental because Qwen picture order includes "
                    "semantic entries that have no matching VAE reference block."
                )

        requested_frames = QUALITY_PROFILES[quality_profile]
        latent, natural_frames = _empty_h3_edit_latent(width, height, requested_frames)
        additional_modes = [item["transport"] for item in reference_specs]
        if not anchored_edit:
            additional_modes = additional_modes[1:]
        encoded_prompt = _build_prompt(
            prompt,
            prompt_mode,
            additional_modes,
            requested_frames,
            primary_image_role,
        )
        tokens = clip.tokenize(encoded_prompt, minimax_ref_items=visual_items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning_values: dict[str, Any] = {"minimax_frame_count": natural_frames}
        if anchored_edit:
            conditioning_values["minimax_keyframes"] = [{"resolved_frame_index": 0, "latent": source_latent}]
        if ref_blocks:
            conditioning_values["minimax_refs"] = ref_blocks
        conditioning = node_helpers.conditioning_set_values(conditioning, conditioning_values)

        ignored_note = " The direct reference_image is intentionally ignored." if ignored_direct else ""
        if anchored_edit:
            mode_note = f"Strong-anchor edit | Picture 1 native frame-zero keyframe ({tuple(source_latent.shape)})"
        else:
            route = "REF2VA" if native_count else "FL2VA"
            mode_note = f"Reference-driven generation | no frame-zero keyframe | expected route {route}"
        decoder_note = (
            "Decode H3 Character Sheet extracts the calibrated views and returns a stitched sheet."
            if quality_profile in CHARACTER_SHEET_FRAME_INDICES
            else "Decode H3 Edit to One Image scores the decoded context and returns one stable high-quality still."
        )
        info = (
            f"{mode_note} | {quality_profile} | requested context {requested_frames} frames | "
            f"natural packet {natural_frames} frames | "
            f"output {width}x{height} | {reference_note}{ignored_note} "
            f"{decoder_note}"
        )
        return conditioning, latent, fitted_source, encoded_prompt, info


def _decode_h3_frames(
    samples: dict[str, Any],
    vae: Any,
    caller: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not isinstance(samples, dict) or "samples" not in samples:
        raise ValueError(f"{caller} expects a LATENT dictionary with a samples entry.")
    packed = samples["samples"]
    video = packed.unbind()[0] if getattr(packed, "is_nested", False) else packed
    if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[1] != H3_VIDEO_CHANNELS:
        raise ValueError(f"{caller} expects H3 video latents shaped [B,24,T,H,W].")
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
    return video, frames, decoded_frames


def _stitch_character_panels(
    panels: torch.Tensor,
    columns: int,
    gutter_px: int,
    gutter_value: float,
) -> torch.Tensor:
    if panels.ndim != 4 or int(panels.shape[0]) < 1:
        raise ValueError("Character-sheet panels must be a non-empty IMAGE batch [N,H,W,C].")
    panel_count, height, width, channels = (int(value) for value in panels.shape)
    if panel_count % columns:
        raise ValueError(f"Cannot place {panel_count} panels into a {columns}-column grid.")
    gutter_px = max(0, int(gutter_px))
    row_images = []
    for row_start in range(0, panel_count, columns):
        pieces = []
        for column in range(columns):
            if column and gutter_px:
                pieces.append(panels.new_full((height, gutter_px, channels), gutter_value))
            pieces.append(panels[row_start + column])
        row_images.append(torch.cat(pieces, dim=1))
    if len(row_images) == 1:
        return row_images[0].unsqueeze(0)
    sheet_parts = []
    stitched_width = columns * width + (columns - 1) * gutter_px
    for row_index, row in enumerate(row_images):
        if row_index and gutter_px:
            sheet_parts.append(panels.new_full((gutter_px, stitched_width, channels), gutter_value))
        sheet_parts.append(row)
    return torch.cat(sheet_parts, dim=0).unsqueeze(0)


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
        video, frames, decoded_frames = _decode_h3_frames(samples, vae, "Decode H3 Edit to One Image")

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


class DecodeH3CharacterSheet:
    """Decode an H3 orbit, extract calibrated views, and stitch one character sheet."""

    DESCRIPTION = (
        "Decodes a 73- or 124-frame H3 character orbit, extracts the calibrated four or six views, and stitches a 2x2 "
        "or 3x2 sheet. Also returns the selected views and full decoded frame batch."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Sampled H3 character-sheet latent from Text Encode H3 Edit / Generate."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "layout": (
                    CHARACTER_SHEET_LAYOUTS,
                    {"default": CHARACTER_SHEET_AUTO, "tooltip": "Auto reads the encoded 73/124-frame profile."},
                ),
                "gutter_px": ("INT", {"default": 6, "min": 0, "max": 256, "step": 1}),
                "gutter_color": (GUTTER_COLORS, {"default": "black"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("sheet", "selected_views", "all_frames", "info")
    OUTPUT_TOOLTIPS = (
        "One stitched 2x2 or 3x2 character sheet.",
        "The four or six calibrated views as an IMAGE batch.",
        "Every decoded orbit frame for manual inspection or alternate picks.",
        "Resolved layout, decoded frame count, extraction indices, and output size.",
    )
    FUNCTION = "decode"
    CATEGORY = CATEGORY

    def decode(
        self,
        samples: dict[str, Any],
        vae: Any,
        layout: str,
        gutter_px: int,
        gutter_color: str,
    ):
        if layout not in CHARACTER_SHEET_LAYOUTS:
            raise ValueError(f"Unknown character-sheet layout: {layout}")
        if gutter_color not in GUTTER_COLORS:
            raise ValueError(f"Unknown gutter color: {gutter_color}")

        video, frames, decoded_frames = _decode_h3_frames(samples, vae, "Decode H3 Character Sheet")
        requested_frames = max(1, int(samples.get("h3edit_requested_frames", decoded_frames)))
        if layout == CHARACTER_SHEET_AUTO:
            if requested_frames >= 124:
                profile = QUALITY_CHARACTER_SIX
            elif requested_frames >= 73:
                profile = QUALITY_CHARACTER_FOUR
            else:
                raise ValueError(
                    "Auto character-sheet layout requires a 73- or 124-frame encoded profile; "
                    f"the latent requests {requested_frames} frame(s)."
                )
        else:
            profile = QUALITY_CHARACTER_FOUR if layout == CHARACTER_SHEET_FOUR else QUALITY_CHARACTER_SIX

        indices = CHARACTER_SHEET_FRAME_INDICES[profile]
        if decoded_frames <= max(indices):
            raise ValueError(
                f"{profile} needs decoded frame index {max(indices)}, but the VAE returned only {decoded_frames} frame(s)."
            )
        selected = frames[list(indices)].clone()
        columns = 2 if profile == QUALITY_CHARACTER_FOUR else 3
        gutter_value = {"black": 0.0, "neutral gray": 0.5, "white": 1.0}[gutter_color]
        sheet = _stitch_character_panels(selected, columns, gutter_px, gutter_value)
        return (
            sheet,
            selected,
            frames,
            f"Decoded H3 latent {tuple(video.shape)} to {decoded_frames} frame(s); extracted indices "
            f"{list(indices)} for {profile}; stitched {columns}x{len(indices) // columns} sheet "
            f"{tuple(sheet.shape)} with {gutter_px}px {gutter_color} gutters.",
        )


NODE_CLASS_MAPPINGS = {
    "AddH3EditReference": AddH3EditReference,
    "TextEncodeH3Edit": TextEncodeH3Edit,
    "DecodeH3SingleFrame": DecodeH3SingleFrame,
    "DecodeH3CharacterSheet": DecodeH3CharacterSheet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddH3EditReference": "Add H3 Edit Reference",
    "TextEncodeH3Edit": "Text Encode H3 Edit / Generate",
    "DecodeH3SingleFrame": "Decode H3 Edit to One Image",
    "DecodeH3CharacterSheet": "Decode H3 Character Sheet",
}
