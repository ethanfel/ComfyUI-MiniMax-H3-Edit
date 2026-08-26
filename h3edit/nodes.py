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
QUALITY_DIRECTED_CHANGE = "directed change | 39-frame settle -> 1 image"
QUALITY_CHARACTER_FOUR = "character sheet | 4 panels / 73-frame orbit"
QUALITY_CHARACTER_SIX = "character sheet | 6 panels / 124-frame orbit"
QUALITY_SCENE_SHORT = "scene coverage | 124-frame camera path"
QUALITY_SCENE_MEDIUM = "scene coverage | 243-frame camera path"
QUALITY_SCENE_LONG = "scene coverage | 362-frame camera path"
QUALITY_PROFILES = {
    QUALITY_RECOMMENDED: 5,
    QUALITY_EXTENDED: 9,
    QUALITY_HIGH: 13,
    QUALITY_MAXIMUM: 20,
    QUALITY_EXPERIMENTAL: 1,
    QUALITY_DIRECTED_CHANGE: 39,
    QUALITY_CHARACTER_FOUR: 73,
    QUALITY_CHARACTER_SIX: 124,
    QUALITY_SCENE_SHORT: 124,
    QUALITY_SCENE_MEDIUM: 243,
    QUALITY_SCENE_LONG: 362,
}
SCENE_COVERAGE_PROFILES = {
    QUALITY_SCENE_SHORT,
    QUALITY_SCENE_MEDIUM,
    QUALITY_SCENE_LONG,
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
EDIT_OPTIONS_TYPE = "H3EDIT_OPTIONS"

PRIMARY_EDIT_ANCHOR = "edit | strong scene anchor (FL2VA)"
PRIMARY_SEMANTIC_REFERENCE = "generate | semantic Picture 1 (FL2VA)"
PRIMARY_NATIVE_REFERENCE = "generate | native Picture 1 (REF2VA)"
PRIMARY_IMAGE_ROLES = [PRIMARY_EDIT_ANCHOR, PRIMARY_SEMANTIC_REFERENCE, PRIMARY_NATIVE_REFERENCE]

PROMPT_EDIT = "edit instruction"
PROMPT_REPOSE = "directed | re-pose character"
PROMPT_CHARACTER_SWAP = "directed | character swap"
PROMPT_NEW_ANGLE = "directed | new camera angle"
PROMPT_SCENE_COVERAGE = "directed | frozen scene coverage"
PROMPT_SCENE_CUTS = "directed | frozen cinematic cuts"
PROMPT_ROOM_OBJECT_STUDY = "directed | room and object study cuts"
PROMPT_VERBATIM = "use prompt verbatim"
DIRECTED_PROMPT_MODES = [PROMPT_REPOSE, PROMPT_CHARACTER_SWAP, PROMPT_NEW_ANGLE]
SCENE_PROMPT_MODES = [PROMPT_SCENE_COVERAGE, PROMPT_SCENE_CUTS, PROMPT_ROOM_OBJECT_STUDY]
PROMPT_MODES = [PROMPT_EDIT, *DIRECTED_PROMPT_MODES, *SCENE_PROMPT_MODES, PROMPT_VERBATIM]

OPTION_MODE_STILL = "still | edit or generate"
OPTION_MODE_REPOSE = "directed | re-pose character"
OPTION_MODE_CHARACTER_SWAP = "directed | character swap"
OPTION_MODE_NEW_ANGLE = "directed | new camera angle"
OPTION_MODE_CHARACTER_SHEET = "character sheet | canonical 6 views"
OPTION_MODE_SCENE_COVERAGE = "scene coverage | canonical camera path"
OPTION_MODE_SCENE_CUTS = "scene coverage | cinematic hard cuts"
OPTION_MODE_ROOM_OBJECT_STUDY = "scene coverage | room + object study"
OPTION_MODE_VERBATIM = "advanced | prompt verbatim"
OPTION_PROFILE_CANONICAL = "canonical for selected mode"
EDIT_OPTION_PRESETS = {
    OPTION_MODE_STILL: (PROMPT_EDIT, None),
    OPTION_MODE_REPOSE: (PROMPT_REPOSE, QUALITY_DIRECTED_CHANGE),
    OPTION_MODE_CHARACTER_SWAP: (PROMPT_CHARACTER_SWAP, QUALITY_DIRECTED_CHANGE),
    OPTION_MODE_NEW_ANGLE: (PROMPT_NEW_ANGLE, QUALITY_DIRECTED_CHANGE),
    OPTION_MODE_CHARACTER_SHEET: (PROMPT_EDIT, QUALITY_CHARACTER_SIX),
    OPTION_MODE_SCENE_COVERAGE: (PROMPT_SCENE_COVERAGE, QUALITY_SCENE_SHORT),
    OPTION_MODE_SCENE_CUTS: (PROMPT_SCENE_CUTS, QUALITY_SCENE_SHORT),
    OPTION_MODE_ROOM_OBJECT_STUDY: (PROMPT_ROOM_OBJECT_STUDY, QUALITY_SCENE_LONG),
    OPTION_MODE_VERBATIM: (PROMPT_VERBATIM, None),
}
STILL_QUALITY_PROFILES = [
    QUALITY_RECOMMENDED,
    QUALITY_EXTENDED,
    QUALITY_HIGH,
    QUALITY_MAXIMUM,
    QUALITY_EXPERIMENTAL,
]

SCENE_DIRECTION_CLOCKWISE = "clockwise / camera right"
SCENE_DIRECTION_COUNTERCLOCKWISE = "counterclockwise / camera left"
SCENE_DIRECTIONS = [SCENE_DIRECTION_CLOCKWISE, SCENE_DIRECTION_COUNTERCLOCKWISE]

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


def _format_h3_time(frame_index: int) -> str:
    total_ms = round((max(0, int(frame_index)) / FPS) * 1000)
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _scene_capture_plan(
    frame_count: int,
    view_count: int,
    arc_degrees: float,
    hold_frames: int,
) -> dict[str, Any]:
    frame_count = max(5, int(frame_count))
    view_count = max(2, min(24, int(view_count)))
    arc_degrees = max(15.0, min(360.0, float(arc_degrees)))
    full_orbit = math.isclose(arc_degrees, 360.0, abs_tol=0.001)
    denomin = view_count if full_orbit else view_count - 1
    positions = [index / denomin for index in range(view_count)]
    centers = [round((frame_count - 1) * position) for position in positions]
    angles = [arc_degrees * position for position in positions]
    minimum_gap = min(
        (right - left for left, right in zip(centers, centers[1:], strict=False)),
        default=frame_count,
    )
    requested_radius = max(0, (max(1, int(hold_frames)) - 1) // 2)
    radius = min(requested_radius, max(0, (minimum_gap - 1) // 2))
    windows = [
        (max(0, center - radius), min(frame_count - 1, center + radius))
        for center in centers
    ]
    return {
        "frame_count": frame_count,
        "view_count": view_count,
        "arc_degrees": arc_degrees,
        "full_orbit": full_orbit,
        "centers": tuple(centers),
        "angles": tuple(angles),
        "windows": tuple(windows),
        "hold_frames": radius * 2 + 1,
    }


def _room_object_capture_plan(
    frame_count: int,
    view_count: int,
    arc_degrees: float,
    hold_frames: int,
) -> dict[str, Any]:
    """Schedule a short room survey followed by denser views of one object."""
    plan = _scene_capture_plan(frame_count, view_count, arc_degrees, hold_frames)
    view_count = plan["view_count"]
    arc_degrees = plan["arc_degrees"]
    room_view_count = max(1, min(8, round(view_count * 0.375)))
    object_view_count = view_count - room_view_count
    full_orbit = plan["full_orbit"]

    def phase_angles(count: int, *, half_step: bool = False) -> list[float]:
        if count <= 0:
            return []
        if full_orbit:
            offset = 0.5 if half_step else 0.0
            return [round(arc_degrees * ((index + offset) / count), 6) for index in range(count)]
        if count == 1:
            return [arc_degrees / 2 if half_step else 0.0]
        return [round(arc_degrees * index / (count - 1), 6) for index in range(count)]

    plan["room_view_count"] = room_view_count
    plan["object_view_count"] = object_view_count
    plan["angles"] = tuple(
        phase_angles(room_view_count) + phase_angles(object_view_count, half_step=True)
    )
    return plan


def _scene_reference_contract(
    reference_modes: list[str],
    first_ordinal: int,
) -> str:
    if not reference_modes:
        return (
            "No alternate-angle reference is supplied. Infer only genuinely occluded surfaces as conservative, "
            "geometrically coherent continuations of the visible room; never redesign, replace, move, or duplicate "
            "anything already established by <Picture 1>."
        )
    records = []
    for ordinal, reference_mode in enumerate(reference_modes, start=first_ordinal):
        transport = (
            "semantic Qwen-only alternate view"
            if reference_mode == REFERENCE_SEMANTIC
            else "native Qwen+VAE alternate view"
        )
        records.append(
            f"<Picture {ordinal}> is a {transport} of the exact same physical scene, not a style, identity, or "
            "composition donor. Use its visible walls, openings, fixtures, objects, placements, materials, and lighting "
            "to constrain one shared world coordinate system; it is not a timeline keyframe."
        )
    return " ".join(records) + (
        " Where alternate views overlap <Picture 1>, <Picture 1> has priority; use the alternate views primarily to "
        "resolve geometry and appearance that the source camera cannot see."
    )


def _scene_generation_contract(reference_modes: list[str]) -> str:
    records = []
    for ordinal, reference_mode in enumerate(reference_modes, start=1):
        transport = (
            "semantic Qwen-only design reference"
            if reference_mode == REFERENCE_SEMANTIC
            else "native Qwen+VAE design reference"
        )
        records.append(
            f"<Picture {ordinal}> is a {transport} for <Subject 1>. Use only the room geometry, architecture, furniture, "
            "materials, palette, lighting, or camera relationships explicitly assigned by the instruction; it is not a "
            "source frame or timeline keyframe."
        )
    return " ".join(records)


def _room_object_generation_contract(reference_modes: list[str]) -> str:
    records = []
    for ordinal, reference_mode in enumerate(reference_modes, start=1):
        transport = (
            "semantic Qwen-only survey reference"
            if reference_mode == REFERENCE_SEMANTIC
            else "native Qwen+VAE survey reference"
        )
        records.append(
            f"<Picture {ordinal}> is a {transport} showing an observation of the same physical room. Where the target "
            "object is visible, use it as evidence for that same object's appearance and fixed placement. "
            "Use overlapping architecture, openings, fixtures, furniture, object placement, proportions, materials, and "
            "lighting to solve one shared world coordinate system; it is not a source frame, composition anchor, or "
            "timeline keyframe."
        )
    return " ".join(records) + (
        " Treat the ordered pictures as complementary survey evidence with equal semantic authority, not separate room "
        "designs. No picture is the master, source, pixel anchor, composition anchor, or timeline frame, and Picture 1 "
        "has no priority merely because it is listed first. Reconcile repeated evidence "
        "across all views; when observations conflict, prefer cross-view consensus and use the clearest local view only "
        "for the detail it actually reveals. Regenerate the room "
        "and object cleanly while preserving their recognizable identity and ordinary physical character; discard source "
        "blur, noise, compression, clipped highlights, color casts, lens distortion, and accidental cropping."
    )


def _build_scene_coverage_prompt(
    prompt: str,
    reference_modes: list[str],
    frame_count: int,
    direction: str,
    capture_plan: dict[str, Any],
    loop_closure: bool,
    anchored_scene: bool,
) -> str:
    duration = frame_count / FPS
    direction_word = "clockwise" if direction == SCENE_DIRECTION_CLOCKWISE else "counterclockwise"
    origin_view = "source view" if anchored_scene else "generated opening view"
    waypoint_records = []
    for view_number, (angle, center, window) in enumerate(
        zip(capture_plan["angles"], capture_plan["centers"], capture_plan["windows"], strict=True),
        start=1,
    ):
        start, end = window
        waypoint_records.append(
            f"capture {view_number} at {_format_h3_time(center)} ({angle:g} degrees {direction_word} from the "
            f"{origin_view}), holding completely static from {_format_h3_time(start)} through {_format_h3_time(end)}"
        )
    waypoints = "; ".join(waypoint_records) + "."
    motion_contract = (
        f"The camera follows one continuous {capture_plan['arc_degrees']:g}-degree {direction_word} arc at the initial "
        "camera height and constant radius, always aimed at the declared orbit center. Preserve one fixed focal length, "
        "field of view, exposure, white balance, focus behavior, horizon, and camera roll. Perspective, occlusion, and "
        "parallax change only as physically required by camera translation; view-dependent reflections may respond "
        "naturally, but lighting and material identity do not change. Do not pan independently, zoom, dolly inward or "
        "outward, pedestal, roll, shake, cut, relight, animate, morph geometry, bend walls, slide textures, duplicate or "
        "remove objects, add doors or windows, or turn the room into a different place. Newly revealed surfaces must be "
        "conservative, spatially consistent continuations constrained by all available views. The camera moves smoothly "
        "between exact capture waypoints, decelerates into each hold, becomes perfectly motionless throughout that hold, "
        "and resumes smoothly afterward. Every hold produces crisp matching still frames with no motion blur. Exact "
        f"waypoint schedule: {waypoints}"
    )

    if not anchored_scene:
        picture_contract = _scene_generation_contract(reference_modes)
        return (
            "subject_definitions:\n"
            "<Subject 1> is one completely new, coherent room generated from the ordered reference pictures according "
            f"to this design and orbit-center assignment: {prompt or 'Create a complete room and orbit its geometric center.'}\n"
            f"{picture_contract}\n\n"
            "summary:\n"
            f"[reference generation] The target creates <Subject 1> as one new frozen three-dimensional room, then records "
            f"{capture_plan['view_count']} consistent camera views across one {capture_plan['arc_degrees']:g}-degree "
            f"{direction_word} arc without treating any input picture as a source frame.\n\n"
            "retention_analysis:\n"
            "<Subject 1> (appears throughout [Shot 1]): fully_preserved - after the new room is established, preserve its "
            "complete architecture, dimensions, wall openings, fixtures, furniture, objects, materials, colors, lighting, "
            "and spatial relationships without drift at every camera position.\n\n"
            "detailed_description:\n"
            "The target uses the requested source-matched or photorealistic visual style with crisp architectural detail "
            "and stable world-space lighting. [Shot 1] First synthesize <Subject 1> as one complete physical room from the "
            "assigned aspects of the reference pictures. No input picture is a source frame, composition anchor, or pixel "
            "anchor. Resolve the references into one new layout rather than copying any source composition wholesale. Once "
            "established, treat the complete room as a rigid, frozen three-dimensional set in one fixed world coordinate "
            "system. Every person remains frozen in the identical pose and expression; every movable object, fabric fold, "
            "prop, door, window, fixture, reflection source, shadow caster, and light source remains fixed. Only the physical "
            f"camera moves. {motion_contract}\n\n"
            "overall_soundscape:\nN/A\n\n"
            "non_diegetic_music:\nN/A"
        )

    first_reference_ordinal = 3 if loop_closure else 2
    references = _scene_reference_contract(reference_modes, first_reference_ordinal)
    if loop_closure:
        alignment = (
            "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
            f"0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the {duration:.2f}-second "
            "mark of the target video."
        )
        closure = (
            "<Picture 2> is an internal duplicate of <Picture 1> and fixes the final frame to the exact original "
            "view for 360-degree loop closure. After the last unique capture, return precisely to <Picture 2>."
        )
    else:
        alignment = (
            "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
        )
        closure = "The final viewpoint remains free and is determined by the requested camera arc."

    return (
        f"{alignment}\n\n"
        "integrated_multimodal_description: [Shot 1] Live-action or source-matched imagery. <Picture 1> establishes "
        "the exact opening pixels, composition, people, objects, environment, geometry, materials, colors, exposure, "
        "shadows, and lighting of one physical scene. Treat the complete scene as a rigid, frozen three-dimensional "
        "set in one fixed world coordinate system. Every person remains frozen in the identical pose and expression; "
        "every movable object, fabric fold, prop, door, window, fixture, reflection source, shadow caster, and light "
        "source remains fixed. Only the physical camera moves. "
        f"Scene and orbit-center instruction: {prompt or 'Orbit around the geometric center of the room.'} "
        f"{references} {closure} {motion_contract}\n\n"
        "overall_soundscape: N/A\n\n"
        "non_diegetic_music: N/A"
    )


def _build_scene_cut_prompt(
    prompt: str,
    reference_modes: list[str],
    frame_count: int,
    direction: str,
    capture_plan: dict[str, Any],
    anchored_scene: bool,
    object_study: bool = False,
) -> str:
    """Build discrete, static cinematic shots around a user-designated scene target."""
    duration = frame_count / FPS
    direction_word = "clockwise" if direction == SCENE_DIRECTION_CLOCKWISE else "counterclockwise"
    origin_view = "source viewpoint" if anchored_scene else "generated opening viewpoint"
    target_assignment = prompt or (
        "Target object: the primary freestanding object near the visual center of the room."
        if object_study
        else "Coverage target: the primary person or object at the visual center of the scene."
    )
    if object_study:
        room_setups = (
            "a wide room-establishing composition at eye level with a 28 mm lens",
            "a reverse wide room-establishing composition at eye level with a 28 mm lens",
            "a slightly high-angle room overview with a 32 mm lens",
            "a low eye-level room composition from a distinct corner or opening with a 35 mm lens",
            "a balanced wide composition that resolves the target object's position against the room with a 35 mm lens",
            "a lateral medium-wide composition that resolves the target, rug, furniture, and opposite wall with a 40 mm lens",
        )
        object_setups = (
            "a contextual object-height three-quarter composition with a 40 mm lens",
            "an opposite contextual three-quarter composition at object height with a 50 mm lens",
            "a contextual side composition preserving the rug and far wall with a 50 mm lens",
            "a low environmental object-corner composition with a 35 mm lens",
            "a low contextual profile composition preserving the floor and doorway with a 50 mm lens",
            "an eye-level construction-aware medium composition with a 65 mm lens",
            "a slightly elevated environmental three-quarter composition with a 50 mm lens",
            "a high-angle top-surface and rug-footprint composition with a 50 mm lens",
            "an opposite high three-quarter environmental composition with a 50 mm lens",
            "a contextual seam and material study with a 65 mm lens while room landmarks remain visible",
            "a low contextual contact-point study with a 50 mm lens while floorboards and furniture remain visible",
            "a balanced room-and-object hero composition at eye level with a 40 mm lens",
            "an opposite-side contextual object study with a 65 mm lens",
            "a floor-level environmental silhouette composition with a 40 mm lens",
            "an elevated object-geometry composition that retains surrounding architecture with a 50 mm lens",
            "a contextual three-quarter composition emphasizing object depth and room parallax with a 65 mm lens",
            "an environmental medium composition revealing the nearest wall relationship with a 50 mm lens",
            "an environmental medium composition revealing the opposite wall relationship with a 50 mm lens",
            "a construction study that preserves the rug, floor, and one complete architectural opening with a 65 mm lens",
            "a final balanced room-and-object composition at eye level with a 40 mm lens",
        )
    else:
        setups = (
            "a wide establishing composition at eye level with a 32 mm lens",
            "a medium-wide three-quarter composition at eye level with a 40 mm lens",
            "a medium profile composition at eye level with a 65 mm lens",
            "a low-angle three-quarter composition with a 35 mm lens",
            "a reverse wide composition at eye level with a 32 mm lens",
            "a slightly high-angle three-quarter composition with a 50 mm lens",
            "a tight profile or detail composition with an 85 mm lens",
            "a medium three-quarter hero composition at eye level with a 50 mm lens",
        )
    centers = capture_plan["centers"]
    cut_frames = [0]
    cut_frames.extend(round((left + right) / 2) for left, right in zip(centers, centers[1:], strict=False))
    shot_records = []
    for index, (angle, center, window) in enumerate(
        zip(capture_plan["angles"], centers, capture_plan["windows"], strict=True)
    ):
        shot_number = index + 1
        start, end = window
        if object_study:
            if index < capture_plan["room_view_count"]:
                setup = room_setups[index % len(room_setups)]
            else:
                object_index = index - capture_plan["room_view_count"]
                setup = object_setups[object_index % len(object_setups)]
        else:
            setup = setups[index % len(setups)]
        if index == 0:
            shot_open = "[Shot 1] Live-action or source-matched imagery. "
            if anchored_scene:
                setup = (
                    "The shot begins from the exact pixels, camera placement, lens, framing, and composition of "
                    "<Picture 1>"
                )
            else:
                setup = f"The generated scene opens in {setup}"
        else:
            cut_time = _format_h3_time(cut_frames[index])
            shot_open = f"[Shot {shot_number}] At {cut_time}, the shot cuts instantly to "
        camera_position = (
            f"a discrete camera placement {angle:g} degrees {direction_word} from the {origin_view}, with the optical "
            "axis aimed precisely at the designated coverage target"
        )
        phase = ""
        if object_study:
            if index < capture_plan["room_view_count"]:
                phase = (
                    " This room-survey shot keeps the target object clearly visible in its exact architectural and "
                    "furniture context while prioritizing room geometry and spatial relationships."
                )
            else:
                phase = (
                    " This contextual object-study shot keeps the exact target at roughly 25 to 45 percent of the frame, "
                    "with its complete silhouette, the surrounding rug or floor, and at least two recognizable room "
                    "anchors visible. Resolve its dimensions, construction, material, seams or joinery, surface wear, "
                    "and contact with the floor without moving, rotating, or redesigning it."
                )
        capture = (
            f"{phase} The camera is completely static for the entire shot; extraction capture {shot_number} is centered at "
            f"{_format_h3_time(center)} and its guaranteed motionless window runs from {_format_h3_time(start)} through "
            f"{_format_h3_time(end)}."
        )
        if index == 0:
            shot_records.append(f"{shot_open}{setup}, using {camera_position}. {capture}")
        else:
            shot_records.append(f"{shot_open}{setup}, already settled at {camera_position}. {capture}")

    target_kind = "one specific visible physical object" if object_study else (
        "one specific visible person, object, architectural feature, or declared point in world space"
    )
    study_contract = (
        f"The first {capture_plan['room_view_count']} shots establish the complete room from separated viewpoints; the "
        f"remaining {capture_plan['object_view_count']} shots form a dense multi-height, multi-distance study of the same "
        "target object while retaining clear room context. Keep that object visible in every room survey. In every object "
        "study, the target occupies roughly 25 to 45 percent of the frame; show its complete silhouette, its rug or floor "
        "contact, and at least two stable architectural or furniture anchors. Never use an isolated product shot, extreme "
        "close-up, macro-only crop, blank background, or composition that removes the room. The target is bolted to one "
        "world-space orientation: the same faces remain aimed toward the same doors, windows, walls, floorboards, and rug "
        "fibers in every shot. The camera cuts around the stationary target; the target never spins, yaws, turns, pivots, "
        "or behaves like a turntable product. Background parallax and changing occlusion must prove that the camera changed "
        "position. Never replace the target with a similar object, alter its dimensions, move it, clean it up independently, "
        "change its upholstery or construction, change seams or joinery, or detach it from its fixed floor position. "
        if object_study else ""
    )
    cut_contract = (
        f"Coverage-target assignment: {target_assignment} Resolve that wording to {target_kind} and keep the exact same "
        "target point for every shot. "
        f"{study_contract}"
        f"Across the {duration:.2f}-second timeline, the {capture_plan['view_count']} viewpoints are separate editorial "
        "shots distributed across a "
        f"{capture_plan['arc_degrees']:g}-degree {direction_word} coverage arc. Their angular coordinates specify only "
        "discrete camera placements; they never describe camera travel. Every cut is a true instantaneous hard cut: the "
        "first frame after the cut is already the fully resolved new perspective. Never render an orbit, pan, truck, "
        "dolly, zoom, whip-pan, blur, dissolve, morph, or intermediate camera position between shots. Across every cut, "
        "preserve one rigid frozen world coordinate system. Every person keeps the identical pose, expression, gaze, "
        "hand position, hair strand, and garment fold; every object, wall, opening, fixture, material, light source, "
        "shadow caster, and reflection source remains fixed. Only camera position, camera height, focal length, and "
        "framing may change as explicitly declared by each shot. Perspective, parallax, occlusion, and view-dependent "
        "reflections change only as physically required by that new camera. Newly revealed surfaces are conservative, "
        "geometrically coherent continuations constrained by every available view. "
    )
    shots = " ".join(shot_records)
    first_shot_marker = "[Shot 1] Live-action or source-matched imagery. "

    if not anchored_scene:
        if object_study:
            picture_contract = _room_object_generation_contract(reference_modes)
            subject_definitions = (
                "<Subject 1> is one completely new, coherent photorealistic reconstruction of the same physical room "
                "observed across the ordered reference pictures.\n"
                f"<Subject 2> is the one exact target object inside <Subject 1> designated by this assignment: "
                f"{target_assignment}\n"
            )
            summary = (
                f"[reference generation] The target reconstructs <Subject 1> and <Subject 2>, freezes their shared world "
                f"completely, then records {capture_plan['room_view_count']} room-establishing shots and "
                f"{capture_plan['object_view_count']} contextual object-study shots separated only by exact hard cuts."
            )
            retention = (
                "<Subject 1> (appears in every shot): fully_preserved - preserve the regenerated architecture, dimensions, "
                "openings, fixtures, furniture, materials, lighting, and spatial relationships without drift across cuts.\n"
                "<Subject 2> (appears in every shot): fully_preserved - preserve the target object's exact identity, "
                "dimensions, silhouette, construction, material, seams or joinery, surface character, orientation, and "
                "fixed position while only viewpoint, lens, height, and framing change."
            )
            detail_open = (
                "The target uses high-fidelity live-action architectural photography with crisp natural detail, accurate "
                "perspective, physically plausible materials, neutral color, controlled highlights, realistic global "
                "illumination, clean shadow detail, and stable world-space lighting. First reconcile every ordered picture "
                "as survey evidence of one source room, then regenerate <Subject 1> and <Subject 2> as one improved but "
                "recognizably identical physical place. No input picture supplies source pixels or a timeline frame. "
            )
        else:
            picture_contract = _scene_generation_contract(reference_modes)
            subject_definitions = (
                "<Subject 1> is one completely new, coherent physical scene generated from the ordered reference pictures. "
                f"Its designated cinematic coverage target and design assignment are: {target_assignment}\n"
            )
            summary = (
                f"[reference generation] The target creates <Subject 1>, freezes it completely, and records "
                f"{capture_plan['view_count']} exact cinematic viewpoints as static shots separated only by hard cuts."
            )
            retention = (
                "<Subject 1> (appears in every shot): fully_preserved - preserve the generated scene, designated target, "
                "architecture, people, objects, materials, lighting, and world-space relationships without drift while the "
                "camera cuts among discrete viewpoints."
            )
            detail_open = (
                "The target uses the requested source-matched or photorealistic visual style with crisp detail and stable "
                "world-space lighting. First resolve all assigned reference aspects into one new scene; no input picture is "
                "a source frame, composition anchor, or timeline keyframe. "
            )
        cinematic_shots = shots.replace(first_shot_marker, f"{first_shot_marker}{cut_contract}", 1)
        return (
            "subject_definitions:\n"
            f"{subject_definitions}"
            f"{picture_contract}\n\n"
            "summary:\n"
            f"{summary}\n\n"
            "retention_analysis:\n"
            f"{retention}\n\n"
            "detailed_description:\n"
            f"{detail_open}"
            f"{cinematic_shots}\n\n"
            "overall_soundscape:\nN/A\n\n"
            "non_diegetic_music:\nN/A"
        )

    references = _scene_reference_contract(reference_modes, 2)
    anchored_context = (
        f"{cut_contract}<Picture 1> establishes the exact scene, target identity, people, objects, environment, geometry, "
        f"materials, colors, exposure, shadows, and lighting. {references} "
    )
    cinematic_shots = shots.replace(first_shot_marker, f"{first_shot_marker}{anchored_context}", 1)
    return (
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
        f"integrated_multimodal_description: {cinematic_shots}\n\n"
        "overall_soundscape: N/A\n\n"
        "non_diegetic_music: N/A"
    )


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


def _build_directed_change_prompt(
    prompt: str,
    prompt_mode: str,
    reference_modes: list[str],
    frame_count: int,
) -> str:
    duration = frame_count / FPS
    settle_time = duration * 0.65
    guide_notes = " ".join(
        (
            f"<Picture {ordinal}> is a semantic Qwen-only guide with no timeline alignment or VAE pixel anchor; "
            "use only the role explicitly assigned in the requested transformation."
            if reference_mode == REFERENCE_SEMANTIC
            else f"<Picture {ordinal}> is a native Qwen+VAE guide with no timeline alignment; use its detailed "
            "appearance only for the role explicitly assigned in the requested transformation."
        )
        for ordinal, reference_mode in enumerate(reference_modes, start=2)
    )

    if prompt_mode == PROMPT_REPOSE:
        action = (
            "The camera remains completely static. Immediately after the first frame, the person smoothly changes only "
            "their body pose toward the requested target pose. Preserve identity, facial structure, hairstyle, physique, "
            "clothing design, fabric construction, colors, accessories, environment, lighting, framing, lens, and camera "
            "position. Transfer limb placement, torso orientation, head direction, hand positions, weight distribution, "
            "and expression only when the request assigns them. Do not inherit a pose guide's identity, clothing, scene, "
            "or lighting. Maintain natural joint motion, correct anatomy, intact hands, and stable garment topology."
        )
        result = "the character is fully settled into the requested new pose"
    elif prompt_mode == PROMPT_CHARACTER_SWAP:
        action = (
            "The camera remains completely static. Immediately after the first frame, replace only the source character "
            "with the requested reference character. Preserve the source scene geometry, subject placement, pose, action, "
            "framing, camera, perspective, lighting, shadows, contact points, and every unaffected object. Transfer only "
            "the identity, face, skin, hair, physique, wardrobe, or accessories explicitly assigned by the request. Do not "
            "copy a donor background, camera angle, pose, lighting, or unrelated person. The replacement must integrate "
            "with correct anatomy, occlusion, perspective, environmental light, cast shadows, and contact shadows."
        )
        result = "the replacement character is fully integrated and every source-scene detail is stable"
    else:
        action = (
            "The subject and world remain rigidly frozen while only the camera moves in one smooth controlled arc from the "
            "source viewpoint to the requested new viewpoint. Preserve character identity, anatomy, pose, expression, "
            "wardrobe construction, props, environment geometry, object placement, materials, colors, and lighting. The "
            "camera movement must follow the requested azimuth, elevation, distance, framing, and lens behavior without "
            "zoom drift, roll, shake, subject motion, or scene morphing. Reconstruct newly visible surfaces consistently "
            "from the references and do not invent duplicate objects or accessories."
        )
        result = "the camera is locked at the requested new angle and the reconstructed view is fully stable"

    return (
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
        "integrated_multimodal_description: [Shot 1] The target begins from the exact composition, pixels, subjects, "
        "environment, and lighting established by <Picture 1>. "
        f"{guide_notes} Requested transformation: {prompt} {action} Complete the controlled change by "
        f"00:{settle_time:06.3f}; from that point through 00:{duration:06.3f}, {result}. Hold the completed result "
        "perfectly unchanged across every remaining frame so the tail is a sequence of crisp matching stills.\n\n"
        "overall_soundscape: N/A\n\n"
        "non_diegetic_music: N/A"
    )


def _build_prompt(
    prompt: str,
    prompt_mode: str,
    reference_modes: list[str],
    frame_count: int,
    primary_image_role: str,
    *,
    scene_capture_plan: dict[str, Any] | None = None,
    scene_direction: str = SCENE_DIRECTION_CLOCKWISE,
    scene_loop_closure: bool = False,
) -> str:
    prompt = (prompt or "").strip()
    if prompt_mode == PROMPT_VERBATIM:
        return prompt

    if prompt_mode in SCENE_PROMPT_MODES:
        if scene_capture_plan is None:
            raise ValueError("Frozen scene coverage requires a camera capture plan.")
        anchored_scene = primary_image_role == PRIMARY_EDIT_ANCHOR
        if anchored_scene:
            picture_modes = reference_modes
        else:
            primary_transport = (
                REFERENCE_SEMANTIC if primary_image_role == PRIMARY_SEMANTIC_REFERENCE else REFERENCE_NATIVE
            )
            picture_modes = [primary_transport, *reference_modes]
        if prompt_mode in {PROMPT_SCENE_CUTS, PROMPT_ROOM_OBJECT_STUDY}:
            return _build_scene_cut_prompt(
                prompt,
                picture_modes,
                frame_count,
                scene_direction,
                scene_capture_plan,
                anchored_scene,
                object_study=prompt_mode == PROMPT_ROOM_OBJECT_STUDY,
            )
        return _build_scene_coverage_prompt(
            prompt,
            picture_modes,
            frame_count,
            scene_direction,
            scene_capture_plan,
            scene_loop_closure and anchored_scene,
            anchored_scene,
        )

    if frame_count in {73, 124}:
        return _build_character_sheet_prompt(prompt, reference_modes, primary_image_role, frame_count)

    if prompt_mode in DIRECTED_PROMPT_MODES:
        return _build_directed_change_prompt(prompt, prompt_mode, reference_modes, frame_count)

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


class H3EditOptions:
    """Keep task-specific controls off the main H3 Edit encoder."""

    DESCRIPTION = (
        "One mode selects a compatible H3 prompt compiler and frame profile together. Connect this node to Text Encode "
        "H3 Edit / Generate to hide the encoder's legacy advanced widgets. Leave Show Overrides off to send the full "
        "canonical preset; coverage controls appear only when overrides are enabled for scene coverage."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    list(EDIT_OPTION_PRESETS),
                    {
                        "default": OPTION_MODE_STILL,
                        "tooltip": (
                            "The complete task preset. Scene coverage, directed edits, and character sheets each "
                            "select their required frame profile automatically."
                        ),
                    },
                ),
                "show_overrides": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Reveal expert overrides. Leave off to use the complete canonical preset.",
                    },
                ),
                "profile_override": (
                    [OPTION_PROFILE_CANONICAL, *QUALITY_PROFILES],
                    {
                        "default": OPTION_PROFILE_CANONICAL,
                        "tooltip": "Override the canonical frame profile only when a task needs deliberate tuning.",
                    },
                ),
                "direct_reference_transport": (
                    REFERENCE_MODES,
                    {
                        "default": REFERENCE_SEMANTIC,
                        "tooltip": "Transport for the optional direct reference_image on the encoder.",
                    },
                ),
                "source_fit": (
                    SOURCE_FIT_MODES,
                    {
                        "default": "crop center",
                        "tooltip": "How an anchored Picture 1 is fitted to the output canvas.",
                    },
                ),
                "semantic_resolution": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": MAX_SEMANTIC_RESOLUTION,
                        "step": 32,
                        "tooltip": "Equivalent-square Qwen pixel budget for semantic references.",
                    },
                ),
                "native_reference_size": (
                    NATIVE_SIZE_MODES,
                    {
                        "default": NATIVE_SIZE_MATCH,
                        "tooltip": "Resize policy for native Qwen+VAE references.",
                    },
                ),
                "coverage_views": (
                    "INT",
                    {
                        "default": 12,
                        "min": 2,
                        "max": 24,
                        "step": 1,
                        "tooltip": "Scene-coverage output viewpoints. Ignored by every other mode.",
                    },
                ),
                "coverage_arc_degrees": (
                    "FLOAT",
                    {
                        "default": 360.0,
                        "min": 15.0,
                        "max": 360.0,
                        "step": 15.0,
                        "tooltip": "Scene-coverage camera arc. Ignored by every other mode.",
                    },
                ),
                "coverage_direction": (
                    SCENE_DIRECTIONS,
                    {
                        "default": SCENE_DIRECTION_CLOCKWISE,
                        "tooltip": "Scene-coverage camera direction. Ignored by every other mode.",
                    },
                ),
                "coverage_hold_frames": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 9,
                        "step": 2,
                        "tooltip": "Static frames around each coverage capture. Ignored by every other mode.",
                    },
                ),
                "coverage_loop_closure": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Reuse an anchored source for 360-degree loop closure when possible.",
                    },
                ),
            }
        }

    RETURN_TYPES = (EDIT_OPTIONS_TYPE,)
    RETURN_NAMES = ("options",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(
        self,
        mode: str,
        show_overrides: bool,
        profile_override: str,
        direct_reference_transport: str,
        source_fit: str,
        semantic_resolution: int,
        native_reference_size: str,
        coverage_views: int,
        coverage_arc_degrees: float,
        coverage_direction: str,
        coverage_hold_frames: int,
        coverage_loop_closure: bool,
    ):
        if mode not in EDIT_OPTION_PRESETS:
            raise ValueError(f"Unknown H3 Edit option mode: {mode}")
        prompt_mode, fixed_quality = EDIT_OPTION_PRESETS[mode]
        canonical_quality = fixed_quality or QUALITY_RECOMMENDED
        quality_profile = canonical_quality
        overrides_enabled = bool(show_overrides)
        if overrides_enabled and profile_override != OPTION_PROFILE_CANONICAL:
            compatible_profiles = {
                OPTION_MODE_STILL: set(STILL_QUALITY_PROFILES),
                OPTION_MODE_VERBATIM: set(STILL_QUALITY_PROFILES),
                OPTION_MODE_CHARACTER_SHEET: {QUALITY_CHARACTER_FOUR, QUALITY_CHARACTER_SIX},
                OPTION_MODE_SCENE_COVERAGE: set(SCENE_COVERAGE_PROFILES),
                OPTION_MODE_SCENE_CUTS: set(SCENE_COVERAGE_PROFILES),
                OPTION_MODE_ROOM_OBJECT_STUDY: set(SCENE_COVERAGE_PROFILES),
            }.get(mode, {QUALITY_DIRECTED_CHANGE})
            if profile_override not in compatible_profiles:
                raise ValueError(
                    f"{mode} cannot use profile override {profile_override!r}; use its canonical profile or a "
                    "compatible override."
                )
            quality_profile = profile_override
        if not overrides_enabled:
            direct_reference_transport = REFERENCE_SEMANTIC
            source_fit = "crop center"
            semantic_resolution = 1024
            native_reference_size = NATIVE_SIZE_MATCH
            if mode == OPTION_MODE_ROOM_OBJECT_STUDY:
                coverage_views = 16
            elif mode == OPTION_MODE_SCENE_CUTS:
                coverage_views = 8
            else:
                coverage_views = 12
            coverage_arc_degrees = 360.0
            coverage_direction = SCENE_DIRECTION_CLOCKWISE
            coverage_hold_frames = 5
            coverage_loop_closure = mode not in {OPTION_MODE_SCENE_CUTS, OPTION_MODE_ROOM_OBJECT_STUDY}
        result = {
            "mode": mode,
            "show_overrides": overrides_enabled,
            "prompt_mode": prompt_mode,
            "quality_profile": quality_profile,
            "reference_mode": direct_reference_transport,
            "source_fit": source_fit,
            "semantic_resolution": int(semantic_resolution),
            "native_reference_size": native_reference_size,
            "coverage_views": int(coverage_views),
            "coverage_arc_degrees": float(coverage_arc_degrees),
            "coverage_direction": coverage_direction,
            "coverage_hold_frames": int(coverage_hold_frames),
            "coverage_loop_closure": bool(coverage_loop_closure),
        }
        if mode == OPTION_MODE_ROOM_OBJECT_STUDY:
            result["primary_image_role"] = PRIMARY_SEMANTIC_REFERENCE
            result["reference_mode"] = REFERENCE_SEMANTIC
        return (result,)


class TextEncodeH3Edit:
    """Prepare a source-anchored edit or reference-driven still generation."""

    DESCRIPTION = (
        "Switch Picture 1 between a strong FL2VA source anchor, a semantic FL2VA generation reference, or a native "
        "REF2VA generation reference. Additional ordered guides independently use Qwen-only semantic or native "
        "Qwen+VAE transport. It also compiles frozen-scene camera coverage from one anchored room or semantic/native "
        "references for a completely new room. An optional compiled_prompt input lets an external compiler profile "
        "replace only the built-in prompt construction while preserving the authoring prompt and conditioning graph."
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
                        "tooltip": (
                            "Choose ordinary edit/generation, an anchored directed transformation, frozen-scene camera "
                            "coverage, or verbatim text. Scene coverage requires a matching 124/243/362-frame profile."
                        ),
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
                            "the dedicated sheet decoder. Scene-coverage profiles create a trained-range camera path for "
                            "the dedicated coverage decoder. True 1-frame mode is often poor quality."
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
                "coverage_views": (
                    "INT",
                    {
                        "default": 12,
                        "min": 2,
                        "max": 24,
                        "step": 1,
                        "tooltip": "Number of unique scene viewpoints and output images for frozen scene coverage.",
                    },
                ),
                "coverage_arc_degrees": (
                    "FLOAT",
                    {
                        "default": 360.0,
                        "min": 15.0,
                        "max": 360.0,
                        "step": 15.0,
                        "tooltip": "Physical camera arc from the source/generated opening viewpoint.",
                    },
                ),
                "coverage_direction": (
                    SCENE_DIRECTIONS,
                    {
                        "default": SCENE_DIRECTION_CLOCKWISE,
                        "tooltip": "Direction in which the physical camera travels around the declared orbit center.",
                    },
                ),
                "coverage_hold_frames": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 9,
                        "step": 2,
                        "tooltip": "Requested static frames around each capture; automatically reduced if views are close.",
                    },
                ),
                "coverage_loop_closure": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "For a 360-degree anchored room, internally reuse the one source image as the final keyframe. "
                            "Ignored for partial arcs and reference-generated rooms."
                        ),
                    },
                ),
                "options": (
                    EDIT_OPTIONS_TYPE,
                    {
                        "tooltip": (
                            "Recommended: connect H3 Edit Options to choose one coherent task preset and hide all "
                            "legacy task-specific widgets on this encoder."
                        )
                    },
                ),
                "compiled_prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Optional complete H3 prompt from an external compiler profile. When connected, this "
                            "bypasses only the built-in prompt compiler; the prompt widget remains unchanged, and "
                            "reference images, Qwen ordering, latent timing, and decoder metadata are still prepared "
                            "by this encoder. The connected text must already be a complete H3 prompt."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("positive", "latent", "fitted_source", "encoded_prompt", "info")
    OUTPUT_TOOLTIPS = (
        "Positive conditioning with the selected Picture 1 role and every ordered guide transport.",
        "An H3 latent with the selected still, character-sheet, or frozen-scene temporal profile.",
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
        coverage_views: int = 12,
        coverage_arc_degrees: float = 360.0,
        coverage_direction: str = SCENE_DIRECTION_CLOCKWISE,
        coverage_hold_frames: int = 5,
        coverage_loop_closure: bool = True,
        options: dict[str, Any] | None = None,
        compiled_prompt: str | None = None,
    ):
        import node_helpers

        option_mode = None
        if options is not None:
            if not isinstance(options, dict):
                raise ValueError("H3 Edit options must come from an H3 Edit Options node.")
            option_mode = str(options.get("mode", ""))
            if option_mode not in EDIT_OPTION_PRESETS:
                raise ValueError(f"Unknown H3 Edit option mode: {option_mode}")
            prompt_mode = str(options.get("prompt_mode", prompt_mode))
            quality_profile = str(options.get("quality_profile", quality_profile))
            primary_image_role = str(options.get("primary_image_role", primary_image_role))
            reference_mode = str(options.get("reference_mode", reference_mode))
            source_fit = str(options.get("source_fit", source_fit))
            semantic_resolution = int(options.get("semantic_resolution", semantic_resolution))
            native_reference_size = str(options.get("native_reference_size", native_reference_size))
            coverage_views = int(options.get("coverage_views", coverage_views))
            coverage_arc_degrees = float(options.get("coverage_arc_degrees", coverage_arc_degrees))
            coverage_direction = str(options.get("coverage_direction", coverage_direction))
            coverage_hold_frames = int(options.get("coverage_hold_frames", coverage_hold_frames))
            coverage_loop_closure = bool(options.get("coverage_loop_closure", coverage_loop_closure))

        semantic_survey_task = prompt_mode == PROMPT_ROOM_OBJECT_STUDY
        if semantic_survey_task:
            # Room/object study is deliberately an all-semantic reconstruction.
            # Preserve a disconnected direct-reference socket, but coerce every
            # connected picture away from native/VAE transport.
            primary_image_role = PRIMARY_SEMANTIC_REFERENCE
            if reference_mode != REFERENCE_NONE:
                reference_mode = REFERENCE_SEMANTIC

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
        directed_task = prompt_mode in DIRECTED_PROMPT_MODES
        scene_task = prompt_mode in SCENE_PROMPT_MODES
        cinematic_cut_task = prompt_mode in {PROMPT_SCENE_CUTS, PROMPT_ROOM_OBJECT_STUDY}
        scene_profile = quality_profile in SCENE_COVERAGE_PROFILES
        if directed_task and primary_image_role != PRIMARY_EDIT_ANCHOR:
            raise ValueError("Directed re-pose, character-swap, and camera-angle tasks require the strong Picture 1 anchor.")
        if directed_task and quality_profile != QUALITY_DIRECTED_CHANGE:
            raise ValueError("Directed tasks require 'directed change | 39-frame settle -> 1 image'.")
        if scene_task and not scene_profile:
            raise ValueError("Frozen scene coverage requires a 'scene coverage | ... camera path' quality profile.")
        if scene_profile and not scene_task:
            raise ValueError("Scene-coverage quality profiles require a frozen scene-coverage prompt mode.")
        if coverage_direction not in SCENE_DIRECTIONS:
            raise ValueError(f"Unknown coverage direction: {coverage_direction}")
        coverage_views = max(2, min(24, int(coverage_views)))
        coverage_arc_degrees = max(15.0, min(360.0, float(coverage_arc_degrees)))
        coverage_hold_frames = max(1, min(9, int(coverage_hold_frames)))
        if quality_profile in CHARACTER_SHEET_FRAME_INDICES and primary_image_role == PRIMARY_EDIT_ANCHOR:
            raise ValueError(
                "Character-sheet profiles require semantic or native generation mode; Picture 1 cannot be a frame anchor."
            )

        width = _round_dimension(width)
        height = _round_dimension(height)
        primary_image = _validate_image(source_image, "source_image")
        anchored_edit = primary_image_role == PRIMARY_EDIT_ANCHOR
        scene_loop_closure = bool(
            scene_task
            and anchored_edit
            and not cinematic_cut_task
            and coverage_loop_closure
            and math.isclose(coverage_arc_degrees, 360.0, abs_tol=0.001)
        )
        source_latent = None
        visual_items: list[dict[str, Any]] = []
        ref_blocks: list[dict[str, Any]] = []
        reference_specs: list[dict[str, Any]] = []
        if anchored_edit:
            fitted_source = _resize(primary_image, width, height, source_fit)
            source_latent = vae.encode(fitted_source)
            visual_items.append({"type": "image", "data": fitted_source})
            if scene_loop_closure:
                visual_items.append({"type": "image", "data": fitted_source})
                first_reference_ordinal = 3
            else:
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
                        "transport": REFERENCE_SEMANTIC if semantic_survey_task else transport,
                        "semantic_resolution": int(item.get("semantic_resolution", semantic_resolution)),
                        "native_reference_size": size_mode,
                    }
                )

        if prompt_mode in {PROMPT_REPOSE, PROMPT_CHARACTER_SWAP} and not reference_specs:
            raise ValueError(f"{prompt_mode} requires at least one connected guide after Picture 1.")

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
            reference_note = (
                "No alternate-angle guides; Picture 1 and its internal Picture 2 loop duplicate are presented to Qwen."
                if scene_loop_closure
                else "No guide references; only Picture 1 is presented to Qwen."
            )
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
        if option_mode:
            latent["h3edit_option_mode"] = option_mode
        scene_capture_plan = None
        if scene_task:
            capture_planner = (
                _room_object_capture_plan
                if prompt_mode == PROMPT_ROOM_OBJECT_STUDY
                else _scene_capture_plan
            )
            scene_capture_plan = capture_planner(
                natural_frames, coverage_views, coverage_arc_degrees, coverage_hold_frames
            )
            latent["h3edit_scene_capture_centers"] = scene_capture_plan["centers"]
            latent["h3edit_scene_capture_windows"] = scene_capture_plan["windows"]
            latent["h3edit_scene_capture_angles"] = scene_capture_plan["angles"]
            latent["h3edit_scene_direction"] = coverage_direction
            latent["h3edit_scene_loop_closure"] = scene_loop_closure
        if quality_profile == QUALITY_DIRECTED_CHANGE:
            latent["h3edit_selection_strategy"] = "settled_tail"
            latent["h3edit_tail_candidates"] = 5
            latent["h3edit_directed_task"] = prompt_mode
        additional_modes = [item["transport"] for item in reference_specs]
        if not anchored_edit:
            additional_modes = additional_modes[1:]
        external_compiler = compiled_prompt is not None
        if external_compiler:
            encoded_prompt = str(compiled_prompt).strip()
            if not encoded_prompt:
                raise ValueError("The connected compiled_prompt is empty; connect a complete H3 prompt or disconnect it.")
        else:
            encoded_prompt = _build_prompt(
                prompt,
                prompt_mode,
                additional_modes,
                requested_frames,
                primary_image_role,
                scene_capture_plan=scene_capture_plan,
                scene_direction=coverage_direction,
                scene_loop_closure=scene_loop_closure,
            )
        tokens = clip.tokenize(encoded_prompt, minimax_ref_items=visual_items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning_values: dict[str, Any] = {"minimax_frame_count": natural_frames}
        if anchored_edit:
            keyframes = [{"resolved_frame_index": 0, "latent": source_latent}]
            if scene_loop_closure:
                keyframes.append({"resolved_frame_index": natural_frames - 1, "latent": source_latent})
            conditioning_values["minimax_keyframes"] = keyframes
        if ref_blocks:
            conditioning_values["minimax_refs"] = ref_blocks
        conditioning = node_helpers.conditioning_set_values(conditioning, conditioning_values)

        ignored_note = " The direct reference_image is intentionally ignored." if ignored_direct else ""
        if anchored_edit:
            task_label = f" | {prompt_mode}" if directed_task or scene_task else ""
            mode_note = (
                f"Strong-anchor edit{task_label} | Picture 1 native frame-zero keyframe ({tuple(source_latent.shape)})"
            )
        else:
            route = "REF2VA" if native_count else "FL2VA"
            mode_note = f"Reference-driven generation | no frame-zero keyframe | expected route {route}"
        if quality_profile in CHARACTER_SHEET_FRAME_INDICES:
            decoder_note = "Decode H3 Character Sheet extracts the calibrated views and returns a stitched sheet."
        elif scene_profile:
            decoder_note = (
                f"Decode H3 Scene Coverage scores {coverage_views} timed hold windows and returns every selected view "
                "plus a contact sheet."
            )
        elif quality_profile == QUALITY_DIRECTED_CHANGE:
            decoder_note = (
                "Decode H3 Edit to One Image scores only the settled tail and returns the completed transformation."
            )
        else:
            decoder_note = "Decode H3 Edit to One Image scores the decoded context and returns one stable high-quality still."
        info = (
            f"{mode_note} | {option_mode + ' | ' if option_mode else ''}{quality_profile} | "
            f"prompt compiler={'external compiled_prompt' if external_compiler else 'built-in'} | "
            f"requested context {requested_frames} frames | "
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


def _stitch_scene_panels(
    panels: torch.Tensor,
    columns: int,
    gutter_px: int,
    gutter_value: float,
) -> tuple[torch.Tensor, int, int]:
    if panels.ndim != 4 or int(panels.shape[0]) < 1:
        raise ValueError("Scene-coverage panels must be a non-empty IMAGE batch [N,H,W,C].")
    panel_count = int(panels.shape[0])
    columns = max(1, min(int(columns), panel_count))
    rows = math.ceil(panel_count / columns)
    missing = rows * columns - panel_count
    if missing:
        filler = panels.new_full((missing, *panels.shape[1:]), gutter_value)
        grid_panels = torch.cat((panels, filler), dim=0)
    else:
        grid_panels = panels
    return _stitch_character_panels(grid_panels, columns, gutter_px, gutter_value), columns, rows


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
        selection_strategy = str(samples.get("h3edit_selection_strategy", "full_context"))
        if selection_strategy == "settled_tail":
            tail_count = min(candidate_count, max(1, int(samples.get("h3edit_tail_candidates", 5))))
            first_candidate = candidate_count - tail_count
            candidates = frames[first_candidate:candidate_count]
        else:
            first_candidate = 0
            candidates = frames[:candidate_count]
        relative_index, score = _stable_quality_frame(candidates)
        frame_index = first_candidate + relative_index
        image = frames[frame_index : frame_index + 1].clone()
        selection_note = (
            f"settled-tail frames {first_candidate}-{candidate_count - 1}"
            if selection_strategy == "settled_tail"
            else f"{candidate_count} requested candidate(s)"
        )
        return (
            image,
            f"Decoded H3 latent {tuple(video.shape)} to {decoded_frames} frame(s); scored {selection_note} and returned "
            f"stable-quality frame {frame_index} (score {score:.4f}) as one image {tuple(image.shape)}.",
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


class DecodeH3SceneCoverage:
    """Decode a frozen-scene camera path into individually scored viewpoints."""

    DESCRIPTION = (
        "Decodes an H3 frozen-scene camera path, scores each encoded hold window for a stable crisp frame, and returns "
        "every selected viewpoint plus a contact sheet and the complete generated path."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Sampled latent from frozen scene coverage mode."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "columns": (
                    "INT",
                    {"default": 4, "min": 1, "max": 8, "step": 1, "tooltip": "Contact-sheet columns."},
                ),
                "gutter_px": ("INT", {"default": 6, "min": 0, "max": 256, "step": 1}),
                "gutter_color": (GUTTER_COLORS, {"default": "black"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("contact_sheet", "selected_views", "all_frames", "info")
    OUTPUT_TOOLTIPS = (
        "One contact sheet containing every selected scene viewpoint.",
        "The selected viewpoints as an IMAGE batch in camera-path order.",
        "Every decoded camera-path frame for inspection or alternate picks.",
        "Capture targets, chosen frames, angles, windows, loop mode, and sheet dimensions.",
    )
    FUNCTION = "decode"
    CATEGORY = CATEGORY

    def decode(
        self,
        samples: dict[str, Any],
        vae: Any,
        columns: int,
        gutter_px: int,
        gutter_color: str,
    ):
        if gutter_color not in GUTTER_COLORS:
            raise ValueError(f"Unknown gutter color: {gutter_color}")
        windows = samples.get("h3edit_scene_capture_windows") if isinstance(samples, dict) else None
        centers = samples.get("h3edit_scene_capture_centers") if isinstance(samples, dict) else None
        angles = samples.get("h3edit_scene_capture_angles") if isinstance(samples, dict) else None
        if not isinstance(windows, (tuple, list)) or not windows:
            raise ValueError("Decode H3 Scene Coverage requires a latent encoded by frozen scene coverage mode.")
        if not isinstance(centers, (tuple, list)) or len(centers) != len(windows):
            raise ValueError("Frozen scene coverage latent has invalid capture-center metadata.")

        video, frames, decoded_frames = _decode_h3_frames(samples, vae, "Decode H3 Scene Coverage")
        requested_frames = min(
            decoded_frames,
            max(1, int(samples.get("h3edit_requested_frames", decoded_frames))),
        )
        selected_indices = []
        selected_frames = []
        for view_number, window in enumerate(windows, start=1):
            if not isinstance(window, (tuple, list)) or len(window) != 2:
                raise ValueError(f"Frozen scene capture window {view_number} is invalid: {window}")
            start = max(0, int(window[0]))
            end = min(requested_frames - 1, int(window[1]))
            if start > end:
                raise ValueError(
                    f"Frozen scene capture window {view_number} ({start}-{end}) is outside "
                    f"the decoded {requested_frames}-frame path."
                )
            relative_index, _score = _stable_quality_frame(frames[start : end + 1])
            selected_index = start + relative_index
            selected_indices.append(selected_index)
            selected_frames.append(frames[selected_index])

        selected = torch.stack(selected_frames).clone()
        gutter_value = {"black": 0.0, "neutral gray": 0.5, "white": 1.0}[gutter_color]
        sheet, resolved_columns, rows = _stitch_scene_panels(
            selected,
            columns,
            gutter_px,
            gutter_value,
        )
        angle_note = [round(float(angle), 3) for angle in angles] if isinstance(angles, (tuple, list)) else []
        direction = str(samples.get("h3edit_scene_direction", "unknown direction"))
        loop_note = "first/final source loop closure" if samples.get("h3edit_scene_loop_closure") else "open path"
        return (
            sheet,
            selected,
            frames,
            f"Decoded H3 latent {tuple(video.shape)} to {decoded_frames} frame(s); scored capture windows "
            f"{[list(window) for window in windows]} and selected frames {selected_indices} at angles {angle_note}; "
            f"{direction}, {loop_note}; stitched {resolved_columns}x{rows} contact sheet {tuple(sheet.shape)} with "
            f"{gutter_px}px {gutter_color} gutters.",
        )


NODE_CLASS_MAPPINGS = {
    "AddH3EditReference": AddH3EditReference,
    "H3EditOptions": H3EditOptions,
    "TextEncodeH3Edit": TextEncodeH3Edit,
    "DecodeH3SingleFrame": DecodeH3SingleFrame,
    "DecodeH3CharacterSheet": DecodeH3CharacterSheet,
    "DecodeH3SceneCoverage": DecodeH3SceneCoverage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddH3EditReference": "Add H3 Edit Reference",
    "H3EditOptions": "H3 Edit Options",
    "TextEncodeH3Edit": "Text Encode H3 Edit / Generate",
    "DecodeH3SingleFrame": "Decode H3 Edit to One Image",
    "DecodeH3CharacterSheet": "Decode H3 Character Sheet",
    "DecodeH3SceneCoverage": "Decode H3 Scene Coverage",
}
