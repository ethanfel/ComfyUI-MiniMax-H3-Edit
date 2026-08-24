# ComfyUI MiniMax H3 Edit

A deliberately small ComfyUI node pack for **single-image MiniMax H3 generation and photo editing**. It has no Director, hidden state, prompt analyzer, model loader, or sampler wrapper. Its small frontend keeps task-specific settings off the main encoder and exposes them through the Options node.

The pack adds six nodes:

- **Add H3 Edit Reference** — builds an ordered, chainable reference stack; every image independently chooses semantic or native transport.
- **H3 Edit Options** — selects one canonical task preset; expert overrides remain collapsed until explicitly enabled.
- **Text Encode H3 Edit / Generate** — switches between a strong source-photo edit and reference-driven generation, then builds H3 conditioning from optional ordered guides.
- **Decode H3 Edit to One Image** — decodes the packet, scores its candidates, and returns one stable high-quality frame.
- **Decode H3 Character Sheet** — extracts calibrated views from a 73/124-frame character orbit and stitches a 2x2 or 3x2 sheet.
- **Decode H3 Scene Coverage** — scores every timed camera hold and returns 2–24 consistent viewpoints, a contact sheet, and the complete camera path.

The still path produces exactly one image. By default, it samples the same short 5-frame context used by H3 Studio because H3 is a video model and a literal one-token latent leaves no temporal context around the result. Character-sheet profiles instead produce one stitched sheet plus optional selected/all-frame batches. The old true-one-frame path remains available as an explicitly experimental, low-quality option.

## One mode, canonical settings

The main encoder is always compact. Its old task-specific inputs remain in the backend only so saved workflows still load; they are no longer displayed as editable widgets. Connect **H3 Edit Options** to the encoder's `options` input and normal operation requires one choice: `mode`. Every mode sends a complete compatible preset:

| Mode | Canonical settings |
|---|---|
| `still \| edit or generate` | recommended 5-frame still context |
| `directed \| re-pose character` | re-pose compiler + 39-frame settled change |
| `directed \| character swap` | swap compiler + 39-frame settled change |
| `directed \| new camera angle` | camera compiler + 39-frame settled change |
| `character sheet \| canonical 6 views` | six-view compiler + 124-frame orbit |
| `scene coverage \| canonical camera path` | frozen-scene compiler + 124 frames + 12 views + 360° clockwise arc + five-frame holds + loop closure |
| `scene coverage \| cinematic hard cuts` | frozen-scene compiler + 124 frames + eight discrete cinematic shots around one named target; no camera travel |
| `advanced \| prompt verbatim` | unchanged prompt + recommended 5-frame context |

Leave `show_overrides` disabled and hidden stale values are ignored—the canonical preset is authoritative. Enable it only when deliberately changing a compatible frame profile, reference transport/size, source fit, or scene-coverage geometry. An incompatible profile override produces a clear validation error rather than silently building a contradictory task.

## Primary image role switch

The encoder's **primary image role** switch changes what `<Picture 1>` means and which checkpoint route the graph expects:

| Role | Picture 1 conditioning | Scene lock | Checkpoint |
|---|---|---:|---|
| `edit \| strong scene anchor (FL2VA)` | Qwen + frame-zero VAE keyframe | strong | FL2VA |
| `generate \| semantic Picture 1 (FL2VA)` | Qwen visual tokens only | none | FL2VA |
| `generate \| native Picture 1 (REF2VA)` | Qwen + `minimax_refs` | none | REF2VA |

Both generation roles remove `minimax_keyframes` entirely and use a creation prompt contract. Picture 1 remains an input reference, but it no longer fixes the source pixels, geometry, background, or composition. Because this pack uses standard model loaders, change the diffusion model in `UNETLoader` when switching between the FL2VA and REF2VA roles.

## Semantic versus native references

In the default edit role, `<Picture 1>` is the image being edited. It goes through both Qwen and the H3 VAE and is attached as the frame-zero keyframe. In either generation role it is instead the first reference and is never attached as a keyframe.

Every optional guide has a transport selector:

| Mode | Qwen visual tokens | H3 VAE latent | Conditioning payload | Intended use |
|---|---:|---:|---|---|
| `semantic (Qwen only)` | yes | no | no `minimax_refs` | Transfer the idea of glasses, an object category, a material, palette, or attribute |
| `native (Qwen + VAE ref)` | yes | yes | one `minimax_refs` block | Stronger low-level visual reference; experimental when combined with the source keyframe |
| `none (source only)` | no direct Picture 2 | no direct Picture 2 | no direct guide payload | Disable the direct-reference socket |

For “add these glasses to this woman,” connect the woman to `source_image`, the glasses image to `reference_image`, and use `semantic (Qwen only)`. That direct socket is the convenient `<Picture 2>` path. A semantic guide can use an equivalent-square Qwen budget from 256 to 3584 pixels without allocating another VAE latent.

## Scalable ordered references

For more guides, chain **Add H3 Edit Reference** nodes and connect the final `references` output to `reference_stack` on **Text Encode H3 Edit**. The stack has no fixed reference count. An IMAGE batch is expanded into separate ordered references, and additional builder nodes can be chained within normal ComfyUI graph limits. In practice, Qwen context length, conditioning time, RAM, and VRAM still limit useful reference counts.

Ordering is deterministic:

1. `source_image` is always `<Picture 1>`; the primary-image switch determines whether it is an edit anchor or a generation reference.
2. A connected direct `reference_image` is `<Picture 2>` unless its mode is `none`.
3. Stack entries follow in chain and batch order as the next `<Picture N>` values.

Each builder declares its own `semantic (Qwen only)` or `native (Qwen + VAE ref)` transport, semantic resolution, and native resize policy. Semantic and native entries can be used simultaneously in the same stack—for example, semantic glasses as `<Picture 2>`, a native texture as `<Picture 3>`, and a semantic hairstyle as `<Picture 4>`. Only `<Picture 3>` is VAE-encoded in that example. Mixed native guide latents remain experimental; an all-semantic guide stack uses only the source VAE anchor.

Mixed transport is intentionally labeled experimental. ComfyUI's H3 packed layout can carry the payload, but Qwen-only pictures do not have corresponding VAE reference blocks and the released FL2VA and REF2VA weights were trained for different task presentations. Runtime compatibility does not guarantee that every checkpoint responds well to every mixed ordering.

## Character sheets

The two character-sheet profiles generate all views in one continuous H3 pass so identity, costume, palette, and proportions share one denoising trajectory. The encoder automatically writes a silent locked-character turntable prompt in H3 full-reference structure; your prompt only needs to assign a job and ignore list to each `<Picture N>`.

| Profile | Orbit frames | Extracted indices | Sheet |
|---|---:|---|---|
| `4 panels` | 73 | `2, 24, 45, 68` | 2x2 |
| `6 panels` | 124 | `2, 21, 42, 63, 84, 113` | 3x2 |

Use a generation role rather than the strong edit anchor. Native Picture 1 plus native builders is the default high-fidelity REF2VA route. An all-semantic set uses Qwen-only FL2VA generation without any input VAE encoding; semantic/native mixing remains experimental.

**Decode H3 Character Sheet** returns:

- one stitched sheet;
- the four or six selected views as a batch;
- all decoded orbit frames for manual frame selection;
- the resolved indices and dimensions as diagnostic text.

The included [six-panel character-sheet workflow](example_workflows/H3_Character_Sheet_6_Panel.json) uses three chained native references, 768x1344 frames, Euler sampling, the linear-quadratic scheduler, 25 steps, and a REF2VA checkpoint. The optional selected-view and all-frame save nodes are bypassed by default. Add, remove, or change builder transports as needed; the final builder always connects to `reference_stack`.

## Directed transformations

The three directed prompt presets turn a still edit into one tightly constrained H3 motion, let it settle, and extract the finished result as one image. They always use Picture 1 as a strong FL2VA frame-zero anchor and the `directed change | 39-frame settle -> 1 image` profile. The prompt compiler asks H3 to complete the change by 65% of the 39-frame sequence and hold it perfectly still; the decoder then scores only zero-based frames 34–38.

| Prompt preset | Picture 2 | Allowed change | Locked elements |
|---|---|---|---|
| `directed \| re-pose character` | Required pose guide; semantic recommended | Body pose and explicitly requested expression | Identity, wardrobe, scene, lighting, lens, framing, camera |
| `directed \| character swap` | Required donor character; semantic recommended | Explicitly assigned identity, face, hair, physique, wardrobe, accessories | Source pose, placement, action, scene, camera, perspective, lighting |
| `directed \| new camera angle` | Optional angle/composition guide | One smooth camera arc to the requested azimuth, elevation, distance, and framing | Subject, pose, expression, wardrobe, props, world geometry, lighting |

Write only the transformation you want. The selected preset rewrites it into the full timed I2VA prompt, including the Picture 1 alignment line, preservation rules, controlled-motion phase, settled tail, and silent audio fields. Re-pose and character swap reject a missing guide; new camera angle can run from Picture 1 and text alone. Additional semantic or native guides can be chained as usual, although native guides mixed with the frame-zero FL2VA keyframe remain experimental.

The included [directed-transformations workflow](example_workflows/H3_Directed_Transformations.json) defaults to semantic re-posing with Euler, the linear-quadratic scheduler, and 25 steps. Its visible note contains example instructions for all three presets and credits the continuous-camera method that inspired this extension.

## Frozen scene coverage

`scene coverage | canonical camera path` generalizes the character-sheet camera method from one isolated character to a complete rigid room or scene. Selecting this one mode supplies a complete 124-frame, 12-view, 360° clockwise preset. Enable overrides only when a 243/362-frame path, different view count, partial arc, opposite direction, different hold, or disabled loop closure is intentional. Decode with **Decode H3 Scene Coverage**. The compiler writes exact timed camera waypoints and static capture windows; the decoder scores every window independently and returns its best stable frame.

`scene coverage | cinematic hard cuts` is the discontinuous alternative. In the instruction, identify one exact coverage target—a person, object, architectural feature, or fixed point in the scene. The canonical preset creates eight separately composed static shots around that target, using wide, three-quarter, profile, low-angle, high-angle, reverse, detail, and hero-style setups. Every cut has an exact timestamp and every shot has a timed extraction window. The prompt explicitly forbids an orbit or any intermediate camera travel: the first frame after each hard cut must already be the resolved new perspective. The scene remains frozen across the cut; only camera placement, height, lens, and framing may change.

Example instruction:

```text
Coverage target: the woman seated at the desk in <Picture 1>. Keep her centered as the persistent visual subject while preserving her exact pose, expression, wardrobe, desk, room geometry, props, materials, lighting, and shadows across every cinematic camera cut.
```

There are two scene origins:

| Primary image role | Result | Input VAE anchor |
|---|---|---:|
| `edit \| strong scene anchor (FL2VA)` | Freeze and cover the room shown in Picture 1 | first frame; also final frame for optional 360° loop closure |
| `generate \| semantic Picture 1 (FL2VA)` | Create a completely new room from Qwen-only design references, freeze it, then cover it | none |
| `generate \| native Picture 1 (REF2VA)` | Create a new room with native high-detail references, freeze it, then cover it | no timeline keyframe; native reference blocks only |

The anchored route works with only `source_image`. For a 360° path with loop closure enabled, the node internally presents that one image as both Picture 1 and Picture 2 and anchors the first and final frames to the same VAE latent. The user does not need to load the image twice. For a partial arc, only the first frame is anchored.

Optional images from **Add H3 Edit Reference** are treated as additional angles of the same physical anchored room. They constrain occluded walls, openings, fixtures, furniture, materials, object placement, and lighting in one shared coordinate system. Semantic Qwen-only angles are recommended first; native alternate views remain experimental when mixed with FL2VA timeline keyframes.

In semantic generation mode, Picture 1 and every semantic stack entry are design references rather than source frames. H3 synthesizes one new coherent room from their explicitly assigned architecture, furniture, materials, palette, or lighting, then freezes that generated world while the camera moves. An all-semantic setup does not VAE-encode any input image.

| Coverage profile | Duration at 24 fps | Practical starting point |
|---|---:|---:|
| 124 frames | 5.17 s | 6–8 views |
| 243 frames | 10.13 s | 8–12 views |
| 362 frames | 15.08 s | 12–16 views |

The interface permits up to 24 captures. More views mean shorter travel and hold windows; 362 frames is the safer choice for dense coverage. A single photograph cannot contain ground-truth geometry for invisible surfaces, so those areas remain a conservative H3 reconstruction. Real alternate angles improve fidelity substantially.

## Quality profiles

The quality selector chooses the temporal context. Still profiles feed the one-image decoder; character-sheet profiles feed the sheet decoder.

| Profile | Candidate frames | Video latent tokens | Use |
|---|---:|---:|---|
| `recommended` | 5 | 2 | H3 Studio's default short-context balance |
| `extended` | 9 | 3 | More candidates at moderately higher sampling cost |
| `high` | 13 | 4 | Additional temporal context for difficult edits |
| `maximum` | 20 | 7 | Slowest option; its natural 22-frame decode is cropped to 20 candidates |
| `experimental` | 1 | 1 | Literal one-frame path retained for comparison; often poor quality |
| `directed change` | 39 | 12 | One controlled transformation; decoder scores settled tail frames 34–38 |
| `character sheet · 4 panels` | 73 | 22 | 180-degree body orbit plus front facial view |
| `character sheet · 6 panels` | 124 | 37 | 360-degree body orbit plus two facial views |

The first five entries are Studio-style compact still-image profiles. Their prompt asks for a locked camera and an immediately completed result held unchanged, then the still decoder scores sharpness, contrast, exposure, and temporal stability. The directed profile assigns the temporal budget to exactly one transformation before scoring only the completed tail. The two character-sheet profiles deliberately use long moving-camera orbits and fixed extraction indices instead.

This pack does not freeze temporal RoPE. The quality paths intentionally use H3's natural temporal coordinates so the model can evolve away from the edit anchor or form a new reference-driven image. In true-one-frame mode there is only one target temporal coordinate, so a RoPE freeze would be a no-op.

## Requirements

- A current ComfyUI build with native MiniMax H3 support.
- MiniMax H3 FL2VA diffusion model for anchored edits and semantic-only generation.
- MiniMax H3 REF2VA diffusion model for native reference generation.
- MiniMax H3 Qwen3-VL text encoder.
- MiniMax H3 video VAE.

The edit path still uses the H3 VAE for Picture 1. In semantic generation, “Qwen only” also applies to Picture 1, so no input image is VAE-encoded.

## Install

Place the folder in `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ethanfel/ComfyUI-MiniMax-H3-Edit
```

Restart ComfyUI. The nodes appear under `MiniMax H3/Edit`.

## Basic graph

1. Load the H3 diffusion model, Qwen text encoder, and video VAE with standard ComfyUI loaders.
2. Connect the photo being edited or the first generation reference to `source_image` on **Text Encode H3 Edit / Generate**.
3. Select its role: strong edit anchor, semantic generation reference, or native generation reference.
4. Connect **H3 Edit Options** and select one `mode`. Leave overrides closed for its canonical settings.
5. For one additional guide, use the direct `reference_image` socket. For multiple guides, chain **Add H3 Edit Reference** nodes into `reference_stack`.
6. Declare semantic or native transport for each stacked guide and refer to it with its ordered `<Picture N>` tag.
7. Use `ModelSamplingMiniMaxH3`, `BasicGuider`, `RandomNoise`, a sampler and scheduler, and `SamplerCustomAdvanced` as in the normal native H3 graph.
8. Decode with the output node matching the selected task: one image, character sheet, or scene coverage.

The included [mixed-reference still workflow](example_workflows/H3_Edit_Mixed_References.json) defaults to a single-image FL2VA edit using the recommended hidden 5-frame context. Its encoder exposes the role switch directly. It uses semantic glasses as `<Picture 2>` and a native wardrobe/material guide as `<Picture 3>`. Replace its model filenames and input images with files available in your ComfyUI installation; select the REF2VA model before using a native Picture 1 generation role.

For a character sheet, load [H3_Character_Sheet_6_Panel.json](example_workflows/H3_Character_Sheet_6_Panel.json). Replace all three placeholder images and rewrite the per-picture assignment prompt before queueing. A 124-frame REF2VA pass is substantially slower and uses more memory than the still profiles.

For re-posing, character replacement, or camera movement, load [H3_Directed_Transformations.json](example_workflows/H3_Directed_Transformations.json). Keep the strong-anchor role and directed-change quality profile selected, then choose one of the three `directed` prompt modes. The workflow note includes a short prompt for each mode.

## Prompt behavior

`edit instruction` applies either a preservation contract or a new-image creation contract according to the primary-image switch, and explicitly scopes every `<Picture N>` as semantic or native. The three `directed` modes compile short instructions into distinct timed re-pose, character-swap, or camera-angle contracts. `use prompt verbatim` leaves the text unchanged; the Qwen picture blocks are still prepended in picture order.

For a larger task-aware editor, connect the `text` output from
[H3 Prompt IDE](https://github.com/ethanfel/ComfyUI-H3-Prompt-IDE) directly to
this node's `prompt` input. Prompt IDE 0.8.2 and later reads the connected Options mode and detects the selected edit,
re-pose, character-swap, new-angle, character-sheet, or frozen-scene task and displays its
matching short instruction template. This encoder still owns the full H3 task
and timing wrapper. With the verbatim mode, Prompt IDE
keeps its manually selected full H3 schema instead.

Example instruction:

```text
Add the black acetate glasses from <Picture 2> to the woman. Fit them naturally to her face with realistic temples, reflections, and contact shadows. Change nothing else.
```

Example generation instruction:

```text
Create a completely new nighttime street portrait of the woman from <Picture 1>, wearing the glasses from <Picture 2> and the jacket from <Picture 3>. Use a new pose, background, lighting design, and composition.
```

Example directed instructions:

```text
Re-pose: Use <Picture 2> only for the target body pose: left hand on hip and weight on the left leg.
Character swap: Replace the woman with the woman from <Picture 2>, including her face, hair, physique, and wardrobe.
New camera angle: Arc 45 degrees to camera right at the same height and focal length, keeping a medium shot.
```

## License and provenance

MIT. This is an independent implementation built against ComfyUI's public MiniMax H3 conditioning contracts. The experimental one-token fallback was interoperability-checked against [ComfyUI-MiniMaxH3-SingleFrame](https://github.com/tori29umai0123/ComfyUI-MiniMaxH3-SingleFrame), but this pack does not copy its unlicensed code.

The continuous-orbit character-sheet method and calibrated four/six-panel frame selections are adapted with attribution from C_Nugget's [H3 Character Sheet Generator](https://huggingface.co/PoopMan333/H3_Character_Sheet_Generator). This pack replaces its prompt concatenation and stitching subgraph with its own structured prompt compiler and decoder. MiniMax H3 model and output use remains subject to the [MiniMax H3 community license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
