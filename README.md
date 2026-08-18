# ComfyUI MiniMax H3 Edit

A deliberately small ComfyUI node pack for **single-image MiniMax H3 generation and photo editing**. It has no Director, hidden state, prompt analyzer, model loader, sampler wrapper, or custom frontend.

The pack adds four nodes:

- **Add H3 Edit Reference** — builds an ordered, chainable reference stack; every image independently chooses semantic or native transport.
- **Text Encode H3 Edit / Generate** — switches between a strong source-photo edit and reference-driven generation, then builds H3 conditioning from optional ordered guides.
- **Decode H3 Edit to One Image** — decodes the packet, scores its candidates, and returns one stable high-quality frame.
- **Decode H3 Character Sheet** — extracts calibrated views from a 73/124-frame character orbit and stitches a 2x2 or 3x2 sheet.

The still path produces exactly one image. By default, it samples the same short 5-frame context used by H3 Studio because H3 is a video model and a literal one-token latent leaves no temporal context around the result. Character-sheet profiles instead produce one stitched sheet plus optional selected/all-frame batches. The old true-one-frame path remains available as an explicitly experimental, low-quality option.

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

## Quality profiles

The quality selector chooses the temporal context. Still profiles feed the one-image decoder; character-sheet profiles feed the sheet decoder.

| Profile | Candidate frames | Video latent tokens | Use |
|---|---:|---:|---|
| `recommended` | 5 | 2 | H3 Studio's default short-context balance |
| `extended` | 9 | 3 | More candidates at moderately higher sampling cost |
| `high` | 13 | 4 | Additional temporal context for difficult edits |
| `maximum` | 20 | 7 | Slowest option; its natural 22-frame decode is cropped to 20 candidates |
| `experimental` | 1 | 1 | Literal one-frame path retained for comparison; often poor quality |
| `character sheet · 4 panels` | 73 | 22 | 180-degree body orbit plus front facial view |
| `character sheet · 6 panels` | 124 | 37 | 360-degree body orbit plus two facial views |

The first five entries are Studio-style compact still-image profiles. Their prompt asks for a locked camera and an immediately completed result held unchanged, then the still decoder scores sharpness, contrast, exposure, and temporal stability. The two character-sheet profiles deliberately use long moving-camera orbits and fixed extraction indices instead.

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
4. For one additional guide, use its direct `reference_image` socket. For multiple guides, chain **Add H3 Edit Reference** nodes into `reference_stack`.
5. Declare semantic or native transport for each guide and refer to it with its ordered `<Picture N>` tag.
6. Use `ModelSamplingMiniMaxH3`, `BasicGuider`, `RandomNoise`, a sampler and scheduler, and `SamplerCustomAdvanced` as in the normal native H3 graph.
7. Connect the sampled latent to **Decode H3 Edit to One Image**, then preview or save its image output.

The included [mixed-reference still workflow](example_workflows/H3_Edit_Mixed_References.json) defaults to a single-image FL2VA edit using the recommended hidden 5-frame context. Its encoder exposes the role switch directly. It uses semantic glasses as `<Picture 2>` and a native wardrobe/material guide as `<Picture 3>`. Replace its model filenames and input images with files available in your ComfyUI installation; select the REF2VA model before using a native Picture 1 generation role.

For a character sheet, load [H3_Character_Sheet_6_Panel.json](example_workflows/H3_Character_Sheet_6_Panel.json). Replace all three placeholder images and rewrite the per-picture assignment prompt before queueing. A 124-frame REF2VA pass is substantially slower and uses more memory than the still profiles.

## Prompt behavior

`edit instruction` applies either a preservation contract or a new-image creation contract according to the primary-image switch, and explicitly scopes every `<Picture N>` as semantic or native. `use prompt verbatim` leaves the text unchanged; the Qwen picture blocks are still prepended in picture order.

Example instruction:

```text
Add the black acetate glasses from <Picture 2> to the woman. Fit them naturally to her face with realistic temples, reflections, and contact shadows. Change nothing else.
```

Example generation instruction:

```text
Create a completely new nighttime street portrait of the woman from <Picture 1>, wearing the glasses from <Picture 2> and the jacket from <Picture 3>. Use a new pose, background, lighting design, and composition.
```

## License and provenance

MIT. This is an independent implementation built against ComfyUI's public MiniMax H3 conditioning contracts. The experimental one-token fallback was interoperability-checked against [ComfyUI-MiniMaxH3-SingleFrame](https://github.com/tori29umai0123/ComfyUI-MiniMaxH3-SingleFrame), but this pack does not copy its unlicensed code.

The continuous-orbit character-sheet method and calibrated four/six-panel frame selections are adapted with attribution from C_Nugget's [H3 Character Sheet Generator](https://huggingface.co/PoopMan333/H3_Character_Sheet_Generator). This pack replaces its prompt concatenation and stitching subgraph with its own structured prompt compiler and decoder. MiniMax H3 model and output use remains subject to the [MiniMax H3 community license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
