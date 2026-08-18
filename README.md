# ComfyUI MiniMax H3 Edit

A deliberately small ComfyUI node pack for **single-image MiniMax H3 photo editing**. It has no Director, hidden state, prompt analyzer, model loader, sampler wrapper, or custom frontend.

The pack adds three nodes:

- **Add H3 Edit Reference** — builds an ordered, chainable reference stack; every image independently chooses semantic or native transport.
- **Text Encode H3 Edit** — turns a source image, edit instruction, and optional ordered guides into H3 conditioning plus a short, valid H3 temporal packet.
- **Decode H3 Edit to One Image** — decodes the packet, scores its candidates, and returns one stable high-quality frame.

The graph still produces exactly one image. By default, it samples the same short 5-frame context used by H3 Studio because H3 is a video model and a literal one-token latent leaves no temporal context around the edit. The old true-one-frame path remains available as an explicitly experimental, low-quality option.

## Semantic versus native references

`<Picture 1>` is always the image being edited. It goes through both Qwen and the H3 VAE and is attached as the frame-zero keyframe.

Every optional guide has a transport selector:

| Mode | Qwen visual tokens | H3 VAE latent | Conditioning payload | Intended use |
|---|---:|---:|---|---|
| `semantic (Qwen only)` | yes | no | no `minimax_refs` | Transfer the idea of glasses, an object category, a material, palette, or attribute |
| `native (Qwen + VAE ref)` | yes | yes | one `minimax_refs` block | Stronger low-level visual reference; experimental when combined with the source keyframe |
| `none (source only)` | source only | source only | keyframe only | Ordinary one-image edit |

For “add these glasses to this woman,” connect the woman to `source_image`, the glasses image to `reference_image`, and use `semantic (Qwen only)`. That direct socket is the convenient `<Picture 2>` path. A semantic guide can use an equivalent-square Qwen budget from 256 to 3584 pixels without allocating another VAE latent.

## Scalable ordered references

For more guides, chain **Add H3 Edit Reference** nodes and connect the final `references` output to `reference_stack` on **Text Encode H3 Edit**. The stack has no fixed reference count. An IMAGE batch is expanded into separate ordered references, and additional builder nodes can be chained within normal ComfyUI graph limits. In practice, Qwen context length, conditioning time, RAM, and VRAM still limit useful reference counts.

Ordering is deterministic:

1. `source_image` is always `<Picture 1>`.
2. A connected direct `reference_image` is `<Picture 2>` unless its mode is `none`.
3. Stack entries follow in chain and batch order as the next `<Picture N>` values.

Each builder declares its own `semantic (Qwen only)` or `native (Qwen + VAE ref)` transport, semantic resolution, and native resize policy. Semantic and native entries can be used simultaneously in the same stack—for example, semantic glasses as `<Picture 2>`, a native texture as `<Picture 3>`, and a semantic hairstyle as `<Picture 4>`. Only `<Picture 3>` is VAE-encoded in that example. Mixed native guide latents remain experimental; an all-semantic guide stack uses only the source VAE anchor.

The native mode is intentionally labeled experimental. ComfyUI's H3 packed layout can carry keyframes and `minimax_refs` together, but the released FL2VA and REF2VA weights were trained for different task presentations. Runtime compatibility does not guarantee that every checkpoint responds well to the mixed transport.

## Quality profiles

The quality selector changes only the hidden context; every profile still outputs one image.

| Profile | Candidate frames | Video latent tokens | Use |
|---|---:|---:|---|
| `recommended` | 5 | 2 | H3 Studio's default short-context balance |
| `extended` | 9 | 3 | More candidates at moderately higher sampling cost |
| `high` | 13 | 4 | Additional temporal context for difficult edits |
| `maximum` | 20 | 7 | Slowest option; its natural 22-frame decode is cropped to 20 candidates |
| `experimental` | 1 | 1 | Literal one-frame path retained for comparison; often poor quality |

These are Studio-style compact still-image profiles. The prompt asks for a locked camera and an immediately completed edit held unchanged. The decoder then evaluates sharpness, contrast, exposure, and temporal stability at reduced resolution and returns only the highest-scoring still.

This pack does not freeze temporal RoPE. The quality paths intentionally use H3's natural temporal coordinates so the model can evolve away from the frame-zero source anchor. In true-one-frame mode there is only one target temporal coordinate, so a RoPE freeze would be a no-op.

## Requirements

- A current ComfyUI build with native MiniMax H3 support.
- MiniMax H3 FL2VA diffusion model.
- MiniMax H3 Qwen3-VL text encoder.
- MiniMax H3 video VAE.

The semantic path still uses the H3 VAE for the source photo. “Qwen only” applies to the optional guide, not the image being edited.

## Install

Place the folder in `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ethanfel/ComfyUI-MiniMax-H3-Edit
```

Restart ComfyUI. The nodes appear under `MiniMax H3/Edit`.

## Basic graph

1. Load the H3 diffusion model, Qwen text encoder, and video VAE with standard ComfyUI loaders.
2. Connect the photo being edited to `source_image` on **Text Encode H3 Edit**.
3. For one guide, use its direct `reference_image` socket. For multiple guides, chain **Add H3 Edit Reference** nodes into `reference_stack`.
4. Declare semantic or native transport for each guide and refer to it with its ordered `<Picture N>` tag.
5. Use `ModelSamplingMiniMaxH3`, `BasicGuider`, `RandomNoise`, a sampler and scheduler, and `SamplerCustomAdvanced` as in the normal native H3 graph.
6. Connect the sampled latent to **Decode H3 Edit to One Image**, then preview or save its image output.

The included [mixed-reference example workflow](example_workflows/H3_Edit_Mixed_References.json) is configured for a single-image FL2VA edit using the recommended hidden 5-frame context. It uses semantic glasses as `<Picture 2>` and a native wardrobe/material guide as `<Picture 3>` in the same generation. Replace its model filenames and input images with files available in your ComfyUI installation.

## Prompt behavior

`edit instruction` wraps your request with a short preservation contract and explicitly scopes every `<Picture N>` as semantic or native. `use prompt verbatim` leaves the text unchanged; the Qwen picture blocks are still prepended in source-then-guide order.

Example instruction:

```text
Add the black acetate glasses from <Picture 2> to the woman. Fit them naturally to her face with realistic temples, reflections, and contact shadows. Change nothing else.
```

## License and provenance

MIT. This is an independent implementation built against ComfyUI's public MiniMax H3 conditioning contracts. The experimental one-token fallback was interoperability-checked against [ComfyUI-MiniMaxH3-SingleFrame](https://github.com/tori29umai0123/ComfyUI-MiniMaxH3-SingleFrame), but this pack does not copy its unlicensed code.
