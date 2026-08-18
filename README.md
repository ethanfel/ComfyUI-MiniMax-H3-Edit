# ComfyUI MiniMax H3 Edit

A deliberately small ComfyUI node pack for **single-image MiniMax H3 photo editing**. It has no Director, hidden state, prompt analyzer, model loader, sampler wrapper, or custom frontend.

The pack adds two nodes:

- **Text Encode H3 Edit** — turns a source image, edit instruction, and optional guide into H3 conditioning plus a short, valid H3 temporal packet.
- **Decode H3 Edit to One Image** — decodes the packet and returns only its completed final frame.

The graph still produces exactly one image. By default, it samples a hidden 22-frame H3 packet because H3 is a video model and a literal one-token latent leaves no post-anchor temporal space for the edit to resolve. The old true-one-frame path remains available as an explicitly experimental, low-quality option.

## Semantic versus native Picture 2

`<Picture 1>` is always the image being edited. It goes through both Qwen and the H3 VAE and is attached as the frame-zero keyframe.

The optional `<Picture 2>` guide has a selector:

| Mode | Qwen visual tokens | H3 VAE latent | Conditioning payload | Intended use |
|---|---:|---:|---|---|
| `semantic (Qwen only)` | yes | no | no `minimax_refs` | Transfer the idea of glasses, an object category, a material, palette, or attribute |
| `native (Qwen + VAE ref)` | yes | yes | one `minimax_refs` block | Stronger low-level visual reference; experimental when combined with the source keyframe |
| `none (source only)` | source only | source only | keyframe only | Ordinary one-image edit |

For “add these glasses to this woman,” connect the woman to `source_image`, the glasses image to `reference_image`, and use `semantic (Qwen only)`. The guide can use an equivalent-square Qwen budget from 256 to 3584 pixels without allocating a second VAE latent.

The native mode is intentionally labeled experimental. ComfyUI's H3 packed layout can carry keyframes and `minimax_refs` together, but the released FL2VA and REF2VA weights were trained for different task presentations. Runtime compatibility does not guarantee that every checkpoint responds well to the mixed transport.

## Quality profiles

The quality selector changes only the hidden context; every profile still outputs one image.

| Profile | Internal frames | Video latent tokens | Use |
|---|---:|---:|---|
| `recommended` | 22 | 7 | Default balance for giving the edit time to settle |
| `fast` | 5 | 2 | Cheapest model-native short packet |
| `high` | 39 | 12 | More temporal context at higher sampling cost |
| `maximum` | 56 | 17 | Most context offered by this pack; watch VRAM and drift |
| `experimental` | 1 | 1 | Literal one-frame path retained for comparison; often poor quality |

The multi-frame choices follow H3's native `17k + 5` frame grid. The prompt asks for a locked camera and an immediately completed edit held unchanged, and the decoder selects only the final still.

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
2. Connect two `Load Image` nodes to `source_image` and `reference_image` on **Text Encode H3 Edit**.
3. Choose `semantic (Qwen only)` and describe the specific edit.
4. Use `ModelSamplingMiniMaxH3`, `BasicGuider`, `RandomNoise`, a sampler and scheduler, and `SamplerCustomAdvanced` as in the normal native H3 graph.
5. Connect the sampled latent to **Decode H3 Edit to One Image**, then preview or save its image output.

The example workflow is configured for a single-image FL2VA edit using the recommended hidden 22-frame context. Replace its model filenames and input images with files available in your ComfyUI installation.

## Prompt behavior

`edit instruction` wraps your request with a short preservation contract and explicitly scopes `<Picture 2>` as semantic or native. `use prompt verbatim` leaves the text unchanged; the Qwen picture blocks are still prepended in source-then-guide order.

Example instruction:

```text
Add the black acetate glasses from <Picture 2> to the woman. Fit them naturally to her face with realistic temples, reflections, and contact shadows. Change nothing else.
```

## License and provenance

MIT. This is an independent implementation built against ComfyUI's public MiniMax H3 conditioning contracts. The experimental one-token fallback was interoperability-checked against [ComfyUI-MiniMaxH3-SingleFrame](https://github.com/tori29umai0123/ComfyUI-MiniMaxH3-SingleFrame), but this pack does not copy its unlicensed code.
