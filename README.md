# ComfyUI MiniMax H3 Edit

A deliberately small ComfyUI node pack for **one-frame MiniMax H3 photo editing**. It has no Director, hidden state, prompt analyzer, model loader, sampler wrapper, or custom frontend.

The pack adds two nodes:

- **Text Encode H3 Edit** — turns a source image, edit instruction, and optional guide into H3 conditioning plus a true one-frame latent.
- **Decode H3 Single Frame** — decodes that one-frame latent directly with the native H3 video VAE.

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

This pack does not add a temporal RoPE-freeze node. A true one-frame latent has one temporal token, so every spatial row of the target already shares the same temporal coordinate; there is no target-frame time variation left to freeze. That kind of patch can still matter for multi-token interpolation, which is outside this pack's scope.

## Requirements

- A current ComfyUI build with native MiniMax H3 support, including one-token H3 VAE encode/decode.
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
5. Connect the sampled latent to **Decode H3 Single Frame**, then preview or save its image output.

The example workflow is configured for a one-frame FL2VA edit. Replace its model filenames and input images with files available in your ComfyUI installation.

## Prompt behavior

`edit instruction` wraps your request with a short preservation contract and explicitly scopes `<Picture 2>` as semantic or native. `use prompt verbatim` leaves the text unchanged; the Qwen picture blocks are still prepended in source-then-guide order.

Example instruction:

```text
Add the black acetate glasses from <Picture 2> to the woman. Fit them naturally to her face with realistic temples, reflections, and contact shadows. Change nothing else.
```

## License and provenance

MIT. This is an independent implementation built against ComfyUI's public MiniMax H3 conditioning contracts. The experimental one-token approach was interoperability-checked against [ComfyUI-MiniMaxH3-SingleFrame](https://github.com/tori29umai0123/ComfyUI-MiniMaxH3-SingleFrame), but this pack does not copy its unlicensed code.
