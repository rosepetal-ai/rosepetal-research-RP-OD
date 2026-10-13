# Where RP-OD comes from

RP-OD is a real-time DETR: closest to **D-FINE**, which itself builds on **RT-DETR**, with the loss of **DEIM**, YOLO's
mosaic and dataset format, and a few changes of its own for production. This page says which part comes from where.

| Part | Taken from | In RP-OD |
|---|---|---|
| Set prediction: Hungarian matching, one prediction per object, no NMS | DETR (ECCV 2020) | same |
| Deformable attention: each query samples a few points around its box, on every level | Deformable DETR (ICLR 2021) | same, 4 points per level |
| Hybrid encoder: attention only on the coarsest level (AIFI), then FPN + PAN fusion (CCFF) with RepVGG blocks | RT-DETR (CVPR 2024) | same |
| Query selection: a dense head scores every position and the top 300 become the decoder's queries | RT-DETR | same, and the dense head is supervised on every position, not only on the selected ones (ours) |
| Denoising queries: noisy copies of the ground truth the decoder learns to fix | DN-DETR (CVPR 2022), DINO (ICLR 2023) | same |
| Boxes as distributions over each edge, refined layer by layer (FDR); the last layer teaches the earlier ones (GO-LSD: FGL + DDF losses) | D-FINE (ICLR 2025); edge distributions from GFL / DFL (NeurIPS 2020) | same |
| HGNetv2 backbones (ImageNet SSLD weights) and the S / M / L balance: B0 / B2 / B4 with 3 / 4 / 6 decoder layers | D-FINE (backbone from PaddleClas) | same sizes; the `m` neck is narrower |
| Classification loss MAL: full weight on every match, target IoU^1.5 | DEIM (CVPR 2025) | same |
| Mosaic early in training | YOLO; DEIM's dense one-to-one | first quarter of the epochs |
| Dataset format and the one-object API (`train`, `val`, `predict`, `export`) | YOLO / Ultralytics | same style |

What RP-OD does differently:

| | Usual (RT-DETR, D-FINE, DEIM) | RP-OD |
|---|---|---|
| Fitting the image to the input | stretched to a square | letterboxed, aspect ratio kept, exactly as the production pre-processing |
| Multi-scale training | square sizes from 480 to 800 | height and width drawn separately: 320–960 in the first quarter, 480–800 until the last sixth, then the training size |
| Learning rate | constant after the warm-up (RT-DETR, D-FINE); flat then cosine down to half (DEIM) | flat for the first half, cosine decay to 5 % over the second |
| Dense head | supervised on the 300 selected queries | supervised on every position |
| Not adopted from D-FINE | GELAN neck blocks, decoder attention without value / output projections, gated fusion, location quality estimator, 3 / 6 / 3 points per level | — |
| Delivery | research code with YAML configs | a library: three sizes, dataset checks, default augmentation (replaceable), ONNX + config export for serving (TensorRT, ONNX Runtime, Triton) |

The diagram `docs/architecture.svg` shows the same lineage next to each block.
