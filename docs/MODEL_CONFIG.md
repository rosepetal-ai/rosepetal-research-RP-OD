# Model configuration

The model YAML only describes the architecture: three knobs. Classes come from the dataset and the input size
is chosen when training. Arguments passed to `RPOD(...)` win over the YAML. Everything else is fixed or
computed automatically. This page is generated from `rpod/cfg/schema.py`.

| Knob | Default | Values | Why it is exposed | Explanation |
|---|---|---|---|---|
| `backbone` | `hgnetv2-b2` | hgnetv2-b0, hgnetv2-b2, hgnetv2-b4, hgnetv2-b5, resnet18 | the main lever between capacity and speed | The part of the model that looks at the image and extracts its features. It is the heaviest part: a bigger one tells hard textures and look-alike classes apart better, but it is slower and needs more images to pay off. Start with hgnetv2-b2 and go up only if the model confuses classes or fails on difficult backgrounds. hgnetv2-b0 is the light one, hgnetv2-b4 the precise one, hgnetv2-b5 the maximum, resnet18 a well-known reference. |
| `neck.size` | `m` | s, m, l | speed versus precision when mixing scales | The neck combines what the backbone sees at different scales: fine detail and the big picture. A bigger neck mixes the scales better, which helps when one image holds objects of very different sizes or the background is complex. s is faster; l is more precise. |
| `decoder.layers` | `3` | 3, 4, 5, 6 | tighter boxes for little cost | How many times the model reviews and corrects each detection. More passes give boxes that fit the object edges more tightly, useful when you need to locate or measure precisely. Each extra pass adds little compute, a bit of latency. More than 6 rarely helps. |
