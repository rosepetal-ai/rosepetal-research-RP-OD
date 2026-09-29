"""Model configuration: the few knobs a user may change, each with its explanation, and everything derived from them.

A model YAML only describes the architecture. Classes come from the dataset and the input size is chosen when
training, so both live in the checkpoint, not here.

    backbone: hgnetv2-b2
    neck:
      size: m
      small_objects: false
    decoder:
      layers: 3
"""
import copy
import difflib
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml

MODELS_DIR = Path(__file__).parent / 'models'


class ConfigError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ fixed choices
# Backbones: timm model with ImageNet weights, and the neck width that goes with it (fixed per backbone, so a
# single pretraining per backbone later covers every neck size).
BACKBONES = {
    'hgnetv2-b0': dict(timm='hgnetv2_b0.ssld_stage1_in22k_in1k', expansion=0.5),
    'hgnetv2-b2': dict(timm='hgnetv2_b2.ssld_stage1_in22k_in1k', expansion=0.5),
    'hgnetv2-b4': dict(timm='hgnetv2_b4.ssld_stage1_in22k_in1k', expansion=1.0),
    'hgnetv2-b5': dict(timm='hgnetv2_b5.ssld_stage1_in22k_in1k', expansion=1.0),
    'resnet18': dict(timm='resnet18.tv_in1k', expansion=0.5),
}
NECK_BLOCKS = {'s': 1, 'm': 2, 'l': 3}   # neck size changes depth only, never width
HIDDEN_DIM = 256                        # every feature map and query; fixed so weights stay reusable
ATTENTION = dict(layers=1, heads=8, ffn_dim=1024)
DECODER = dict(heads=8, points=4, ffn_dim=1024)
DENOISING = dict(num=100, label_noise=0.5, box_noise=1.0)
MIN_QUERIES = 300


def knob(default, explanation, choices=None, why=None):
    meta = dict(explanation=explanation, choices=choices, why=why)
    if isinstance(default, (list, dict)):
        return field(default_factory=lambda d=default: copy.deepcopy(d), metadata=meta)
    return field(default=default, metadata=meta)


def section(cls, explanation):
    return field(default_factory=cls, metadata=dict(explanation=explanation, section=True))


# ------------------------------------------------------------------------------------------------ user knobs
@dataclass
class NeckConfig:
    size: str = knob(
        'm', 'The neck combines what the backbone sees at different scales: fine detail and the big picture. A bigger '
        'neck mixes the scales better, which helps when one image holds objects of very different sizes or the '
        'background is complex. s is faster; l is more precise.',
        choices=['s', 'm', 'l'], why='speed versus precision when mixing scales')
    small_objects: bool = knob(
        False, 'Adds a grid twice as fine (each cell covers 4 × 4 pixels instead of 8 × 8). Turn it on only if the '
        'smallest objects you need to detect are under about 8 pixels at the training size. Before turning it on, '
        'try a larger input size or cropping the region of interest: it often works as well or better. Roughly '
        'doubles the compute.',
        why='better detection of objects a few pixels wide, at a high cost')


@dataclass
class DecoderConfig:
    layers: int = knob(
        3, 'How many times the model reviews and corrects each detection. More passes give boxes that fit the object '
        'edges more tightly, useful when you need to locate or measure precisely. Each extra pass adds little '
        'compute, a bit of latency. More than 6 rarely helps.',
        choices=[3, 4, 5, 6], why='tighter boxes for little cost')


@dataclass
class ModelConfig:
    backbone: str = knob(
        'hgnetv2-b2', 'The part of the model that looks at the image and extracts its features. It is the heaviest '
        'part: a bigger one tells hard textures and look-alike classes apart better, but it is slower and needs more '
        'images to pay off. Start with hgnetv2-b2 and go up only if the model confuses classes or fails on difficult '
        'backgrounds. hgnetv2-b0 is the light one, hgnetv2-b4 the precise one, hgnetv2-b5 the maximum, resnet18 a '
        'well-known reference.',
        choices=list(BACKBONES), why='the main lever between capacity and speed')
    neck: NeckConfig = section(NeckConfig, 'Neck: fuses the feature maps of the backbone.')
    decoder: DecoderConfig = section(DecoderConfig, 'Decoder: refines every detection, layer by layer.')


# ------------------------------------------------------------------------------------------------ loading
ARG_KEYS = {'backbone': 'backbone', 'neck_size': 'neck.size', 'small_objects': 'neck.small_objects',
            'decoder_layers': 'decoder.layers'}


def _names(obj):
    return [f.name for f in fields(obj)]


def _unknown(key, path, names):
    hint = difflib.get_close_matches(key, names, n=1)
    msg = f"unknown key '{path}{key}'" + (f" (did you mean '{path}{hint[0]}'?)" if hint else '')
    return ConfigError(f"{msg}; valid keys here: {', '.join(names)}")


def _merge(obj, data, path=''):
    for k, v in data.items():
        if k not in _names(obj):
            raise _unknown(k, path, _names(obj))
        cur = getattr(obj, k)
        if is_dataclass(cur):
            if not isinstance(v, dict):
                raise ConfigError(f"'{path}{k}' is a section: give it keys, not a value")
            _merge(cur, v, f'{path}{k}.')
        else:
            setattr(obj, k, v)


def _set(cfg, dotted, value):
    *parents, last = dotted.split('.')
    obj = cfg
    for p in parents:
        obj = getattr(obj, p)
    setattr(obj, last, value)


def load_config(source=None, **args):
    """Defaults <- YAML file (optional) <- arguments. Arguments: backbone, neck_size, small_objects, decoder_layers."""
    cfg = ModelConfig()
    if source is not None:
        path = Path(source)
        if not path.exists() and (MODELS_DIR / path.name).exists():
            path = MODELS_DIR / path.name
        if not path.exists():
            raise ConfigError(f"model config '{source}' not found")
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for k, v in args.items():
        if k not in ARG_KEYS:
            hint = difflib.get_close_matches(k, list(ARG_KEYS), n=1)
            raise ConfigError(f"unknown argument '{k}'" + (f" (did you mean '{hint[0]}'?)" if hint else '')
                              + f"; valid arguments: {', '.join(ARG_KEYS)}")
        _set(cfg, ARG_KEYS[k], v)
    validate(cfg)
    return cfg


def to_dict(obj):
    return {f.name: to_dict(getattr(obj, f.name)) if is_dataclass(getattr(obj, f.name)) else getattr(obj, f.name)
            for f in fields(obj)}


def from_dict(data):
    cfg = ModelConfig()
    _merge(cfg, data)
    validate(cfg)
    return cfg


def _walk(obj, path=''):
    for f in fields(obj):
        v = getattr(obj, f.name)
        if is_dataclass(v):
            yield from _walk(v, f'{path}{f.name}.')
        else:
            yield f'{path}{f.name}', f, v


def validate(cfg):
    errors = []
    for key, f, v in _walk(cfg):
        default = f.default if f.default is not MISSING else f.default_factory()
        expected = type(default)
        if not isinstance(v, expected) or (expected is int and isinstance(v, bool)):
            errors.append(f'{key}: expected {expected.__name__}, got {v!r}')
        elif f.metadata.get('choices') and v not in f.metadata['choices']:
            errors.append(f"{key}: {v!r} is not one of {', '.join(map(str, f.metadata['choices']))}")
    if errors:
        raise ConfigError('invalid model config:\n  - ' + '\n  - '.join(errors))


# ------------------------------------------------------------------------------------------------ derived
def derive(cfg, num_classes, num_queries=MIN_QUERIES):
    """Everything the network needs, computed from the user knobs. Nothing here is user-facing."""
    levels = [2, 3, 4, 5] if cfg.neck.small_objects else [3, 4, 5]
    return dict(
        timm_name=BACKBONES[cfg.backbone]['timm'], levels=levels, hidden_dim=HIDDEN_DIM,
        attn_layers=ATTENTION['layers'], attn_heads=ATTENTION['heads'], attn_ffn=ATTENTION['ffn_dim'],
        fusion_blocks=NECK_BLOCKS[cfg.neck.size], fusion_expansion=BACKBONES[cfg.backbone]['expansion'],
        num_classes=num_classes, num_queries=max(MIN_QUERIES, num_queries),
        dec_layers=cfg.decoder.layers, dec_heads=DECODER['heads'], dec_points=DECODER['points'],
        dec_ffn=DECODER['ffn_dim'], anchor_size=0.05 * 2.0 ** (levels[0] - 3),
        num_denoising=DENOISING['num'], label_noise=DENOISING['label_noise'], box_noise=DENOISING['box_noise'])


def queries_for(max_objects_per_image):
    """Candidates the decoder refines: at least 300, and at least 5x the busiest image of the dataset."""
    return max(MIN_QUERIES, 5 * int(max_objects_per_image))


# ------------------------------------------------------------------------------------------------ docs
def _wrap(text, width, prefix):
    lines, cur = [], ''
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f'{cur} {word}'.strip()
    return [prefix + line for line in lines + ([cur] if cur else [])]


def template():
    """The default model YAML, every knob with its explanation."""
    out = ['# RP-OD model configuration. Only the architecture lives here: classes come from the dataset and the',
           '# input size is chosen when training. Any key can also be passed as an argument, which wins over this file:',
           "#   RPOD('rpod.yaml', backbone='hgnetv2-b4', neck_size='l', small_objects=True, decoder_layers=4)", '']

    def emit(obj, indent):
        pad = '  ' * indent
        for f in fields(obj):
            m, v = f.metadata, getattr(obj, f.name)
            out.extend(_wrap(m['explanation'], 100, f'{pad}# '))
            if m.get('section'):
                out.append(f'{pad}{f.name}:')
                emit(v, indent + 1)
                continue
            if m.get('choices'):
                out.append(f"{pad}#   choices: {' | '.join(map(str, m['choices']))}")
            out.append(f'{pad}{f.name}: {yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()}')
            out.append('')
    emit(ModelConfig(), 0)
    return '\n'.join(out).rstrip() + '\n'


def guide():
    rows = []
    for key, f, v in _walk(ModelConfig()):
        choices = ', '.join(map(str, f.metadata['choices'])) if f.metadata.get('choices') else 'true, false'
        rows.append(f"| `{key}` | `{v}` | {choices} | {f.metadata['why']} | {f.metadata['explanation']} |")
    return '\n'.join([
        '# Model configuration', '',
        'The model YAML only describes the architecture: four knobs. Classes come from the dataset and the input size',
        'is chosen when training. Arguments passed to `RPOD(...)` win over the YAML. Everything else is fixed or',
        'computed automatically. This page is generated from `rpod/cfg/schema.py`.', '',
        '| Knob | Default | Values | Why it is exposed | Explanation |', '|---|---|---|---|---|', *rows, ''])
