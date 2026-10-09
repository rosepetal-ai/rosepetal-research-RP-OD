import json

import numpy as np
import onnx
import pytest
import torch
from PIL import Image

from rpod import RPOD
from rpod.cfg.schema import SIZES, Architecture, derive
from rpod.engine.predictor import postprocess
from rpod.nn import Detector
from rpod.utils.checkpoint import load_weights

from .conftest import NAMES, make_net


def test_checkpoint_roundtrip(trained_ckpt):
    m = RPOD(trained_ckpt)
    assert m.names == NAMES and m.imgsz == [256, 320] and m.threshold == 0.01
    assert 'size s' in repr(m) and 'trained, 3 classes' in repr(m)


def test_untrained_model_refuses_to_predict():
    with pytest.raises(RuntimeError, match='not trained'):
        RPOD().predict(np.zeros((8, 8, 3), np.uint8))


def test_info_reports_blocks():
    s = RPOD('s').info(imgsz=[250, 330], verbose=False)
    assert s['imgsz'] == [256, 352]          # rounded up to multiples of 32
    assert s['total'] == s['backbone'] + s['neck'] + s['query selection'] + s['decoder'] and s['gflops'] > 0


def test_predict_inputs_and_coordinates(trained_ckpt, tmp_path):
    m = RPOD(trained_ckpt)
    img = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    Image.fromarray(img).save(tmp_path / 'a.png')
    r = m.predict(img, color='BGR', device='cpu')[0]
    assert len(r) > 0 and r.orig_shape == (480, 640)
    b = r.boxes
    assert (b.xyxy[:, [0, 2]] <= 640).all() and (b.xyxy[:, [1, 3]] <= 480).all() and (b.xyxy >= 0).all()
    assert torch.allclose(b.xyxyn * torch.tensor([640, 480, 640, 480]), b.xyxy, atol=1e-3)
    assert len(r) <= 100 and (b.conf > 0.01).all()        # default conf = checkpoint threshold
    assert len(m.predict([tmp_path / 'a.png', img], device='cpu')) == 2
    assert len(m.predict(tmp_path, device='cpu')) == 1
    assert r.plot().shape == (480, 640, 3)


def test_postprocess_undoes_the_letterbox():
    # 1920x1080 image letterboxed into 640x640: scale 1/3, 140 px of padding on top
    geom = (1 / 3, 360, 640, 140, 0)
    boxes = torch.tensor([[0.5, 0.5, 0.25, 0.1]])          # center of the canvas, 160 x 64 px
    logits = torch.tensor([[5.0, -5.0]])
    xyxy, score, cls = postprocess(logits, boxes, geom, (1080, 1920), [640, 640], 0.5, 100)
    assert cls.tolist() == [0]
    assert torch.allclose(xyxy, torch.tensor([[720.0, 444.0, 1200.0, 636.0]]), atol=0.05)


def test_export_contract(trained_ckpt):
    out = RPOD(trained_ckpt).export()
    model = onnx.load(str(out / 'model.onnx'))
    inp = model.graph.input[0]
    assert [i.name for i in model.graph.input] == ['images']
    assert [d.dim_value for d in inp.type.tensor_type.shape.dim] == [1, 3, 256, 320]
    assert [o.name for o in model.graph.output] == ['dets', 'labels']
    cfg = json.loads((out / 'config.json').read_text())
    assert cfg['common']['image_shape'] == {'height': 256, 'width': 320, 'channels': 3}
    assert cfg['task_specific']['classes'] == {'0': 'scratch', '1': 'screw', '2': 'label'}
    assert cfg['task_specific']['confidence'] == 0.01 and cfg['common']['model_name'] == 'rpod-s'


def test_a_pretrained_model_initializes_the_same_size_for_other_classes():
    src = make_net(num_classes=80)                                           # e.g. the COCO pretraining of s
    dst = Detector(derive(SIZES['s'], 3), pretrained_backbone=False)
    report = load_weights(dst, src.state_dict())
    assert report['loaded_fraction'] > 0.95
    assert {'decoder.enc_score_head', 'decoder.dec_score_head', 'decoder.denoising_class_embed'} == set(
        report['reinitialized'])                                         # only the class heads start fresh


def test_decoder_truncation_is_exact():
    """Each decoder layer is trained as a full detector: keeping the first N gives exactly layer N's predictions."""
    full = make_net(arch=Architecture('hgnetv2-b0', 1, 6))
    short = make_net(arch=Architecture('hgnetv2-b0', 1, 3))
    load_weights(short, full.state_dict())
    x = torch.rand(1, 3, 128, 160)
    with torch.no_grad():
        out_short = short(x)
        full.decoder.num_layers = 3
        out_full = full(x)
    assert torch.equal(out_short['pred_logits'], out_full['pred_logits'])
    assert torch.equal(out_short['pred_boxes'], out_full['pred_boxes'])


def test_pretrained_weights_download_once_and_check_their_checksum(tmp_path, monkeypatch, trained_ckpt):
    import hashlib
    from rpod import weights as W
    release = tmp_path / 'repo' / 'releases' / 'download' / 'weights-v1'
    release.mkdir(parents=True)
    (release / 'rpod-s-coco.pt').write_bytes(trained_ckpt.read_bytes())
    sha = hashlib.sha256(trained_ckpt.read_bytes()).hexdigest()
    monkeypatch.setattr(W, 'REPO', (tmp_path / 'repo').as_uri())
    monkeypatch.setenv('RPOD_CACHE', str(tmp_path / 'cache'))
    monkeypatch.setattr(W, 'PRETRAINED', {'s': ('weights-v1', 'rpod-s-coco.pt', sha)})
    assert W.pretrained_weights('m') is None                                # nothing published for m
    path = W.pretrained_weights('s', log=lambda m: None)
    assert path == tmp_path / 'cache' / 'weights-v1' / 'rpod-s-coco.pt' and RPOD(path).names == NAMES
    (release / 'rpod-s-coco.pt').unlink()
    assert W.pretrained_weights('s') == path                                # cached: no second download
    monkeypatch.setattr(W, 'PRETRAINED', {'s': ('weights-v2', 'rpod-s-coco.pt', '0' * 64)})
    (tmp_path / 'repo' / 'releases' / 'download' / 'weights-v2').mkdir()
    (tmp_path / 'repo' / 'releases' / 'download' / 'weights-v2' / 'rpod-s-coco.pt').write_bytes(b'corrupted')
    with pytest.raises(RuntimeError, match='checksum'):
        W.pretrained_weights('s', log=lambda m: None)
    assert not list((tmp_path / 'cache' / 'weights-v2').iterdir())         # nothing half-written left behind
