import pytest
import yaml

from rpod.cfg.schema import ConfigError, ModelConfig, derive, guide, load_config, template, to_dict


def test_defaults():
    c = load_config()
    assert (c.backbone, c.neck.size, c.neck.small_objects, c.decoder.layers) == ('hgnetv2-b2', 'm', False, 3)


def test_yaml_then_arguments_win(tmp_path):
    p = tmp_path / 'm.yaml'
    p.write_text('backbone: hgnetv2-b4\nneck: {size: l}\ndecoder: {layers: 5}\n')
    c = load_config(p, decoder_layers=6, small_objects=True)
    assert (c.backbone, c.neck.size, c.neck.small_objects, c.decoder.layers) == ('hgnetv2-b4', 'l', True, 6)


@pytest.mark.parametrize('args,message', [
    ({'backbone': 'resnet50'}, 'not one of'),
    ({'neck_size': 'xl'}, 'not one of'),
    ({'decoder_layers': 8}, 'not one of'),
    ({'small_objects': 'yes'}, 'expected bool'),
    ({'decoder_layer': 4}, "did you mean 'decoder_layers'"),
])
def test_bad_arguments_explain_the_fix(args, message):
    with pytest.raises(ConfigError, match=message):
        load_config(**args)


def test_unknown_yaml_key_suggests(tmp_path):
    p = tmp_path / 'm.yaml'
    p.write_text('neck: {sise: s}\n')
    with pytest.raises(ConfigError, match="did you mean 'neck.size'"):
        load_config(p)


def test_template_is_the_default_config_and_guide_lists_every_knob(tmp_path):
    p = tmp_path / 't.yaml'
    p.write_text(template())
    assert to_dict(load_config(p)) == to_dict(ModelConfig())
    for key in ('backbone', 'neck.size', 'neck.small_objects', 'decoder.layers'):
        assert f'`{key}`' in guide()


def test_derived_settings():
    d = derive(load_config(backbone='hgnetv2-b4', neck_size='s', small_objects=True), num_classes=4, num_queries=100)
    assert d['levels'] == [2, 3, 4, 5] and d['anchor_size'] == 0.025
    assert d['fusion_blocks'] == 1 and d['fusion_expansion'] == 1.0
    assert d['num_queries'] == 300   # never below the minimum
    assert derive(load_config(), 1)['levels'] == [3, 4, 5]
