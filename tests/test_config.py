# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
import pytest

from rpod import RPOD
from rpod.cfg.schema import SIZES, Architecture, ConfigError, derive, from_dict, size_of, to_dict


def test_three_sizes_and_nothing_else():
    assert set(SIZES) == {'s', 'm', 'l'} and RPOD().cfg == SIZES['m']
    assert [a.backbone for a in SIZES.values()] == ['hgnetv2-b0', 'hgnetv2-b2', 'hgnetv2-b4']   # grows with the size
    with pytest.raises(ValueError, match="give a size"):
        RPOD('xl')
    with pytest.raises(TypeError):
        RPOD('s', backbone='hgnetv2-b4')


def test_checkpoint_config_round_trip_and_the_knobs_of_older_checkpoints():
    for arch in SIZES.values():
        assert from_dict(to_dict(arch)) == arch
    assert from_dict({'backbone': 'hgnetv2-b0', 'neck': {'size': 's'}, 'decoder': {'layers': 3}}) == SIZES['s']
    assert size_of(from_dict({'backbone': 'hgnetv2-b0', 'neck': {'size': 'l'}, 'decoder': {'layers': 6}})) is None
    with pytest.raises(ConfigError, match='not part of this version'):
        from_dict({'backbone': 'resnet18', 'neck_blocks': 1, 'decoder_layers': 3})


def test_derived_settings():
    d = derive(SIZES['l'], num_classes=4, num_queries=100)
    assert d['levels'] == [3, 4, 5] and d['anchor_size'] == 0.05
    assert d['fusion_blocks'] == 3 and d['fusion_expansion'] == 1.0 and d['dec_layers'] == 6
    assert d['num_queries'] == 300   # never below the minimum
    assert derive(Architecture('hgnetv2-b0', 2, 5), 1)['fusion_blocks'] == 2
