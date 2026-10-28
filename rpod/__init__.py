# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""RP-OD: real-time object detection. A dense head proposes, a DETR-style decoder refines. No NMS."""
from .model import RPOD

__version__ = '0.2.1'
