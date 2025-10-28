# hescape/models/image_models/_gigapath.py
"""GigaPath tile/slide encoder integration built on the upstream pipeline utilities.

Portions of this implementation draw from
``related_proj/prov-gigapath/gigapath/pipeline.py`` in order to keep the behaviour
aligned with the reference training code while avoiding heavy preprocessing
dependencies inside the training loop.
"""
from __future__ import annotations
from typing import Tuple

from gigapath import pipeline


def _build_gigapath_model() -> Tuple[object, object]:
    """Construct the pretrained GigaPath tile and slide encoders."""
    return pipeline.load_tile_slide_encoder(global_pool=True)


__all__ = ["_build_gigapath_model"]
