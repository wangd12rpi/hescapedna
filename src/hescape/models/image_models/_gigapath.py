# hescape/models/image_models/_gigapath.py
"""GigaPath tile/slide encoder integration built on the upstream pipeline utilities.

Portions of this implementation draw from
``related_proj/prov-gigapath/gigapath/pipeline.py`` in order to keep the behaviour
aligned with the reference training code while avoiding heavy preprocessing
dependencies inside the training loop.
"""
from __future__ import annotations
from gigapath import pipeline

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _build_gigapath_model(
):
    tile_encoder, slide_encoder = pipeline.load_tile_slide_encoder(global_pool=True)
    return tile_encoder, slide_encoder


__all__ = ["_build_gigapath_model"]
