# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import math

from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()

def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)

def pad(data, padded_shape=(224, 224), pad_constant_value=10):
    matrix = np.array(data)

    vertical_pad = (padded_shape[0] - matrix.shape[0]) / 2.0
    horizontal_pad = (padded_shape[1] - matrix.shape[1]) / 2.0

    matrix_padded = np.pad(
        matrix,
        pad_width=(
            (math.floor(vertical_pad), math.ceil(vertical_pad)),
            (math.floor(horizontal_pad), math.ceil(horizontal_pad))
        ),
        constant_values=pad_constant_value
    )

    return matrix_padded


def unpad(data, unpad_constant_value=10):
    matrix = np.array(data)

    unpad_mask = (matrix != unpad_constant_value)
    dim1 = np.any(unpad_mask, axis=1).sum()
    matrix_unpadded = matrix[unpad_mask].reshape(dim1, -1)

    return matrix_unpadded
