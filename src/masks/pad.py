from logging import getLogger
from src.masks.utils import pad

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class PadCollator(object):

    def __call__(self, batch):
        batch_padded = []
        for item in batch:
            batch_padded.append(pad(item))

        collated_batch = torch.utils.data.default_collate(batch_padded)
        return collated_batch, None, None
