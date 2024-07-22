# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import shutil
import cv2

import numpy as np

from logging import getLogger

import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()

def json_file_read(file_path):
    file = open(file_path, 'r')
    json_data = json.loads(file.read())
    file.close()
    return json_data

def image_to_file(dir_name, image_name, image_data, image_variations=0):
    image = np.array(image_data)

    cv2.imwrite(os.path.join(dir_name, f"{image_name}-0.png"), image)

    if image_variations > 0:
        values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        for i in range(1, min(11, image_variations + 1), 1):
            image_variation = (np.copy(image) + i) % 10

            cv2.imwrite(os.path.join(dir_name, f"{image_name}-{i}.png"), image_variation)

def image_from_file(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def is_valid_input_file(file_path):
    return file_path.__contains__("input-")

def challenge_to_image_files(working_dir, challenge_id, trainings, tests, solutions, image_variations=0):
    trainings_dir = os.path.join(working_dir, 'train', challenge_id)
    tests_dir = os.path.join(working_dir, 'test', challenge_id)
    solutions_dir = os.path.join(working_dir, 'test', challenge_id)

    for d in [trainings_dir, tests_dir, solutions_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    for i, train in enumerate(trainings):
        image_to_file(trainings_dir, 'input', train['input'], image_variations)
        image_to_file(trainings_dir, 'output', train['output'], image_variations)

    for i, test in enumerate(tests):
        image_to_file(tests_dir, 'input', test['input'], image_variations)
        image_to_file(solutions_dir, 'output', solutions[i], image_variations)


def challenges_to_image_files(working_dir, challenges, solutions, image_variations=0, skip_if_working_dir_exists=False):
    if skip_if_working_dir_exists and os.path.exists(working_dir):
        return
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)

    for challenge_id in challenges:
        challenge_to_image_files(
            working_dir,
            challenge_id,
            challenges[challenge_id]['train'],
            challenges[challenge_id]['test'],
            solutions[challenge_id],
            image_variations
        )

def make_arcprize2024(
    transform,
    target_transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    drop_last=True
):
    dataset = ArcPrize2024(
        root=root_path,
        image_folder=image_folder,
        train=training,
        index_targets=False,
        transform=transform,
        target_transform=target_transform
    )
    # if subset_file is not None:
    #     dataset = ImageNetSubset(dataset, subset_file)
    logger.info('ArcPrize2024 dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('ArcPrize2024 unsupervised data loader created')

    return dataset, data_loader, dist_sampler

class ArcPrize2024(torchvision.datasets.DatasetFolder):

    def __init__(
        self,
        root,
        image_folder='arc-prize-2024/images/',
        train=True,
        job_id=None,
        local_rank=None,
        index_targets=False,
        transform=None,
        target_transform=None,
        loader=image_from_file,
        is_valid_file=is_valid_input_file
    ):
        """
        ArcPrize2024

        Dataset wrapper (can copy data locally to machine)

        :param root: root network directory for ImageNet data
        :param image_folder: path to images inside root network directory
        :param train: whether to load training data (or evaluation)
        :param job_id: scheduler job-id used to create dir on local machine
        :param index_targets: whether to index the id of each labeled image
        """

        suffix = 'training/train' if train else 'evaluation/train'
        data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')

        super(ArcPrize2024, self).__init__(
            root=data_path,
            loader=loader,
            is_valid_file=is_valid_file,
            transform=transform,
            target_transform=target_transform
        )
        logger.info('Initialized ArcPrize2024')

        if index_targets:
            self.targets = []
            for sample in self.samples:
                self.targets.append(sample[1])
            self.targets = np.array(self.targets)
            self.samples = np.array(self.samples)

            mint = None
            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(
                    self.targets == t)).tolist()
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(mint, len(indices))
                logger.debug(f'num-labeled target {t} {len(indices)}')
            logger.info(f'min. labeled indices {mint}')

    def __getitem__(self, index: int):
        path, _ = self.samples[index]

        sample = self.loader(path)
        target = self.loader(path.replace("input-", "output-"))

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target