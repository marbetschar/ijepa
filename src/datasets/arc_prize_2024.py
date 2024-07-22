# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
import time
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

def challenge_to_image_files(working_dir, challenge_id, trainings, tests, solutions, image_variations=0):
    trainings_dir = os.path.join(working_dir, 'train', challenge_id)
    tests_dir = os.path.join(working_dir, 'test', challenge_id)
    solutions_dir = os.path.join(working_dir, 'solution', challenge_id)

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

##########################################################################

def make_imagenet1k(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None
):
    dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training,
        copy_data=copy_data,
        index_targets=False)
    if subset_file is not None:
        dataset = ImageNetSubset(dataset, subset_file)
    logger.info('ImageNet dataset created')
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
    logger.info('ImageNet unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class ImageNet(torchvision.datasets.ImageFolder):

    def __init__(
        self,
        root,
        image_folder='imagenet_full_size/061417/',
        tar_file='imagenet_full_size-061417.tar.gz',
        transform=None,
        train=True,
        job_id=None,
        local_rank=None,
        copy_data=True,
        index_targets=False
    ):
        """
        ImageNet

        Dataset wrapper (can copy data locally to machine)

        :param root: root network directory for ImageNet data
        :param image_folder: path to images inside root network directory
        :param tar_file: zipped image_folder inside root network directory
        :param train: whether to load train data (or validation)
        :param job_id: scheduler job-id used to create dir on local machine
        :param copy_data: whether to copy data from network file locally
        :param index_targets: whether to index the id of each labeled image
        """

        suffix = 'train/' if train else 'val/'
        data_path = None
        if copy_data:
            logger.info('copying data locally')
            data_path = copy_imgnt_locally(
                root=root,
                suffix=suffix,
                image_folder=image_folder,
                tar_file=tar_file,
                job_id=job_id,
                local_rank=local_rank)
        if (not copy_data) or (data_path is None):
            data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')

        super(ImageNet, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized ImageNet')

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


class ImageNetSubset(object):

    def __init__(self, dataset, subset_file):
        """
        ImageNetSubset

        :param dataset: ImageNet dataset object
        :param subset_file: '.txt' file containing IDs of IN1K images to keep
        """
        self.dataset = dataset
        self.subset_file = subset_file
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file):
        """ Filter self.dataset to a subset """
        root = self.dataset.root
        class_to_idx = self.dataset.class_to_idx
        # -- update samples to subset of IN1k targets/samples
        new_samples = []
        logger.info(f'Using {subset_file}')
        with open(subset_file, 'r') as rfile:
            for line in rfile:
                class_name = line.split('_')[0]
                target = class_to_idx[class_name]
                img = line.split('\n')[0]
                new_samples.append(
                    (os.path.join(root, class_name, img), target)
                )
        self.samples = new_samples

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.dataset.loader(path)
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target



