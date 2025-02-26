"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.datasets.datasets.align_datasets import (
    AlignmentDataset,
    AlignmentEvalDataset,
)

from lavis.common.registry import registry

@registry.register_builder("alignment")
class AlignmentBuilder(BaseDatasetBuilder):
    train_dataset_cls = AlignmentDataset
    eval_dataset_cls = AlignmentEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/alignment/defaults.yaml"}

