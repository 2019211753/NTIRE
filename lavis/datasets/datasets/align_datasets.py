"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import numpy as np

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": os.path.basename(ann["image"]),
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class AlignmentDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # self.annotation_element = []
        # for ann in self.annotation:
        #     for key in ann['element_score'].keys():
        #         self.annotation_element.append({
        #             'img_path': ann['img_path'],
        #             'prompt': key.rpartition('(')[0],
        #             'total_score': ann['element_score'][key] * 4 + 1
        #         })
        # self.annotation = self.annotation_element

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["img_path"])
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            return None

        image = self.vis_processor(image)
        caption = self.text_processor(ann["prompt"])
        score = np.mean(ann['total_score'])

        var = None
        if 'var' in ann:
            var = ann['var']

        mask = None
        token_score = None
        if 'mask' in ann:
            if len(ann['mask']) > 32:
                ann['mask'] = ann['mask'][:32]
                ann['token_score'] = ann['token_score'][:32]
            mask = ann['mask'] + [0] * (32 - len(ann['mask']))
            token_score = ann['token_score'] + [0] * (32 - len(ann['token_score']))
            mask[0] = 1
            token_score[0] = (score - 1) / 4
        # score = torch.tensor(score)
        return {"image": image, "text_input": caption, "score": score, "mask":mask, "token_score":token_score, 'var': var}
    
class AlignmentEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        caption = ann["prompt"]
        score = np.mean(ann['total_score'])

        return {"image": image, "text_input": caption, "score": score}

