"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.models.blip_models.blip_outputs import BlipOutput
try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")
import argparse
import os
from torchvision.transforms.functional import to_pil_image

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import cv2
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
    def forward(self, input):
        return torch.sigmoid(self.layers(input))


def visualize(pred_mask, image_path, work_dir):
    visualizer = Visualizer()
    img = cv2.imread(image_path)
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
    visual_result = visualizer.get_image()

    output_path = os.path.join(work_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, visual_result)
@registry.register_model("fga_blip2")
class FGA_Blip2(Blip2Qformer):
    """
    BLIP Image-Text Matching (ITM) model.
    Supported model types:
        - pretrained: pretrained model
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_image_text_matching", "pretrained")
        >>> model = load_model("blip2_image_text_matching", "coco")
    """

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )
        # self.mask_proj = torch.nn.Linear(self.Qformer.config.hidden_size, 1)
        # self.weight_proj = MLP(self.Qformer.config.hidden_size)
        self.mask_proj = MLP(self.Qformer.config.hidden_size)
        # for name, parms in self.named_parameters():
        #     if '_proj' not in name:
        #         parms.requires_grad_(False)
        # self.itm_score_head = nn.Sequential(
        #     nn.Linear(num_query_token, 16),  # num_query_token 通常是 32
        #     nn.ReLU(),
        #     nn.Linear(16, 1)
        # )186945088   0卡     
        self.itm_score_head = nn.Sequential(
            nn.Linear(num_query_token, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
#186949280   1卡
    def element_score(self, image, caption):
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        # breakpoint()
        text = self.tokenizer(
            caption,
            # padding="max_length",
            truncation=False,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
        output_itm = self.Qformer.bert(
            text.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        itm_embeddings = output_itm.last_hidden_state[:, :, :]
        itm_logit = self.itm_head(itm_embeddings)
        itm_scores = torch.nn.functional.softmax(itm_logit, dim=2)[:,:,1]
        # itm_score = (itm_scores * mask).sum(dim=1) / mask.sum(dim=1)
        alignment_score = itm_scores[:, :query_tokens.size(1)].mean(dim=1) * 4 + 1

        return alignment_score, itm_scores[:, query_tokens.size(1):]

    def forward(self, samples, match_head="itm", inference = False):
        # breakpoint()
        image = samples["image"]
        caption = samples["text_input"]
        # import pdb
        # pdb.set_trace()

        if inference == False:
            mask_gt = torch.tensor(samples["mask"]).to(image.device)
            token_score = torch.tensor(samples["token_score"]).to(image.device)
            score = torch.tensor(samples["score"]).to(image.device)
            var = torch.tensor(samples["var"]).to(image.device)
            image_embeds = self.ln_vision(self.visual_encoder(image))
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        # breakpoint()
        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        if match_head == "itm":
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                image.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
            output_itm = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            itm_embeddings = output_itm.last_hidden_state[:, :, :]
            itm_logit = self.itm_head(itm_embeddings)
            itm_scores = torch.nn.functional.softmax(itm_logit, dim=2)[:,:,1]
            # import pdb;pdb.set_trace()
            # # 1. 计算 token-level 的匹配分数
            # itm_logit = self.itm_head(itm_embeddings)                     # [batch_size, seq_len, 2]
            # itm_scores = torch.nn.functional.softmax(itm_logit, dim=2)[:, :, 1]  # [batch_size, seq_len]

            # # 2. 只取前 query_tokens.size(1) 个 token 对应的分数
            # query_scores = itm_scores[:, :query_tokens.size(1)]           # [batch_size, query_len]

            # # 3. 使用自定义的 MLP 将 query_scores -> [batch_size, 1]
            # #    这里要求 MLP 的第一层线性层输入维度 = query_len
            # final_score = self.itm_score_head(query_scores)               # [batch_size, 1]

            # # 4. 你可以自己决定是否要做一个简单的映射到 [1,5] 区间，或者直接让网络去学习
            # itm_score = final_score.squeeze(-1) * 4 + 1   # [batch_size]


            # mask = self.mask_proj(itm_embeddings).squeeze(dim=2)
            # mask = torch.sigmoid(mask)
            # mask = mask * text.attention_mask


            # mask = torch.sigmoid(mask)
            # mask = mask * text.attention_mask
            # ############## stage 1 #################
            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            mask = self.mask_proj(text_output.last_hidden_state).squeeze(dim=2)

            itm_score = itm_scores[:, :query_tokens.size(1)].mean(dim=1) * 4 + 1


            # itm_score = (itm_scores * mask).sum(dim=1) / mask.sum(dim=1) * 4 + 1
            # itm_logit = (itm_logit * mask).sum(dim=1) / mask.sum(dim=1)
            # breakpoint()
            # itm_scores = torch.nn.functional.softmax(itm_logit, dim=1) * 4 + 1
            
            # breakpoint()
            # itm_scores = self.mlp(itm_embeddings).mean(dim=1) * 4 + 1
            if inference:
                # mask = torch.cat([torch.ones(mask.shape).to(mask.device),mask.detach() > 0.5],dim=1)
                # itm_score = (itm_scores * mask).sum(dim=1) / mask.sum(dim=1) * 4 + 1
                
                # mask = mask.detach() > 0.5
                # itm_score = (itm_scores[:, query_tokens.size(1):] * mask).sum(dim=1) / mask.sum(dim=1) * 4 + 1
                
                return itm_score
            l1_loss = torch.nn.L1Loss(reduction='mean')
            diff_score = torch.abs(itm_score - score)
            diff_token_score = torch.abs(itm_scores[:, query_tokens.size(1):] * mask_gt - token_score).mean(dim=1)
            diff_mask = torch.abs(mask - mask_gt).mean(dim=1)
            loss_itm = torch.mean(var * (diff_score + 0.1 * diff_token_score + 0.1 * diff_mask))
            # loss_itm = (itm_scores[:, 1] - score) * (itm_scores[:, 1] - score)
            # breakpoint()
            # loss_itm = loss_itm.mean()
            return BlipOutput(loss=loss_itm, loss_itm=loss_itm)

            ############## stage 2 #################
            # text_output = self.Qformer.bert(
            #     text.input_ids,
            #     attention_mask=text.attention_mask,
            #     return_dict=True,
            # )
            # # breakpoint()

            # mask = self.mask_proj(text_output.last_hidden_state).squeeze(dim=2)
            # # print(mask[0])
            # weight = self.weight_proj(itm_embeddings).squeeze(dim=2)
            # weight = weight * torch.cat([torch.ones(mask.shape).to(mask.device),mask.detach() > 0.5],dim=1)

            # itm_score = (itm_scores * weight).sum(dim=1) / weight.sum(dim=1) * 4 + 1
            # # itm_score = itm_scores[:, :query_tokens.size(1)].mean(dim=1) * 4 + 1
            # # itm_score = (itm_scores * mask).sum(dim=1) / mask.sum(dim=1) * 4 + 1
            # # itm_logit = (itm_logit * mask).sum(dim=1) / mask.sum(dim=1)
            # # breakpoint()
            # # itm_scores = torch.nn.functional.softmax(itm_logit, dim=1) * 4 + 1
            
            # # itm_scores = self.mlp(itm_embeddings).mean(dim=1) * 4 + 1
            # if inference:
            #     return itm_score 
            # l1_loss = torch.nn.L1Loss(reduction='mean')
            # loss_itm = torch.mean(torch.exp(var) * (torch.abs(itm_score - score))) + l1_loss(mask, mask_gt)
            # # loss_itm = (itm_scores[:, 1] - score) * (itm_scores[:, 1] - score)
            # # breakpoint()
            # # loss_itm = loss_itm.mean()
            # return BlipOutput(loss=loss_itm, loss_itm=loss_itm)
        elif match_head == "itc":
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_feats = F.normalize(
                self.vision_proj(query_output.last_hidden_state), dim=-1
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            # text_feat = F.normalize(
            #     self.text_proj(text_output.last_hidden_state), dim=-1
            # )
            
            # mask = self.mask_proj(text_output.last_hidden_state)
            # mask = torch.softmax(mask.squeeze(), dim=1)
            # sims = torch.bmm(image_feats, text_feat.transpose(1, 2))
            # sims, _ = torch.max(sims, dim=1)
            # sim = torch.sum(sims * mask, dim=1)

            text_feat = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )

            sims = torch.bmm(image_feats, text_feat.unsqueeze(-1))
            sim, _ = torch.max(sims, dim=1)

            itc_scores = sim * 5
            if inference:
                # print(itc_scores.shape)
                return itc_scores.squeeze()
            loss_itc = (itc_scores - score) * (itc_scores - score)
            # print(loss_itc.shape)
            loss_itc = loss_itc.mean()
            return BlipOutput(loss=loss_itc, loss_itc=loss_itc)
