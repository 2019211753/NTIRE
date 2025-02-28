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
import torch.distributed as dist

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
        
        if inference == False:
            mask_gt = torch.tensor(samples["mask"]).to(image.device)
            token_score = torch.tensor(samples["token_score"]).to(image.device)
            score = torch.tensor(samples["score"]).to(image.device)
            var = torch.tensor(samples["var"]).to(image.device) # 衡量prompt难度
            image_embeds = self.ln_vision(self.visual_encoder(image))
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        ) # image和query的交叉注意力是全1的mask
        # breakpoint()
        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device) # [14, 32]

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1) # [1, 32, 768] -> [14, 32, 768]
        # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to( # [14, 32]
        #     image.device
        # )
        # attention_mask = torch.cat([query_atts, text.attention_mask], dim=1) # [14, 64] attention_mask标志padding部分
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )
        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        text_output = self.Qformer.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        )  # last_hidden_state [14, 32, 768]
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        ) # [cls], [14, 256]
        # [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat.unsqueeze(-1)
        ).squeeze() # [14, 14, 32]

        sim_i2t, _ = sim_q2t.max(-1) # [14, 14]
        sim_i2t = sim_i2t / self.temp

        rank = dist.get_rank()
        bs = image.size(0)

        # text-query similarity: [batch_size, batch_size, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats.permute(0, 2, 1)
        ).squeeze() # [14, 14, 32]

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1) # [14, 14]
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_t2i[:, rank * bs: rank * bs + bs].fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            weights_i2t[:, rank * bs: rank * bs + bs].fill_diagonal_(0) # [14, 14[

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text.input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text.input_ids, text.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text.attention_mask, text.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            image.device
        )

        if match_head == "itm":
            output_itm = self.Qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_atts_all,
                return_dict=True,
            )

            itm_embeddings = output_itm.last_hidden_state[:, :, :] # [14, 64, 768]
            itm_logit = self.itm_head(itm_embeddings) # [14, 64, 2]
            cs_logits = itm_logit.mean(dim=1)
            itm_labels = torch.cat(
                [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                dim=0,
            ).to(image.device)
            loss_cs = F.cross_entropy(cs_logits, itm_labels)

            # mask = self.mask_proj(itm_embeddings).squeeze(dim=2)
            # mask = torch.sigmoid(mask)
            # mask = mask * text.attention_mask
            # mask = torch.sigmoid(mask)
            # mask = mask * text.attention_mask

            # ############## stage 1 #################
            itm_scores = torch.nn.functional.softmax(itm_logit, dim=2)[:,:,1] # [14, 64], in [0, 1]
            itm_score = itm_scores[:bs, :].mean(dim=1) * 4 + 1 # item_scores[0:32], project to [0, 5] score
            mask = self.mask_proj(text_output.last_hidden_state).squeeze(dim=2)
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
            diff_token_score = torch.abs(itm_scores[:bs, query_tokens.size(1):] * mask_gt - token_score).mean(dim=1) # token level itm_scores[:32]
            diff_mask = torch.abs(mask - mask_gt).mean(dim=1)
            loss_itm = torch.mean(var * (diff_score + 0.1 * diff_token_score + 0.1 * diff_mask + loss_cs))
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
        return None
