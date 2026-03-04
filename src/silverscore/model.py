"""
CLIP4Clip model for sign language video-text similarity scoring.
Inference-only: all distributed training logic is removed.
Adapted from the CiCo/CLCL codebase (modules/modeling.py).
"""

import logging
import torch
from torch import nn
from .base_modules import PreTrainedModel, CrossEn
from .cross_modules import CrossModel, CrossConfig, Transformer as TransformerClip
from .clip_modules import CLIP, convert_weights

logger = logging.getLogger(__name__)


def _update_attr(target_name, target_config, target_attr_name,
                 source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name,
                    getattr(source_config, source_attr_name))
    return target_config


class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    def __init__(self, cross_config, *inputs, **kwargs):
        super().__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None,
                        cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
        task_config = kwargs.get("task_config", None)
        if task_config is not None:
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None:
            state_dict = {}

        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, "pretrained_clip_name"):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(
            pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(
            cross_model_name, cache_dir, type_vocab_size,
            state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        if model.sim_header in ("seqLSTM", "seqTransf"):
            has_fp = any("frame_position_embeddings" in k
                         for k in state_dict)
            if not has_fp:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict[
                            "frame_position_embeddings.weight"
                        ] = val.clone()
                        continue
                    if (model.sim_header == "seqTransf"
                            and key.startswith("transformer.resblocks")):
                        num_layer = int(key.split(".")[2])
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[
                                key.replace("transformer.",
                                            "transformerClip.")
                            ] = val.clone()

        if state_dict is not None:
            model = cls.init_preweight(
                model, state_dict, task_config=task_config)

        return model


class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super().__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1
        self._stage_one = True
        self.dual_mix = getattr(task_config, "dual_mix", 0.5)
        self.mix_design = getattr(task_config, "mix_design", "balance")

        self.loose_type = False
        if self._stage_one and getattr(task_config, "loose_type", False):
            self.loose_type = True

        vit = "visual.proj" in clip_state_dict
        assert vit, "Only ViT-based CLIP is supported."

        vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
        vision_patch_size = clip_state_dict[
            "visual.conv1.weight"].shape[-1]
        grid_size = round(
            (clip_state_dict[
                "visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        vision_layers = getattr(
            task_config, "visual_num_hidden_layers", 12)

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict[
            "positional_embedding"].shape[0]
        vocab_size = clip_state_dict[
            "token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(
            k.split(".")[2] for k in clip_state_dict
            if k.startswith("transformer.resblocks")))

        self.linear_patch = getattr(task_config, "linear_patch", "2d")

        feature_len = getattr(task_config, "feature_len", 64)
        self.clip = CLIP(
            embed_dim, image_resolution, vision_layers,
            vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width,
            transformer_heads, transformer_layers,
            feature_len=feature_len,
            linear_patch=self.linear_patch,
        ).float()

        self.alpha = getattr(task_config, "alpha", 0.8)

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)

        self.sim_header = getattr(task_config, "sim_header", "Filip")
        if self.sim_header == "tightTransf":
            assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            cross_config = _update_attr(
                "cross_config", cross_config, "num_hidden_layers",
                self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            self.similarity_dense = nn.Linear(
                cross_config.hidden_size, 1)

        if self.sim_header in ("seqLSTM", "seqTransf"):
            self.frame_position_embeddings = nn.Embedding(
                cross_config.max_position_embeddings,
                cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(
                width=transformer_width,
                layers=getattr(
                    task_config, "cross_num_hidden_layers", 4),
                heads=transformer_heads)
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(
                input_size=cross_config.hidden_size,
                hidden_size=cross_config.hidden_size,
                batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn()
        self.apply(self.init_weights)

    def get_sequence_output(self, input_ids, token_type_ids,
                            attention_mask, shaped=False,
                            get_hidden=True):
        if not shaped:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(
                -1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(
                -1, attention_mask.shape[-1])

        if self.sim_header == "Filip" and get_hidden:
            text_mask, seq_hidden, seq_cls = self.clip.encode_text(
                input_ids, return_hidden=True)
            return text_mask, seq_hidden.float(), seq_cls.float()
        else:
            seq_hidden = self.clip.encode_text(input_ids).float()
            bs = input_ids.size(0)
            seq_hidden = seq_hidden.view(bs, -1, seq_hidden.size(-1))
            return seq_hidden

    def get_visual_output(self, video, video_mask, shaped=True,
                          video_frame=-1, get_hidden=True):
        if not shaped:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)

        if video_frame != -1:
            if self.sim_header == "Filip" and get_hidden:
                video_mask, vis_h, vis_cls = self.clip.encode_image(
                    video, return_hidden=True,
                    video_mask=video_mask, video_frame=video_frame)
                vis_h = vis_h.float().view(
                    bs_pair, -1, vis_h.size(-1))
                return video_mask, vis_h, vis_cls.float()
            else:
                vis_h = self.clip.encode_image(
                    video, video_mask=video_mask,
                    video_frame=video_frame).float()
        else:
            vis_h = self.clip.mlp(video.half()).float()

        vis_h = vis_h.view(bs_pair, -1, vis_h.size(-1))
        vis_cls = vis_h[:, 0, :]
        return video_mask, vis_h, vis_cls

    def flip_similarity_softmax(self, sequence_output, visual_output,
                                attention_mask, video_mask,
                                sequence_hidden_aug=None,
                                text_mask_aug=None):
        video_mask = (video_mask == 0)
        attention_mask = (attention_mask == 1)
        text_mask_aug = (text_mask_aug == 1)

        visual_output = (visual_output
                         / visual_output.norm(dim=-1, keepdim=True))
        visual_output = visual_output.squeeze(1)
        sequence_output = (sequence_output
                           / sequence_output.norm(dim=-1, keepdim=True))
        sequence_output = sequence_output.squeeze(1)
        sequence_hidden_aug = (sequence_hidden_aug
                               / sequence_hidden_aug.norm(
                                   dim=-1, keepdim=True))
        sequence_hidden_aug = sequence_hidden_aug.squeeze(1)

        batch_size = visual_output.shape[0]
        batch_size_t = sequence_output.shape[0]
        logit_scale = self.clip.logit_scale.exp()

        i2t_sim = torch.einsum(
            "ais,bjs->abij", [visual_output, sequence_output])
        after_softmax = torch.nansum(
            i2t_sim * torch.softmax(i2t_sim / 0.07, dim=3), dim=3)
        vm_ext = video_mask.unsqueeze(1).repeat(1, batch_size_t, 1)
        after_softmax[~vm_ext] = 0
        I2T = (logit_scale
               * torch.nansum(after_softmax, dim=-1)
               / torch.sum(vm_ext, dim=-1))

        i2t_aug = torch.einsum(
            "ais,bjs->abij", [visual_output, sequence_hidden_aug])
        after_sm_t2i = torch.nansum(
            i2t_aug * torch.softmax(i2t_aug / 0.07, dim=2), dim=2)
        tm_ext = text_mask_aug.unsqueeze(0).repeat(batch_size, 1, 1)
        after_sm_t2i[~tm_ext] = 0
        T2I = (logit_scale
               * torch.nansum(after_sm_t2i * tm_ext, dim=-1)
               / torch.sum(tm_ext, dim=-1))

        return I2T, T2I

    def _loose_similarity(self, sequence_output, visual_output,
                          attention_mask, video_mask,
                          sim_header="meanP"):
        sequence_output = sequence_output.contiguous()
        visual_output = visual_output.contiguous()

        if sim_header == "seqTransf":
            vis_orig = visual_output
            seq_len = visual_output.size(1)
            pos_ids = torch.arange(
                seq_len, dtype=torch.long,
                device=visual_output.device)
            pos_ids = pos_ids.unsqueeze(0).expand(
                visual_output.size(0), -1)
            fp_emb = self.frame_position_embeddings(pos_ids)
            visual_output = visual_output + fp_emb
            ext_mask = ((1.0 - video_mask.unsqueeze(1)) * -1e6)
            ext_mask = ext_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)
            visual_output = self.transformerClip(
                visual_output, ext_mask)
            visual_output = visual_output.permute(1, 0, 2)
            visual_output = visual_output + vis_orig

        visual_output = (visual_output
                         / visual_output.norm(dim=-1, keepdim=True))
        visual_output = visual_output.squeeze(1)
        sequence_output = sequence_output.squeeze(1)
        sequence_output = (sequence_output
                           / sequence_output.norm(dim=-1, keepdim=True))
        logit_scale = self.clip.logit_scale.exp()
        return logit_scale * torch.matmul(
            sequence_output, visual_output.t())

    def get_similarity_logits(self, sequence_output, visual_output,
                              attention_mask, video_mask,
                              shaped=False, loose_type=False,
                              is_train=True,
                              sequence_hidden_aug=None,
                              text_mask_aug=None):
        if not shaped:
            attention_mask = attention_mask.view(
                -1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if sequence_hidden_aug is None:
            sequence_hidden_aug = sequence_output
            text_mask_aug = attention_mask

        if self.sim_header == "Filip" and is_train:
            I2T, T2I = self.flip_similarity_softmax(
                sequence_output, visual_output,
                attention_mask, video_mask,
                sequence_hidden_aug=sequence_hidden_aug,
                text_mask_aug=text_mask_aug)
            return I2T, T2I, ()

        if loose_type:
            logits = self._loose_similarity(
                sequence_output, visual_output,
                attention_mask, video_mask,
                sim_header=self.sim_header)
        else:
            logits = self._cross_similarity(
                sequence_output, visual_output,
                attention_mask, video_mask)
        return logits, ()

    def _cross_similarity(self, sequence_output, visual_output,
                          attention_mask, video_mask):
        sequence_output = sequence_output.contiguous()
        visual_output = visual_output.contiguous()
        b_t, s_t, h_t = sequence_output.size()
        b_v, s_v, h_v = visual_output.size()
        logits_list = []
        step = 2
        splits = [step] * (b_t // step)
        rem = b_t - sum(splits)
        if rem > 0:
            splits += [rem]
        a_mask = torch.ones(b_t, 1).to(
            device=attention_mask.device,
            dtype=attention_mask.dtype)
        seq_splits = torch.split(sequence_output, splits, dim=0)
        mask_splits = torch.split(a_mask, splits, dim=0)
        for i in range(len(splits)):
            sr = seq_splits[i]
            mr = mask_splits[i]
            st = sr.size(0)
            sl = sr.unsqueeze(1).repeat(
                1, b_v, 1, 1).view(-1, s_t, h_t)
            ml = mr.unsqueeze(1).repeat(
                1, b_v, 1).view(-1, s_t)
            vr = visual_output.unsqueeze(0).repeat(
                st, 1, 1, 1).view(-1, s_v, h_v)
            vmr = video_mask.unsqueeze(0).repeat(
                st, 1, 1).view(-1, s_v)
            cat_f = torch.cat((sl, vr), dim=1)
            cat_m = torch.cat((ml, vmr), dim=1)
            tt = torch.zeros_like(ml)
            vt = torch.ones_like(vmr)
            cat_type = torch.cat((tt, vt), dim=1)
            _, pooled = self.cross(
                cat_f, cat_type, cat_m,
                output_all_encoded_layers=True)
            row = self.similarity_dense(
                pooled).squeeze(-1).view(st, b_v)
            logits_list.append(row)
        return torch.cat(logits_list, dim=0)
