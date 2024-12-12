# Some code is copied from the official repository for "ClipCap: CLIP Prefix for Image Captioning"
# https://github.com/rmokady/CLIP_prefix_caption
from typing import Optional

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          LlamaForCausalLM)

from zhclip.models import ZhCLIPModel, ZhCLIPProcessor

Tensor = torch.Tensor


class LLM(nn.Module):

    def __init__(self,
                 model_name: str = 'IDEA-CCNL/Wenzhong-GPT2-110M',
                 freeze_llm: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if freeze_llm:
            self.model = self.model.eval()

    def forward(self, tokens: Tensor) -> Tensor:
        return self.model(tokens)


class CLIP(nn.Module):

    def __init__(self,
                 model_name: str = 'thu-ml/zh-clip-vit-roberta-large-patch14'):
        super().__init__()
        if 'zh-clip' in model_name:
            model = ZhCLIPModel.from_pretrained(model_name)
            self.processor = ZhCLIPProcessor.from_pretrained(model_name)
        elif 'ViT' in model_name or 'RN' in model_name:
            model, self.processor = clip.load(model_name, jit=False)

        # 实际使用时，我们并不会训练 CLIP
        self.model = model.eval()

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        if isinstance(self.model, clip.model.CLIP):
            return torch.stack(
                [self.model.encode_image(img.unsqueeze(0)) for img in x],
                dim=0).to(x.dtype)
        return self.model.get_image_features(x)


class ChineseClipCap(nn.Module):

    def __init__(
        self,
        llm_model: str = 'IDEA-CCNL/Wenzhong-GPT2-110M',
        clip_model: str = 'thu-ml/zh-clip-vit-roberta-large-patch14',
        freeze_llm: bool = False,
        prefix_length: int = 10,
        clip_length: int = 10,
        prefix_dim: int = 512,
        num_layers: int = 8,
        mapping_type: str = 'mlp',
    ):
        """
        Args:
            - llm_model: str. LLM 模型
            - clip_model: str. CLIP 模型
            - freeze_llm: bool. 是否冻结 LLM 模型
            - prefix_length: int. 对 CLIP 输出的 image_features 进行映射的维度
            - clip_length: int. 对 CLIP 输出的 image_features 进行映射的维度 (Transformer Encoder)
            - prefix_dim: int. CLIP 输出的 image_features 的维度
            - num_layers: int. mapping 用到的 Transformer Encoder 的层数
            - mapping_type: str. 用 MLP 或 Transformer Encoder 来 project CLIP 输出的 image_features
        """
        super().__init__()
        self.llm = LLM(llm_model, freeze_llm)
        self.clip = CLIP(clip_model)
        self.prefix_length = prefix_length
        if isinstance(self.llm.model, GPT2LMHeadModel):
            self.llm_embed_size = self.llm.model.config.n_embd
            self.embed_fc = self.llm.model.transformer.wte
        elif isinstance(self.llm.model, LlamaForCausalLM):
            self.llm_embed_size = self.llm.model.config.hidden_size
            self.embed_fc = self.llm.model.model.embed_tokens
        if mapping_type == 'mlp':
            self.clip_project = MLP(
                (prefix_dim, (self.llm_embed_size * prefix_length) // 2,
                 self.llm_embed_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_dim,
                                                  self.llm_embed_size,
                                                  prefix_length, clip_length,
                                                  num_layers)

    def forward(
        self,
        tokens: Tensor,
        img: Tensor,
        mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        clip_feat = self.clip(img)
        prefix_projections = self.clip_project(clip_feat).view(
            -1, self.prefix_length, self.llm_embed_size)
        embedding_text = self.embed_fc(tokens)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_tokens = torch.zeros(tokens.shape[0],
                                       self.prefix_length,
                                       dtype=torch.int64,
                                       device=tokens.device)
            labels = torch.cat((dummy_tokens, tokens), dim=1)
        out = self.llm.model(inputs_embeds=embedding_cat,
                             labels=labels,
                             attention_mask=mask)
        return out

    @torch.no_grad()
    def predict(
        self,
        img: Tensor,
        temperature: float = 1.,
        max_caption_len: int = 67,
        beam_size: int = 5,
        stop_token: str = '</s>',
    ) -> list[str]:
        device = self.llm.model.device
        # BertTokenizerFast 也有可能
        if 'BertTokenizer' in str(type(self.llm.tokenizer)):
            stop_token_index = self.llm.tokenizer.sep_token_id  # BertTokenizer 的 [SEP] 对应的 id
        elif 'LlamaTokenizer' in str(type(self.llm.tokenizer)):
            stop_token_index = self.llm.tokenizer.eos_token_id  # LlamaTokenizer 的 eos token 对应的 id
        else:
            stop_token_index = self.llm.tokenizer.encode(stop_token)[0]
        clip_feat = self.clip(img)
        generated = self.clip_project(clip_feat).view(-1, self.prefix_length,
                                                      self.llm_embed_size)
        batch_results = []
        for i in range(generated.shape[0]):
            generated = generated[i, :, :].unsqueeze(0)
            seq_lengths = torch.ones(beam_size, device=device)
            is_stopped = torch.zeros(beam_size,
                                     device=device,
                                     dtype=torch.bool)
            tokens = None
            scores = None
            for _ in range(max_caption_len):
                outputs = self.llm.model(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature
                                             if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size,
                                                 *generated.shape[1:])
                    next_tokens = next_tokens.permute(1, 0)
                    scores = scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average = scores_sum_average.view(-1)
                    scores_sum_average, next_tokens = scores_sum_average.topk(
                        beam_size, -1)
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                next_token_embed = self.embed_fc(next_tokens.squeeze()).view(
                    generated.shape[0], 1, -1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(
                    stop_token_index).squeeze()
                if is_stopped.all():
                    break

            scores = scores / seq_lengths
            tokens = tokens.cpu().numpy()
            output_texts = [
                self.llm.tokenizer.decode(output[:int(length)],
                                          skip_special_tokens=True).replace(
                                              ' ', '').split('。')[0]
                for output, length in zip(tokens, seq_lengths)
            ]
            order = scores.argsort(descending=True)
            batch_results.append(output_texts[order[0]])

        return batch_results

    @torch.no_grad()
    def predict_wo_beamsearch(
        self,
        img: Tensor,
        temperature: float = 1.0,
        max_caption_len: int = 67,
        top_p: float = 0.8,
        stop_token: str = '</s>',
    ):
        # BertTokenizerFast 也有可能
        if 'BertTokenizer' in str(type(self.llm.tokenizer)):
            stop_token_index = self.llm.tokenizer.sep_token_id  # BertTokenizer 的 [SEP] 对应的 id
        elif 'LlamaTokenizer' in str(type(self.llm.tokenizer)):
            stop_token_index = self.llm.tokenizer.eos_token_id  # LlamaTokenizer 的 eos token 对应的 id
        else:
            stop_token_index = self.llm.tokenizer.encode(stop_token)[0]
        clip_feat = self.clip(img)
        generated = self.clip_project(clip_feat).view(-1, self.prefix_length,
                                                      self.llm_embed_size)
        filter_value = float('-inf')
        tokens = None

        for _ in range(max_caption_len):
            outputs = self.llm.model(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :]
            logits /= temperature if temperature > 0 else 1.0
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                            dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value
            next_token = torch.argmax(logits, -1).unsqueeze(1)
            next_token_embed = self.embed_fc(next_token)
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            flag = stop_token_index == next_token
            if flag.all():
                break

            output_list = list(tokens.cpu().numpy())
            output_texts = [
                self.llm.tokenizer.decode(o, skip_special_tokens=True).replace(
                    ' ', '').split('。')[0] for o in output_list
            ]

        return output_texts


class MLP(nn.Module):

    def __init__(self,
                 sizes: tuple[int, ...],
                 bias: bool = True,
                 act: nn.Module = nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class Transformer(nn.Module):

    def __init__(self,
                 dim_self: int,
                 num_heads: int,
                 num_layers: int,
                 dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2.,
                 act=F.relu,
                 norm_layer: nn.Module = nn.LayerNorm,
                 enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(
                    TransformerLayer(dim_self,
                                     dim_ref,
                                     num_heads,
                                     mlp_ratio,
                                     act=act,
                                     norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(
                    TransformerLayer(dim_self,
                                     dim_self,
                                     num_heads,
                                     mlp_ratio,
                                     act=act,
                                     norm_layer=norm_layer))
            else:  # self or cross
                layers.append(
                    TransformerLayer(dim_self,
                                     dim_ref,
                                     num_heads,
                                     mlp_ratio,
                                     act=act,
                                     norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:  # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self,
                 dim_self,
                 dim_ref,
                 num_heads,
                 mlp_ratio=4.,
                 bias=False,
                 dropout=0.,
                 act=F.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self,
                                       dim_ref,
                                       num_heads,
                                       bias=bias,
                                       dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self,
                                  int(dim_self * mlp_ratio),
                                  act=act,
                                  dropout=dropout)


class MlpTransformer(nn.Module):

    def __init__(self,
                 in_dim,
                 h_dim,
                 out_d: Optional[int] = None,
                 act=F.relu,
                 dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim**-0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads,
                                             c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads,
                                                     c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float('-inf'))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention,
                           values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(
            x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self,
                 dim_clip: int,
                 dim_embedding: int,
                 prefix_length: int,
                 clip_length: int,
                 num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length,
                                                     dim_embedding),
                                         requires_grad=True)
