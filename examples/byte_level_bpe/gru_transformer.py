# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerEncoder, TransformerModel


@register_model("gru_transformer")
class GRUTransformerModel(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return GRUTransformerEncoder(args, src_dict, embed_tokens)


class GRUTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.emb_ctx = nn.GRU(
            input_size=embed_tokens.embedding_dim,
            hidden_size=embed_tokens.embedding_dim // 2,
            num_layers=1,
            bidirectional=True,
        )

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)

        # contextualize embeddings
        x = x.transpose(0, 1)
        x = self.dropout_module(x)
        x, _ = self.emb_ctx.forward(x)
        x = x.transpose(0, 1)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x, embed


@register_model_architecture("gru_transformer", "gru_transformer")
def gru_transformer_base_architecture(args):
    args.encoder_embed_path = args.get("encoder_embed_path", None)
    args.encoder_embed_dim = args.get("encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = args.get("encoder_ffn_embed_dim", 2048)
    args.encoder_layers = args.get("encoder_layers", 6)
    args.encoder_attention_heads = args.get("encoder_attention_heads", 8)
    args.encoder_normalize_before = args.get("encoder_normalize_before", False)
    args.encoder_learned_pos = args.get("encoder_learned_pos", False)
    args.decoder_embed_path = args.get("decoder_embed_path", None)
    args.decoder_embed_dim = args.get("decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = args.get("decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = args.get("decoder_layers", 6)
    args.decoder_attention_heads = args.get("decoder_attention_heads", 8)
    args.decoder_normalize_before = args.get("decoder_normalize_before", False)
    args.decoder_learned_pos = args.get("decoder_learned_pos", False)
    args.attention_dropout = args.get("attention_dropout", 0.0)
    args.activation_dropout = args.get("activation_dropout", 0.0)
    args.activation_fn = args.get("activation_fn", "relu")
    args.dropout = args.get("dropout", 0.1)
    args.adaptive_softmax_cutoff = args.get("adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = args.get("adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = args.get("share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = args.get("share_all_embeddings", False)
    args.no_token_positional_embeddings = args.get("no_token_positional_embeddings", False
    )
    args.adaptive_input = args.get("adaptive_input", False)
    args.no_cross_attention = args.get("no_cross_attention", False)
    args.cross_self_attention = args.get("cross_self_attention", False)
    args.layer_wise_attention = args.get("layer_wise_attention", False)

    args.decoder_output_dim = args.get("decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = args.get("decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = args.get("no_scale_embedding", False)
    args.layernorm_embedding = args.get("layernorm_embedding", False)


@register_model_architecture("gru_transformer", "gru_transformer_big")
def gru_transformer_big(args):
    args.encoder_embed_dim = args.get("encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = args.get("encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = args.get("encoder_attention_heads", 16)
    args.encoder_normalize_before = args.get("encoder_normalize_before", False)
    args.decoder_embed_dim = args.get("decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = args.get("decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = args.get("decoder_attention_heads", 16)
    args.dropout = args.get("dropout", 0.3)
    gru_transformer_base_architecture(args)
