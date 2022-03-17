# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from fairseq.model_parallel.models.transformer import ModelParallelTransformerDecoder
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer_lm import TransformerLanguageModel

try:
    from fairseq.model_parallel.megatron.mpu import VocabParallelEmbedding

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("model_parallel_transformer_lm")
class ModelParallelTransformerLanguageModel(TransformerLanguageModel):
    @staticmethod
    def add_args(parser):
        TransformerLanguageModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        if not has_megatron_submodule:
            raise ImportError(
                "\n\nPlease install the megatron submodule:"
                "\n\n  git submodule update --init "
                "fairseq/model_parallel/megatron"
            )

        # make sure all arguments are present in older models
        base_lm_architecture(args)

        task.source_dictionary.pad_to_multiple_(args.model_parallel_size * 8)
        task.target_dictionary.pad_to_multiple_(args.model_parallel_size * 8)

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if args.get("max_target_positions", None) is None:
            args.max_target_positions = args.get("tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        if args.character_embeddings:
            raise NotImplementedError(
                "Character embeddings is not supported for model parallel"
            )
        elif args.adaptive_input:
            raise NotImplementedError(
                "Adaptive input is not supported for model parallel"
            )
        else:
            embed_tokens = cls.build_embedding(
                args, task.source_dictionary, args.decoder_input_dim
            )

        decoder = ModelParallelTransformerDecoder(
            args,
            task.target_dictionary,
            embed_tokens,
            no_encoder_attn=True,
        )
        return cls(decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        def _vocab_init(tensor, **kwargs):
            nn.init.normal_(tensor, mean=0, std=embed_dim**-0.5)
            nn.init.constant_(tensor[1], 0)

        embed_tokens = VocabParallelEmbedding(
            len(dictionary), embed_dim, dictionary.pad(), init_method=_vocab_init
        )
        return embed_tokens


def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.activation_fn = args.get("activation_fn", "relu")
    args.dropout = args.get("dropout", 0.1)
    args.attention_dropout = args.get("attention_dropout", 0.0)
    args.activation_dropout = args.get("activation_dropout", 0.0)
    args.relu_dropout = args.get("relu_dropout", 0.0)
    args.decoder_embed_dim = args.get("decoder_embed_dim", 512)
    args.decoder_output_dim = args.get("decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = args.get("decoder_input_dim", args.decoder_embed_dim)
    args.decoder_ffn_embed_dim = args.get("decoder_ffn_embed_dim", 2048)
    args.decoder_layers = args.get("decoder_layers", 6)
    args.decoder_attention_heads = args.get("decoder_attention_heads", 8)
    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = args.get("no_decoder_final_norm", False)
    args.adaptive_softmax_cutoff = args.get("adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = args.get("adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = args.get("adaptive_softmax_factor", 4)
    args.no_token_positional_embeddings = args.get("no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = args.get("share_decoder_input_output_embed", False
    )
    args.character_embeddings = args.get("character_embeddings", False)
    args.character_filters = getattr(
        args,
        "character_filters",
        "[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]",
    )
    args.character_embedding_dim = args.get("character_embedding_dim", 4)
    args.char_embedder_highway_layers = args.get("char_embedder_highway_layers", 2)
    args.adaptive_input = args.get("adaptive_input", False)
    args.adaptive_input_factor = args.get("adaptive_input_factor", 4)
    args.adaptive_input_cutoff = args.get("adaptive_input_cutoff", None)
    args.tie_adaptive_weights = args.get("tie_adaptive_weights", False)
    args.tie_adaptive_proj = args.get("tie_adaptive_proj", False)
    args.decoder_learned_pos = args.get("decoder_learned_pos", False)
    args.decoder_layerdrop = args.get("decoder_layerdrop", 0.0)
    args.decoder_layers_to_keep = args.get("decoder_layers_to_keep", None)
    args.layernorm_embedding = args.get("layernorm_embedding", False)
    args.no_scale_embedding = args.get("no_scale_embedding", False)
    args.quant_noise_pq = args.get("quant_noise_pq", 0.0)
    args.quant_noise_pq_block_size = args.get("quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = args.get("quant_noise_scalar", 0.0)
    args.add_bos_token = args.get("add_bos_token", False)


@register_model_architecture("model_parallel_transformer_lm", "transformer_lm_megatron")
def transformer_lm_megatron(args):
    args.decoder_embed_dim = args.get("decoder_embed_dim", 3072)
    args.decoder_ffn_embed_dim = args.get("decoder_ffn_embed_dim", 3072 * 4)
    args.decoder_layers = args.get("decoder_layers", 72)
    args.decoder_attention_heads = args.get("decoder_attention_heads", 32)
    args.dropout = args.get("dropout", 0.1)
    args.attention_dropout = args.get("attention_dropout", 0.1)
    args.activation_fn = args.get("activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture(
    "model_parallel_transformer_lm", "transformer_lm_megatron_11b"
)
def transformer_lm_megatron_11b(args):
    args.decoder_embed_dim = args.get("decoder_embed_dim", 3072)
    args.decoder_ffn_embed_dim = args.get("decoder_ffn_embed_dim", 3072 * 6)
    args.decoder_layers = args.get("decoder_layers", 72)
    args.decoder_attention_heads = args.get("decoder_attention_heads", 32)
    args.dropout = args.get("dropout", 0.1)
    args.attention_dropout = args.get("attention_dropout", 0.1)
    args.activation_fn = args.get("activation_fn", "gelu")
    base_lm_architecture(args)
