import jax
from jax.random import PRNGKey
from flax.traverse_util import unflatten_dict
from jax import lax, random, numpy as jnp
from flax.core.frozen_dict import freeze, FrozenDict
import layers
from layers import FlaxBertModule, FlaxBertOnlyMLMHead, FlaxBertForMaskedLMModule
import numpy as np
from flax import linen as nn
from flax.linen import compact
from typing import Callable, Optional, Dict, Tuple
import json
import os
import logging
from transformers import TensorType, BertConfig


WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
CONFIG_NAME = "config.json"
MODEL_CARD_NAME = "modelcard.json"
logger = logging.getLogger(__name__)

class BertModule(nn.Module):
    """
    This model is the jax version Bert Model which accepts tokenized inputs
    and returns the result. The result is the masked prediction scores which
    has a size [batch_size, length, vocab_size]
    """
    vocab_size: int
    hidden_size: int
    type_vocab_size: int
    max_length: int
    num_encoder_layers: int
    num_heads: int
    intermediate_size: Optional[int] = None
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    hidden_act: Optional = None

    def setup(self):
        self.cls = BertOnlyMLMHead(self.hidden_size, self.vocab_size, self.hidden_size)

    @compact
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask):
        input_ids = jnp.array(input_ids,dtype="i4")
        position_ids = jnp.array(position_ids,dtype="i4")
        token_type_ids = jnp.array(token_type_ids,dtype="i4")
        attention_mask = jnp.array(attention_mask,dtype="i4")
        embeddings = BertEmbedding(self.vocab_size, self.max_length, self.type_vocab_size, self.hidden_size,
                                   name="embeddings")(input_ids,
                                                      position_ids,
                                                      token_type_ids)
        output = embeddings
        qkv_features = self.qkv_features or embeddings.shape[-1]
        out_features = self.out_features or embeddings.shape[-1]
        intermediate_size = self.intermediate_size or 4 * out_features
        self.hidden_act = self.hidden_act or nn.gelu
        for i in range(self.num_encoder_layers):
            output = layers.BertEncoderBlock(vocab_size=self.vocab_size,
                                             hidden_size=self.hidden_size,
                                             type_vocab_size=self.type_vocab_size,
                                             max_length=self.max_length,
                                             num_heads=self.num_heads,
                                             intermediate_size=intermediate_size,
                                             qkv_features=qkv_features,
                                             out_features=out_features, name=f"encoder.layer.{i}")(output,
                                                                                                   attention_mask)

        return self.cls(output)


class BertEmbedding(nn.Module):
    """
    This class is the embedding layer of BERT. With the word_embeddings,
    positional_embeddings and token_embeddings.
    """
    vocab_size: int
    max_length: int
    type_vocab_size: int
    hidden_size: int
    emb_init: Callable[..., jnp.ndarray] = nn.initializers.normal()

    @compact
    def __call__(self, token_ids, position_ids, token_type_ids):
        w_embeddings = layers.InitEmbedding(self.vocab_size, self.hidden_size, self.emb_init, name="word_embeddings")(
            jnp.atleast_2d(token_ids.astype("i4")))
        p_embeddings = layers.InitEmbedding(self.max_length, self.hidden_size, self.emb_init,
                                            name="position_embeddings")(jnp.atleast_2d(position_ids.astype("i4")))
        t_embeddings = layers.InitEmbedding(self.type_vocab_size, self.hidden_size, self.emb_init,
                                            name="token_type_embeddings")(jnp.atleast_2d(token_type_ids.astype("i4")))
        # w_embeddings = nn.Embed(self.vocab_size, self.hidden_size, embedding_init = self.emb_init, name="word_embeddings")(
        #     jnp.atleast_2d(token_ids.astype("i4")))
        # p_embeddings = nn.Embed(self.max_length, self.hidden_size, embedding_init = self.emb_init,
        #                                     name="position_embeddings")(jnp.atleast_2d(position_ids.astype("i4")))
        # t_embeddings = nn.Embed(self.type_vocab_size, self.hidden_size, embedding_init = self.emb_init,
        #                                     name="token_type_embeddings")(jnp.atleast_2d(token_type_ids.astype("i4")))
        sumed_emb = w_embeddings + p_embeddings + t_embeddings
        norm = nn.LayerNorm(name="LayerNorm")
        return norm(sumed_emb)


class FlaxBertPooler(nn.Module):
    @nn.compact
    def __call__(self, hidden_state):
        cls_token = hidden_state[:, 0]
        out = nn.Dense(hidden_state.shape[-1], name="dense")(cls_token)
        return jax.lax.tanh(out)


BERT_INPUTS_DOCSTRING = r""


class FlaxBertForPretrained(object):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    """

    model_class = BertModule
    config_class = BertConfig
    base_model_prefix = "bert"

    @classmethod
    def from_pretrained(cls,pretrained_model_path:str,*model_args,**kwargs):

        r"""Instantiate a pretrained Flax model from a pre-trained model configuration."""
        config = BertConfig.from_pretrained(pretrained_model_path)
        if pretrained_model_path is not None:
            if os.path.isdir(pretrained_model_path):
                if os.path.isfile(os.path.join(pretrained_model_path, WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        "Error no file named {} found in directory {} or `from_tf` set to False".format(
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                            pretrained_model_path,
                        )
                    )
            elif os.path.isfile(pretrained_model_path):
                archive_file = pretrained_model_path
            else:
                raise RuntimeError("Unrecognized file path")
        else:
            raise BaseException("Empty model path")
        logger.info(f"loading weights file from {archive_file}")
        with open(archive_file, "rb") as f:
            try:
                import torch
                state = torch.load(f)
                state = {k: v.numpy() for k, v in state.items()}
                state = cls.convert_from_pytorch(state, config)
                state = unflatten_dict({tuple(k.split(".")[0:]): v for k, v in state.items()})
            except BaseException:
                raise EnvironmentError(
                    f"Unable to convert model {archive_file} to Flax deserializable object. "
                    "Supported format are PyTorch archive or Flax msgpack"
                )
        return cls(config,state,*model_args,**kwargs)

    @classmethod
    def _dict_from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    # @staticmethod
    # def convert_from_pytorch(pt_state: Dict, config: BertConfig) -> Dict:
    #     jax_state = dict(pt_state)
    #
    #     # Need to change some parameters name to match Flax names so that we don't have to fork any layer
    #     for key, tensor in pt_state.items():
    #         # Key parts
    #         key_parts = set(key.split("."))
    #
    #         if "attention.self" in key:
    #             del jax_state[key]
    #             key = key.replace("attention.self", "attention")
    #             # tensor = jnp.asarray(tensor)
    #             jax_state[key] = tensor
    #
    #         # Every dense layer has "kernel" parameters instead of "weight"
    #         if "dense.weight" in key:
    #             del jax_state[key]
    #             key = key.replace("weight", "kernel")
    #             # tensor = jnp.asarray(tensor)
    #             jax_state[key] = tensor
    #
    #         # SelfAttention needs also to replace "weight" by "kernel"
    #         if {"query", "key", "value"} & key_parts:
    #
    #             # Flax SelfAttention decomposes the heads (num_head, size // num_heads)
    #             if "bias" in key:
    #                 jax_state[key] = tensor.reshape((config.num_attention_heads, -1))
    #             elif "weight" in key:
    #                 del jax_state[key]
    #                 key = key.replace("weight", "kernel")
    #                 tensor = tensor.reshape((config.num_attention_heads, -1, config.hidden_size)).transpose((2, 0, 1))
    #                 jax_state[key] = tensor
    #
    #         # SelfAttention output is not a separate layer, remove one nesting
    #         # if "attention.output.dense" in key:
    #         #     del jax_state[key]
    #         #     key = key.replace("attention.output.dense", "attention.out")
    #         #     jax_state[key] = tensor
    #
    #         # SelfAttention output is not a separate layer, remove nesting on layer norm
    #         # if "attention.output.LayerNorm" in key:
    #         #     del jax_state[key]
    #         #     key = key.replace("attention.output.LayerNorm", "LayerNorm")
    #         #     jax_state[key] = tensor
    #
    #         # There are some transposed parameters w.r.t their PyTorch counterpart
    #         # TBD
    #
    #         if "intermediate.dense" in key:
    #             del jax_state[key]
    #             key = key.replace("intermediate.dense", "FFNWithNorm_0.intermediate.dense")
    #             jax_state[key] = tensor.T
    #
    #         # Self Attention output projection needs to be transposed
    #
    #         # Pooler needs to transpose its kernel
    #         if "pooler.dense.kernel" in key:
    #             jax_state[key] = tensor.T
    #
    #         # Handle LayerNorm conversion
    #         if "LayerNorm" in key:
    #             del jax_state[key]
    #
    #             # # Replace LayerNorm by layer_norm
    #             # new_key = key.replace("LayerNorm", "layer_norm")
    #             if "gamma" in key:
    #                 key = key.replace("gamma", "scale")
    #                 # new_key = key.replace("weight","gamma")
    #             elif "beta" in key:
    #                 key = key.replace("beta", "bias")
    #                 # new_key = key.replace("bias","beta")
    #
    #             jax_state[key] = tensor
    #
    #         if "output.LayerNorm" in key:
    #             del jax_state[key]
    #             if "attention" in key:
    #                 key = key.replace("attention.output.LayerNorm", "LayerNorm")
    #             else:
    #                 key = key.replace("output.LayerNorm", "FFNWithNorm_0.LayerNorm")
    #             jax_state[key] = tensor
    #
    #         if "output.dense" in key:
    #             del jax_state[key]
    #             if "attention" in key:
    #                 key = key.replace("attention.output.dense", "attention.out")
    #                 jax_state[key] = tensor
    #             else:
    #                 key = key.replace("output.dense", "FFNWithNorm_0.output.dense")
    #                 jax_state[key] = tensor.T
    #
    #         if "out.kernel" in key:
    #             jax_state[key] = tensor.reshape((config.hidden_size, config.num_attention_heads, -1)).transpose(
    #                 1, 2, 0
    #             )
    #         if "decoder.weight" in key:
    #             del jax_state[key]
    #             key = key.replace("weight", "kernel")
    #             jax_state[key] = tensor
    #
    #     return jax_state

    @staticmethod
    def convert_from_pytorch(pt_state: Dict, config: BertConfig) -> Dict:
        jax_state = dict(pt_state)

        # Need to change some parameters name to match Flax names so that we don't have to fork any layer
        for key, tensor in pt_state.items():
            # Key parts
            key_parts = set(key.split("."))
            # Every dense layer has "kernel" parameters instead of "weight"
            if "dense.weight" in key:
                del jax_state[key]
                key = key.replace("weight", "kernel")
                jax_state[key] = tensor

            if "decoder.weight" in key:
                del jax_state[key]
                key = key.replace("weight", "kernel")
                jax_state[key] = tensor.T

            # SelfAttention needs also to replace "weight" by "kernel"
            if {"query", "key", "value"} & key_parts:

                # Flax SelfAttention decomposes the heads (num_head, size // num_heads)
                if "bias" in key:
                    jax_state[key] = tensor.reshape((config.num_attention_heads, -1))
                elif "weight":
                    del jax_state[key]
                    key = key.replace("weight", "kernel")
                    tensor = tensor.reshape((config.num_attention_heads, -1, config.hidden_size)).transpose((2, 0, 1))
                    jax_state[key] = tensor

            # SelfAttention output is not a separate layer, remove one nesting
            if "attention.output.dense" in key:
                del jax_state[key]
                key = key.replace("attention.output.dense", "attention.self.out")
                jax_state[key] = tensor

            # SelfAttention output is not a separate layer, remove nesting on layer norm
            if "attention.output.LayerNorm" in key:
                del jax_state[key]
                key = key.replace("attention.output.LayerNorm", "attention.LayerNorm")
                jax_state[key] = tensor

            # There are some transposed parameters w.r.t their PyTorch counterpart
            if "intermediate.dense.kernel" in key or "output.dense.kernel" in key or "transform.dense.kernel" in key:
                jax_state[key] = tensor.T

            # Self Attention output projection needs to be transposed
            if "out.kernel" in key:
                jax_state[key] = tensor.reshape((config.hidden_size, config.num_attention_heads, -1)).transpose(
                    1, 2, 0
                )

            # Pooler needs to transpose its kernel
            if "pooler.dense.kernel" in key:
                jax_state[key] = tensor.T

            # Hack to correctly load some pytorch models
            if "predictions.bias" in key:
                del jax_state[key]
                jax_state[".".join(key.split(".")[:2]) + ".decoder.bias"] = tensor

            # Handle LayerNorm conversion
            if "LayerNorm" in key:
                del jax_state[key]

                # Replace LayerNorm by layer_norm
                new_key = key.replace("LayerNorm", "layer_norm")

                if "gamma" in key:
                    new_key = new_key.replace("gamma", "scale")
                elif "beta" in key:
                    new_key = new_key.replace("beta", "bias")

                jax_state[new_key] = tensor

        return jax_state

    def __init__(self, config: BertConfig, state: dict, seed: int = 0, **kwargs):
        # for i in range(config.num_hidden_layers):
        #     state[f"encoder.layer.{i}"] = state["encoder"]["layer"][str(i)]
        #     state[f"encoder.layer.{i}"]["FFNWithNorm_0"]["intermediate.dense"] \
        #         = state[f"encoder.layer.{i}"]["FFNWithNorm_0"]["intermediate"]["dense"]
        #     state[f"encoder.layer.{i}"]["FFNWithNorm_0"]["output.dense"] \
        #         = state[f"encoder.layer.{i}"]["FFNWithNorm_0"]["output"]["dense"]
        # state["cls"] = {}
        # state["cls"]["predictions"] = state["predictions"]
        model = FlaxBertForMaskedLMModule(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            type_vocab_size=config.type_vocab_size,
            max_length=config.max_position_embeddings,
            num_encoder_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            head_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout_rate=config.hidden_dropout_prob,
            hidden_act=config.hidden_act
        )

        if config is None:
            raise ValueError("config cannot be None")

        if state is None:
            raise ValueError("state cannot be None")

        # Those are private to be exposed as typed property on derived classes.
        self._config = config
        self._module = model

        # Those are public as their type is generic to every derived classes.
        self.key = PRNGKey(seed)
        self.params = state

    def init(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            jnp.zeros(input_shape, dtype="i4"), None, None, None
        )

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        return self._module.init(rngs, input_ids, attention_mask, token_type_ids, position_ids)["params"]

    @property
    def module(self) -> nn.Module:
        return self._module

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,dropout_rng=None,train=False, params = None):
        if token_type_ids is None:
            token_type_ids = jnp.ones_like(input_ids)

        if position_ids is None:
            position_ids = jnp.arange(jnp.atleast_2d(input_ids).shape[-1])

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        params_input = freeze({"params": params or self.params})
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self._module.apply(
            params_input,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            rngs=rngs,
        )


ACT2FN = {"gelu": nn.gelu,
          "relu": nn.relu,
          "swish": nn.swish}


class BertLMPredictionHeadTransform(nn.Module):
    hidden_size: int
    act_fn_name: Optional

    @compact
    def __call__(self, inputs):
        if isinstance(self.act_fn_name, str):
            act_fn = ACT2FN[self.act_fn_name]
        else:
            act_fn = nn.gelu
        dense = nn.DenseGeneral(features=self.hidden_size, axis=-1, name="dense")
        norm = nn.LayerNorm(name="LayerNorm")
        return norm(act_fn(dense(inputs)))


class BertLMPredictionHead(nn.Module):
    hidden_size: int
    vocab_size: int
    act_fn_name: Optional

    def setup(self):
        self.transform = BertLMPredictionHeadTransform(self.hidden_size, self.act_fn_name)
        # self.bias =

    @compact
    def __call__(self, inputs):
        decoder = nn.Dense(features=self.vocab_size, name="decoder", use_bias=False)
        return decoder(self.transform(inputs))


class BertOnlyMLMHead(nn.Module):
    hidden_size: int
    vocab_size: int
    hidden_act: Optional = None

    def setup(self):
        self.predictions = BertLMPredictionHead(self.hidden_size, self.vocab_size, self.hidden_act)

    @compact
    def __call__(self, inputs):
        scores = self.predictions(inputs)
        return scores


class FlaxBertPredictMask:
    """This model is used for get the prediction result of MLM with the built
    model and tokenizer"""

    def __init__(self, model: FlaxBertForPretrained, tokenizer, top_k=5):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k

    def __call__(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors=TensorType.JAX)
        logits = self.model(**inputs)[0]
        scores = nn.softmax(logits, axis=-1)
        # values, predictions = scores.topk(self.top_k)
        values, predictions = lax.top_k(scores, self.top_k)
        masked_index = jnp.nonzero(inputs["input_ids"] == self.tokenizer.mask_token_id)
        result = []
        for i in range(masked_index[0].shape[0]):
            for j in range(self.top_k):
                tokens = np.asarray(inputs["input_ids"]).copy()
                x = int(masked_index[0][i])
                y = int(masked_index[1][i])
                tokens[x][y] = predictions[x][y][j]
                tokens = tokens[jnp.where(tokens != self.tokenizer.pad_token_id)]
                result.append(
                    {
                        "sequence": self.tokenizer.decode(tokens),
                        "score": values[x][y][j],
                        "token": predictions[x][y][j],
                        "token_str": self.tokenizer.convert_ids_to_tokens([predictions[x][y][j]]),
                    }
                )
        return result
