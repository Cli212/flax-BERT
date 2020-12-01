import jax
from jax import lax, random, numpy as jnp
from flax.core.frozen_dict import freeze
import layers
import numpy as np
from flax import linen as nn
from flax.linen import compact
from typing import Callable, Optional, Dict
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers import TensorType,BertConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward

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
    def __call__(self, token_ids, position_ids, token_type_ids, attention_mask):
        embeddings = BertEmbedding(self.vocab_size, self.max_length, self.type_vocab_size, self.hidden_size,name="embeddings")(token_ids,
                                                                                                             position_ids,
                                                                                                             token_type_ids)
        output = embeddings
        qkv_features = self.qkv_features or embeddings.shape[-1]
        out_features = self.out_features or embeddings.shape[-1]
        intermediate_size = self.intermediate_size or 4*out_features
        self.hidden_act = self.hidden_act or nn.gelu
        for i in range(self.num_encoder_layers):
            output = layers.BertEncoderBlock(vocab_size = self.vocab_size,
                                    hidden_size = self.hidden_size,
                                    type_vocab_size = self.type_vocab_size,
                                    max_length = self.max_length,
                                    num_heads = self.num_heads,
                                    intermediate_size = intermediate_size,
                                    qkv_features = qkv_features,
                                    out_features = out_features,name=f"encoder.layer.{i}")(output,attention_mask)

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
    emb_init: Callable[...,jnp.ndarray] = nn.initializers.normal()
    @compact
    def __call__(self, token_ids,position_ids,token_type_ids):
        w_embeddings = layers.InitEmbedding(self.vocab_size,self.hidden_size,self.emb_init,name = "word_embeddings")(jnp.atleast_2d(token_ids.astype("i4")))
        p_embeddings = layers.InitEmbedding(self.max_length,self.hidden_size,self.emb_init,name="position_embeddings")(jnp.atleast_2d(position_ids.astype("i4")))
        t_embeddings = layers.InitEmbedding(self.type_vocab_size,self.hidden_size,self.emb_init,name="token_type_embeddings")(jnp.atleast_2d(token_type_ids.astype("i4")))
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

class FlaxBertForMaskedLM(FlaxPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    """

    model_class = BertModule
    config_class = BertConfig
    base_model_prefix = "bert"

    @staticmethod
    def convert_from_pytorch(pt_state: Dict, config: BertConfig) -> Dict:
        jax_state = dict(pt_state)

        # Need to change some parameters name to match Flax names so that we don't have to fork any layer
        for key, tensor in pt_state.items():
            # Key parts
            key_parts = set(key.split("."))

            if "attention.self" in key:
                del jax_state[key]
                key = key.replace("attention.self","attention")
                jax_state[key] = jax.numpy.asarray(tensor)

            # Every dense layer has "kernel" parameters instead of "weight"
            if "dense.weight" in key:
                del jax_state[key]
                key = key.replace("weight", "kernel")
                jax_state[key] = jax.numpy.asarray(tensor)

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
            # if "attention.output.dense" in key:
            #     del jax_state[key]
            #     key = key.replace("attention.output.dense", "attention.out")
            #     jax_state[key] = tensor

            # SelfAttention output is not a separate layer, remove nesting on layer norm
            # if "attention.output.LayerNorm" in key:
            #     del jax_state[key]
            #     key = key.replace("attention.output.LayerNorm", "LayerNorm")
            #     jax_state[key] = tensor

            # There are some transposed parameters w.r.t their PyTorch counterpart
            # TBD

            if "intermediate.dense" in key:
                del jax_state[key]
                key = key.replace("intermediate.dense","FFNWithNorm_0.intermediate.dense")
                jax_state[key] = tensor.T


            # Self Attention output projection needs to be transposed


            # Pooler needs to transpose its kernel
            if "pooler.dense.kernel" in key:
                jax_state[key] = tensor.T

            # Handle LayerNorm conversion
            if "LayerNorm" in key:
                del jax_state[key]

                # # Replace LayerNorm by layer_norm
                # new_key = key.replace("LayerNorm", "layer_norm")
                if "gamma" in key:
                    key = key.replace("gamma", "scale")
                    # new_key = key.replace("weight","gamma")
                elif "beta" in key:
                    key = key.replace("beta", "bias")
                    # new_key = key.replace("bias","beta")

                jax_state[key] = tensor

            if "output.LayerNorm" in key:
                del jax_state[key]
                if "attention" in key:
                    key = key.replace("attention.output.LayerNorm", "LayerNorm")
                else:
                    key = key.replace("output.LayerNorm","FFNWithNorm_0.LayerNorm")
                jax_state[key] = tensor

            if "output.dense" in key:
                del jax_state[key]
                if "attention" in key:
                    key = key.replace("attention.output.dense", "attention.out")
                    jax_state[key] = tensor
                else:
                    key = key.replace("output.dense","FFNWithNorm_0.output.dense")
                    jax_state[key] = tensor.T

            if "out.kernel" in key:
                jax_state[key] = tensor.reshape((config.hidden_size, config.num_attention_heads, -1)).transpose(
                    1, 2, 0
                )
            if "decoder.weight" in key:
                del jax_state[key]
                key = key.replace("weight","kernel")
                jax_state[key] = tensor.T

        return jax_state

    def __init__(self, config: BertConfig, state: dict, seed: int = 0, **kwargs):
        for i in range(config.num_hidden_layers):
            state[f"encoder.layer.{i}"] = state["encoder"]["layer"][str(i)]
            state[f"encoder.layer.{i}"]["FFNWithNorm_0"]["intermediate.dense"] \
                = state[f"encoder.layer.{i}"]["FFNWithNorm_0"]["intermediate"]["dense"]
            state[f"encoder.layer.{i}"]["FFNWithNorm_0"]["output.dense"] \
                = state[f"encoder.layer.{i}"]["FFNWithNorm_0"]["output"]["dense"]
        state["cls"] = {}
        state["cls"]["predictions"] = state["predictions"]
        model = BertModule(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            type_vocab_size=config.type_vocab_size,
            max_length=config.max_position_embeddings,
            num_encoder_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act
        )

        super().__init__(config, model, state, seed)

    @property
    def module(self) -> nn.Module:
        return self._module

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        if token_type_ids is None:
            token_type_ids = jnp.ones_like(input_ids)

        if position_ids is None:
            position_ids = jnp.arange(jnp.atleast_2d(input_ids).shape[-1])

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        params_input = freeze({"params": self.params})
        return self.model.apply(
            params_input,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
        )

ACT2FN = {"gelu":nn.gelu,
          "relu":nn.relu,
          "swish":nn.swish}

class BertLMPredictionHeadTransform(nn.Module):
    hidden_size: int
    act_fn_name: Optional
    @compact
    def __call__(self, inputs):
        if isinstance(self.act_fn_name,str):
            act_fn = ACT2FN[self.act_fn_name]
        else:
            act_fn = nn.gelu
        dense = nn.DenseGeneral(features=self.hidden_size,axis=-1,name="dense")
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
    def __call__(self,inputs):
        decoder = nn.DenseGeneral(features=self.vocab_size,axis=-1,name="decoder",use_bias = False)
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
    def __init__(self,model,tokenizer,top_k=5):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
    def __call__(self, sentence):
        inputs = self.tokenizer(sentence,return_tensors=TensorType.JAX)
        logits = self.model(**inputs)
        scores = nn.softmax(logits, axis=-1)
        # values, predictions = scores.topk(self.top_k)
        values,predictions = lax.top_k(scores,self.top_k)
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




