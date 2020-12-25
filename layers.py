from flax import linen as nn
from typing import (Any, Callable, Tuple, Optional)
import jax
from flax.linen import compact
from jax import lax, random, numpy as jnp
import numpy as np
from functools import partial
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

ACT2FN = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
}

class FlaxBertEmbedding(nn.Module):
    """
    Specify a new class for doing the embedding stuff as Flax's one use 'embedding' for the parameter name and PyTorch
    use 'weight'
    """

    vocab_size: int
    hidden_size: int
    kernel_init_scale: float = 0.2
    emb_init: Callable[..., np.ndarray] = jax.nn.initializers.normal(stddev=kernel_init_scale)
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, inputs):
        embedding = self.param("weight", self.emb_init, (self.vocab_size, self.hidden_size))
        return jnp.take(embedding, inputs, axis=0)

class FlaxBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    vocab_size: int
    hidden_size: int
    type_vocab_size: int
    max_length: int
    kernel_init_scale: float = 0.2
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):

        # Embed
        w_emb = FlaxBertEmbedding(
            self.vocab_size,
            self.hidden_size,
            kernel_init_scale=self.kernel_init_scale,
            name="word_embeddings",
            dtype=self.dtype,
        )(jnp.atleast_2d(input_ids.astype("i4")))
        p_emb = FlaxBertEmbedding(
            self.max_length,
            self.hidden_size,
            kernel_init_scale=self.kernel_init_scale,
            name="position_embeddings",
            dtype=self.dtype,
        )(jnp.atleast_2d(position_ids.astype("i4")))
        t_emb = FlaxBertEmbedding(
            self.type_vocab_size,
            self.hidden_size,
            kernel_init_scale=self.kernel_init_scale,
            name="token_type_embeddings",
            dtype=self.dtype,
        )(jnp.atleast_2d(token_type_ids.astype("i4")))

        # Sum all embeddings
        summed_emb = w_emb + jnp.broadcast_to(p_emb, w_emb.shape) + t_emb

        # Layer Norm
        layer_norm = nn.LayerNorm(name="layer_norm", dtype=self.dtype)(summed_emb)
        embeddings = nn.Dropout(rate=self.dropout_rate)(layer_norm, deterministic=deterministic)
        return embeddings

class FlaxBertAttention(nn.Module):
    num_heads: int
    head_size: int
    dropout_rate: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, hidden_states, attention_mask, deterministic: bool = True):
        # Attention mask comes in as attention_mask.shape == (*batch_sizes, kv_length)
        # FLAX expects: attention_mask.shape == (*batch_sizes, 1, 1, kv_length) such that it is broadcastable
        # with attn_weights.shape == (*batch_sizes, num_heads, q_length, kv_length)
        attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        self_att = SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.head_size,
            out_features = self.head_size,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            kernel_init=jax.nn.initializers.normal(self.kernel_init_scale, self.dtype),
            bias_init=jax.nn.initializers.zeros,
            name="self",
            dtype=self.dtype,
        )(hidden_states, attention_mask)

        layer_norm = nn.LayerNorm(name="layer_norm", dtype=self.dtype)(self_att + hidden_states)
        return layer_norm

class FlaxBertIntermediate(nn.Module):
    output_size: int
    hidden_act: str = "gelu"
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, hidden_states):
        hidden_states = nn.Dense(
            features=self.output_size,
            kernel_init=jax.nn.initializers.normal(self.kernel_init_scale, self.dtype),
            name="dense",
            dtype=self.dtype,
        )(hidden_states)
        hidden_states = ACT2FN[self.hidden_act](hidden_states)
        return hidden_states


class FlaxBertOutput(nn.Module):
    dropout_rate: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, intermediate_output, attention_output, deterministic: bool = True):
        hidden_states = nn.Dense(
            attention_output.shape[-1],
            kernel_init=jax.nn.initializers.normal(self.kernel_init_scale, self.dtype),
            name="dense",
            dtype=self.dtype,
        )(intermediate_output)
        hidden_states = nn.Dropout(rate=self.dropout_rate)(hidden_states, deterministic=deterministic)
        hidden_states = nn.LayerNorm(name="layer_norm", dtype=self.dtype)(hidden_states + attention_output)
        return hidden_states

class FlaxBertLayer(nn.Module):
    num_heads: int
    head_size: int
    intermediate_size: int
    hidden_act: str = "gelu"
    dropout_rate: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, hidden_states, attention_mask, deterministic: bool = True):
        attention = FlaxBertAttention(
            self.num_heads,
            self.head_size,
            kernel_init_scale=self.kernel_init_scale,
            dropout_rate=self.dropout_rate,
            name="attention",
            dtype=self.dtype,
        )(hidden_states, attention_mask, deterministic=deterministic)
        intermediate = FlaxBertIntermediate(
            self.intermediate_size,
            kernel_init_scale=self.kernel_init_scale,
            hidden_act=self.hidden_act,
            name="intermediate",
            dtype=self.dtype,
        )(attention)
        output = FlaxBertOutput(
            kernel_init_scale=self.kernel_init_scale, dropout_rate=self.dropout_rate, name="output", dtype=self.dtype
        )(intermediate, attention, deterministic=deterministic)

        return output

class FlaxBertLayerCollection(nn.Module):
    """
    Stores N BertLayer(s)
    """

    num_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    hidden_act: str = "gelu"
    dropout_rate: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, inputs, attention_mask, deterministic: bool = True):
        assert self.num_layers > 0, f"num_layers should be >= 1, got ({self.num_layers})"

        # Initialize input / output
        input_i = inputs

        # Forward over all encoders
        for i in range(self.num_layers):
            layer = FlaxBertLayer(
                self.num_heads,
                self.head_size,
                self.intermediate_size,
                kernel_init_scale=self.kernel_init_scale,
                dropout_rate=self.dropout_rate,
                hidden_act=self.hidden_act,
                name=f"{i}",
                dtype=self.dtype,
            )
            input_i = layer(input_i, attention_mask, deterministic=deterministic)
        return input_i

class FlaxBertEncoder(nn.Module):
    num_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    hidden_act: str = "gelu"
    dropout_rate: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, hidden_states, attention_mask, deterministic: bool = True):
        layer = FlaxBertLayerCollection(
            self.num_layers,
            self.num_heads,
            self.head_size,
            self.intermediate_size,
            hidden_act=self.hidden_act,
            kernel_init_scale=self.kernel_init_scale,
            dropout_rate=self.dropout_rate,
            name="layer",
            dtype=self.dtype,
        )(hidden_states, attention_mask, deterministic=deterministic)
        return layer


class FlaxBertPooler(nn.Module):
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, hidden_states):
        cls_token = hidden_states[:, 0]
        out = nn.Dense(
            hidden_states.shape[-1],
            kernel_init=jax.nn.initializers.normal(self.kernel_init_scale, self.dtype),
            name="dense",
            dtype=self.dtype,
        )(cls_token)
        return nn.tanh(out)


class FlaxBertPredictionHeadTransform(nn.Module):
    hidden_act: str = "gelu"
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        hidden_states = nn.Dense(hidden_states.shape[-1], name="dense", dtype=self.dtype)(hidden_states)
        hidden_states = ACT2FN[self.hidden_act](hidden_states)
        return nn.LayerNorm(name="layer_norm", dtype=self.dtype)(hidden_states)


class FlaxBertLMPredictionHead(nn.Module):
    vocab_size: int
    hidden_act: str = "gelu"
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        # TODO: The output weights are the same as the input embeddings, but there is
        #   an output-only bias for each token.
        #   Need a link between the two variables so that the bias is correctly
        #   resized with `resize_token_embeddings`

        hidden_states = FlaxBertPredictionHeadTransform(
            name="transform", hidden_act=self.hidden_act, dtype=self.dtype
        )(hidden_states)
        hidden_states = nn.Dense(self.vocab_size, name="decoder", dtype=self.dtype)(hidden_states)
        return hidden_states


class FlaxBertOnlyMLMHead(nn.Module):
    vocab_size: int
    hidden_act: str = "gelu"
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        hidden_states = FlaxBertLMPredictionHead(
            vocab_size=self.vocab_size, hidden_act=self.hidden_act, name="predictions", dtype=self.dtype
        )(hidden_states)
        return hidden_states

class FlaxBertModule(nn.Module):
    vocab_size: int
    hidden_size: int
    type_vocab_size: int
    max_length: int
    num_encoder_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    hidden_act: str = "gelu"
    dropout_rate: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    @nn.compact
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = True):

        # Embedding
        embeddings = FlaxBertEmbeddings(
            self.vocab_size,
            self.hidden_size,
            self.type_vocab_size,
            self.max_length,
            kernel_init_scale=self.kernel_init_scale,
            dropout_rate=self.dropout_rate,
            name="embeddings",
            dtype=self.dtype,
        )(input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic)

        # N stacked encoding layers
        encoder = FlaxBertEncoder(
            self.num_encoder_layers,
            self.num_heads,
            self.head_size,
            self.intermediate_size,
            kernel_init_scale=self.kernel_init_scale,
            dropout_rate=self.dropout_rate,
            hidden_act=self.hidden_act,
            name="encoder",
            dtype=self.dtype,
        )(embeddings, attention_mask, deterministic=deterministic)

        if not self.add_pooling_layer:
            return encoder

        pooled = FlaxBertPooler(kernel_init_scale=self.kernel_init_scale, name="pooler", dtype=self.dtype)(encoder)
        return encoder, pooled

class FlaxBertForMaskedLMModule(nn.Module):
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    head_size: int
    num_heads: int
    num_encoder_layers: int
    type_vocab_size: int
    max_length: int
    hidden_act: str
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, deterministic: bool = True
    ):
        # Model
        encoder = FlaxBertModule(
            vocab_size=self.vocab_size,
            type_vocab_size=self.type_vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            head_size=self.hidden_size,
            num_heads=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            max_length=self.max_length,
            dropout_rate=self.dropout_rate,
            hidden_act=self.hidden_act,
            dtype=self.dtype,
            add_pooling_layer=False,
            name="bert",
        )(input_ids, attention_mask, token_type_ids, position_ids, deterministic=deterministic)

        # Compute the prediction scores
        encoder = nn.Dropout(rate=self.dropout_rate)(encoder, deterministic=deterministic)
        logits = FlaxBertOnlyMLMHead(
            vocab_size=self.vocab_size, hidden_act=self.hidden_act, name="cls", dtype=self.dtype
        )(encoder)

        return (logits,)








# class TransformerEncoderBlock(nn.Module):
#     num_heads: int

# This class only supports encoding operation now
class MultiHeadDotProductAttention(nn.Module):
    num_heads: int
    qkv_features: int
    out_features: int
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.linear.default_kernel_init
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.
    deterministic: bool = False

    def setup(self):
        assert self.qkv_features % self.num_heads == 0
        head_dim = self.qkv_features // self.num_heads
        self.dense = partial(nn.DenseGeneral,
                        axis=-1,
                        features=(self.num_heads, head_dim),
                        kernel_init=self.kernel_init,
                        bias_init=self.bias_init,
                        dtype=self.dtype,
                        name="dense"
                        )

    def __call__(self, inputs_q: Array,inputs_kv: Array, mask: Optional[Array] = None):
        query = self.dense(name="query")(inputs_q)
        key = self.dense(name="key")(inputs_kv)
        value = self.dense(name="value")(inputs_kv)
        if mask is not None:
            attention_bias = lax.select(mask>0,
                                        jnp.full(mask.shape,0).astype(self.dtype),
                                        jnp.full(mask.shape,-1e10).astype(self.dtype))
        else:
            attention_bias = None
        dropout_rng = None
        if not self.deterministic and self.dropout_rate > 0:
            dropout_rng = self.make_rng("dropout")
        x = nn.attention.dot_product_attention(query, key, value,
                                               bias = attention_bias,
                                               dropout_rng = dropout_rng,
                                               dropout_rate = self.dropout_rate,
                                               deterministic = self.deterministic,
                                               dtype = self.dtype)
        output = nn.DenseGeneral(features = self.out_features,
                                 axis = (-2,-1),
                                 dtype = self.dtype,
                                 name = "out")(x)
        return output



class SelfAttention(MultiHeadDotProductAttention):
    @compact
    def __call__(self, inputs: Array, mask: Optional[Array] = None):
        return super().__call__(inputs,inputs,mask)

class InitEmbedding(nn.Module):
    vocab_size: int
    hidden_size: int
    emb_init: Callable[...,jnp.ndarray] = nn.initializers.normal()
    @compact
    def __call__(self, inputs):
        embeddings = self.param("weight",self.emb_init,(self.vocab_size,self.hidden_size))
        return jnp.take(embeddings,inputs,0)

class FFNWithNorm(nn.Module):
    hidden_size: int
    act_fn: Optional = None
    @compact
    def __call__(self,inputs):
        act_fn = self.act_fn or nn.gelu
        middle = act_fn(nn.Dense(self.hidden_size,name="intermediate.dense")(inputs))
        last = nn.Dense(inputs.shape[-1],name="output.dense")(middle)
        return nn.LayerNorm(name="LayerNorm")(inputs+last)

from flax.linen import SelfAttention

class BertEncoderBlock(nn.Module):
    vocab_size: int
    hidden_size: int
    type_vocab_size: int
    max_length: int
    num_heads: int
    intermediate_size: int
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    @compact
    def __call__(self, inputs, attention_mask):
        qkv_features = self.qkv_features or inputs.shape[-1]
        out_features = self.out_features or inputs.shape[-1]
        self_attn = SelfAttention(self.num_heads,qkv_features = qkv_features,out_features = out_features,name="attention")(inputs,attention_mask)
        return FFNWithNorm(self.intermediate_size)(nn.LayerNorm(name="LayerNorm")(self_attn+inputs))