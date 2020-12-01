from flax import linen as nn
from typing import (Any, Callable, Tuple, Optional)
import jax
from flax.linen import compact
from jax import lax, random, numpy as jnp
from functools import partial

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


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
        self_attn = SelfAttention(self.num_heads,qkv_features,out_features,name="attention")(inputs,attention_mask)
        return FFNWithNorm(self.intermediate_size)(nn.LayerNorm(name="LayerNorm")(self_attn+inputs))