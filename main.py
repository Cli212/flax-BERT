from transformers import BertTokenizer
import bert
from bert import FlaxBertForPretrained
import time
import sys
from utils import BertConfig, FlaxDataCollatorForLanguageModeling
from jax import numpy as jnp, nn, random
import jax
from flax.optim import Adam
from flax.training import common_utils


def cross_entropy(logits, targets, weights = None):
    vocab_size = logits.shape[-1]
    soft_targets = common_utils.onehot(targets,vocab_size)
    loss = -jnp.sum(soft_targets*nn.log_softmax(logits),axis=-1)
    if weights is not None:
        loss *= weights
        normalizing_factor = weights.sum()
    else:
        normalizing_factor = jnp.prod(targets.shape)
    return loss.sum(),normalizing_factor

def train_step(optimizer, inputs, dropout_rng):
    dropout_rng, new_dropout_rng = random.split(dropout_rng)
    def loss_fn(params):
        targets = inputs.pop("labels")
        token_mask = jnp.where(targets>0,1.0,0.0)
        logits = model(**inputs,train=True)[0]
        loss, normalizing_factor = cross_entropy(logits, targets, token_mask)
        return loss/normalizing_factor

    step = optimizer.state.step
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(optimizer.target)
    grad = jax.lax.pmean(grad, "batch")
    optimizer = optimizer.apply_gradient(grad, learning_rate=1e-3)

    return loss, optimizer, new_dropout_rng
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data_collator = FlaxDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    model = FlaxBertForPretrained.from_pretrained("../models/bert-base-uncased")
    optimizer = Adam().create(model.params)

    for epoch in range(10):
        rng, training_rng, eval_rng = random.split(rng,3)


