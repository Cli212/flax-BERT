from transformers import BertTokenizer
from bert import FlaxBertForPretrained
import time
from datasets import load_dataset
from utils import BertConfig, FlaxDataCollatorForLanguageModeling
from jax import numpy as jnp, nn, random
import jax
from flax import jax_utils
from flax.optim import Adam
from flax.training import common_utils
import argparse
import numpy as np
from tqdm import tqdm
import logging
from utils import logger_config



def cross_entropy(logits, targets, weights = None, label_smoothing = 0.0):
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
            confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
    )
    soft_targets = common_utils.onehot(targets,vocab_size)
    loss = -jnp.sum(soft_targets*nn.log_softmax(logits),axis=-1)
    loss -= normalizing_constant
    if weights is not None:
        loss *= weights
        normalizing_factor = weights.sum()
    else:
        normalizing_factor = np.prod(targets.shape,dtype="float32")
    return loss.sum(),normalizing_factor

def train_step(optimizer, inputs, dropout_rng):
    dropout_rng, new_dropout_rng = random.split(dropout_rng)
    def loss_fn(params):
        targets = inputs.pop("labels")
        token_mask = jnp.where(targets>0,1.0,0.0)
        logits = model(**inputs, train=True, dropout_rng = dropout_rng, params = params)[0]
        loss, normalizing_factor = cross_entropy(logits, targets, token_mask)

        return loss/normalizing_factor
    step = optimizer.state.step
    lr = lr_scheduler_fn(step)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(optimizer.target)
    grad = jax.lax.pmean(grad, "batch")
    optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

    return loss, optimizer, new_dropout_rng

def tokenized_function(examples):
    examples = [line for line in examples if len(line)>0]
    return tokenizer(examples, return_special_tokens_mask=True, padding=True, truncation=True, max_length=model._module.max_length)

def generate_batch_splits(samples_idx: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    nb_samples = len(samples_idx)
    samples_to_remove = nb_samples % batch_size

    if samples_to_remove != 0:
        samples_idx = samples_idx[:-samples_to_remove]
    sections_split = nb_samples // batch_size
    batch_idx = np.split(samples_idx, sections_split)
    return batch_idx

def create_learning_rate_scheduler(
    factors="constant * linear_warmup * rsqrt_decay",
    base_learning_rate=0.5,
    warmup_steps=1000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000,
):
    """Creates learning rate schedule.
    Interprets factors in the factors string which can consist of:
    * constant: interpreted as the constant value,
    * linear_warmup: interpreted as linear warmup until warmup_steps,
    * rsqrt_decay: divide by square root of max(step, warmup_steps)
    * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
    * decay_every: Every k steps decay the learning rate by decay_factor.
    * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.
    Args:
      factors: string, factors separated by "*" that defines the schedule.
      base_learning_rate: float, the starting constant for the lr schedule.
      warmup_steps: int, how many steps to warm up for in the warmup schedule.
      decay_factor: float, the amount to decay the learning rate by.
      steps_per_decay: int, how often to decay the learning rate.
      steps_per_cycle: int, steps per cycle when using cosine decay.
    Returns:
      a function learning_rate(step): float -> {"learning_rate": float}, the
      step-dependent lr.
    """
    factors = [n.strip() for n in factors.split("*")]

    def step_fn(step):
        """Step to learning rate function."""
        ret = 1.0
        for name in factors:
            if name == "constant":
                ret *= base_learning_rate
            elif name == "linear_warmup":
                ret *= jnp.minimum(1.0, step / warmup_steps)
            elif name == "rsqrt_decay":
                ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
            elif name == "rsqrt_normalized_decay":
                ret *= jnp.sqrt(warmup_steps)
                ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
            elif name == "decay_every":
                ret *= decay_factor ** (step // steps_per_decay)
            elif name == "cosine_decay":
                progress = jnp.maximum(0.0, (step - warmup_steps) / float(steps_per_cycle))
                ret *= jnp.maximum(0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
            else:
                raise ValueError("Unknown factor %s." % name)
        return jnp.asarray(ret, dtype=jnp.float32)

    return step_fn

def training_step(optimizer, batch, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        targets = batch.pop("labels")

        # Hide away tokens which doesn't participate in the optimization
        token_mask = jnp.where(targets > 0, 1.0, 0.0)

        logits = model(**batch, params=params, dropout_rng=dropout_rng, train=args.train)[0]
        loss, weight_sum = cross_entropy(logits, targets, token_mask)
        return loss / weight_sum

    step = optimizer.state.step
    lr = lr_scheduler_fn(step)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(optimizer.target)
    grad = jax.lax.pmean(grad, "batch")
    optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

    return loss, optimizer, new_dropout_rng


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments of Bert")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--model",type=str,required=True)
    parser.add_argument("--dataset",type=str,default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--train", default=True, type=bool)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--warmup_steps",type=int, default=5)
    args = parser.parse_args()
    logger = logger_config("log.txt", logging_name="log")
    #
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    rng = random.PRNGKey(args.seed)
    data_collator = FlaxDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    model = FlaxBertForPretrained.from_pretrained(args.model)
    optimizer = Adam().create(model.params)
    lr_scheduler_fn = create_learning_rate_scheduler(
        base_learning_rate=1e-3, warmup_steps=min(args.warmup_steps, 1)
    )
    logger.info("Load dataset")
    datasets = load_dataset(args.dataset,args.dataset_config)
    column_names = datasets['train'].column_names if args.train else datasets['validation'].column_names
    ## This place needs a customized setting
    text_column_name = "text" if "text" in column_names else column_names[0]
    tokenized_datasets = datasets.map(tokenized_function, input_columns=[text_column_name], batched=True, remove_columns=column_names)
    p_training_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    optimizer = jax_utils.replicate(optimizer)
    dropout_rngs = random.split(rng, jax.local_device_count())
    logger.info("Start training")
    for epoch in range(3):
        rng, training_rng, eval_rng = random.split(rng,3)
        nb_training_samples = len(tokenized_datasets["train"])
        training_samples_idx = jax.random.permutation(training_rng, jnp.arange(nb_training_samples))
        training_batch_idx = generate_batch_splits(training_samples_idx, args.batch_size)
        for batch_idx in tqdm(training_batch_idx, desc="Training...", position=1):
            samples = [tokenized_datasets["train"][int(idx)] for idx in batch_idx]
            model_inputs = data_collator(samples, pad_to_multiple_of=16)

            # Model forward
            model_inputs = common_utils.shard(model_inputs.data)
            loss, optimizer, dropout_rngs = p_training_step(optimizer, model_inputs, dropout_rngs)

        logger.info(f"Loss: {loss}")
    logger.info(f"Finish traininig with the loss {loss}")

