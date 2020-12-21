from transformers import BertTokenizer
from bert import FlaxBertForPretrained
from datasets import load_dataset
from flax.training import common_utils
import jax
from flax import jax_utils
from flax.optim import Adam
import argparse
from tqdm import tqdm
from utils import logger_config, cross_entropy, generate_batch_splits, create_learning_rate_scheduler, FlaxDataCollatorForLanguageModeling
from jax import numpy as jnp, random
import ast


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments of Bert")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--model",type=str,required=True)
    parser.add_argument("--dataset",type=str,default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-v1")
    parser.add_argument("--train", default=False, type=ast.literal_eval)
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
    text_column_name = "review_body" if "review_body" in column_names else column_names[0]
    tokenized_datasets = datasets.map(tokenized_function, input_columns=[text_column_name], batched=True, remove_columns=column_names)
    p_training_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    optimizer = jax_utils.replicate(optimizer)
    dropout_rngs = random.split(rng, jax.local_device_count())
    if args.train:
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
                ## the error jumps out here, with
                loss, optimizer, dropout_rngs = p_training_step(optimizer, model_inputs, dropout_rngs)

            logger.info(f"Loss: {loss}")
        logger.info(f"Finish traininig with the loss {loss}")
    else:
        rng, training_rng, eval_rng = random.split(rng, 3)
        nb_training_samples = len(tokenized_datasets["train"])
        training_samples_idx = jnp.arange(nb_training_samples)
        training_batch_idx = generate_batch_splits(training_samples_idx, args.batch_size)
        for batch_idx in tqdm(training_batch_idx, desc="Evaluating...", position=1):
            samples = [tokenized_datasets["train"][int(idx)] for idx in batch_idx]
            model_inputs = data_collator(samples, pad_to_multiple_of=16)

            # Model forward
            model_inputs = common_utils.shard(model_inputs.data)
            ## the error jumps out here, with
            labels = model_inputs.pop("labels")
            outputs = model(**model_inputs,train=False)[0]
            print(outputs)
