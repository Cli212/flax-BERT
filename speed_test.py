from bert import FlaxBertForPretrained
from utils import cross_entropy
import jax
import numpy as np
from jax import numpy as jnp
import argparse
import csv
import timeit
from flax.optim import Momentum
from tqdm import tqdm
import ast

class FlaxBertBenchmark:
    def __init__(self, workload, model_path = None, use_gpu = True, layout="NT", jit=True, repeat = 3):
        self.workload = workload
        self.model_path = model_path if model_path else "models/bert-base-uncased"
        self.model = FlaxBertForPretrained.from_pretrained(model_path)
        self.vocab_size = self.model._module.vocab_size
        self.layout = layout
        self.jit = jit
        self.repeat = repeat
        self.optimizer = Momentum().create(self.model.params)



    def train_speed_memory(self, batch_size, seq_length):
        input_ids = np.random.randint(0, self.vocab_size, (batch_size, seq_length))
        targets = np.random.randint(0, self.vocab_size, (batch_size, seq_length))
        labels = np.random.randint(0,2, (batch_size, seq_length))
        optimizer = Momentum().create(self.model.params)
        @jax.jit
        def train_step():
            def loss_fn(params):
                token_mask = jnp.where(labels > 0, 1.0, 0.0)
                logits = self.model(input_ids=input_ids, train=True, params=params, dropout_rng=jax.random.PRNGKey(0))[0]
                loss, normalizing_factor = cross_entropy(logits,targets, token_mask)
                return loss / normalizing_factor

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grad = grad_fn(self.model.params)
            # grad = jax.lax.pmean(grad, "batch")
            # optimizer = optimizer.apply_gradient(grad, learning_rate = 0.01)
            return grad
        timeit.repeat(train_step, repeat=1, number=2)
        if self.jit:
            runtimes = timeit.repeat(train_step, repeat=self.repeat, number=3)
        else:
            with jax.disable_jit():
                runtimes = timeit.repeat(train_step,repeat=self.repeat,number=3)
        # optimizer = train_step(optimizer).block_until_ready()
        return float(np.min(runtimes)/3.0)

    def inference_speed_memory(self, batch_size, seq_length):
        input_ids = np.random.randint(0, self.vocab_size, (batch_size, seq_length))
        @jax.jit
        def ref_step():
            out = self.model(input_ids = input_ids)
            return out
        timeit.repeat(ref_step, repeat=1, number=2)
        if self.jit:
            runtimes = timeit.repeat(ref_step, repeat=self.repeat,number=3)
        else:
            with jax.disable_jit():
                runtimes = timeit.repeat(ref_step,repeat=self.repeat,number=3)
        return float(np.min(runtimes)/3.0)


    def run(self, is_train):
        if self.layout == "NT":
            batch_size, seq_length = self.workload
        elif self.layout == "TN":
            seq_length, batch_size = self.workload
        else:
            raise ValueError("Invalid Layout")
        if is_train:
            runtimes = self.train_speed_memory(batch_size, seq_length)
        else:
            runtimes = self.inference_speed_memory(batch_size, seq_length)
        return runtimes



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat",default=3, type=int)
    parser.add_argument("--layout",default="NT",type=str)
    parser.add_argument("--csv_file",default="flax_time_test.csv",type=str)
    parser.add_argument("--train",default=True,type=ast.literal_eval)
    args = parser.parse_args()
    csv_file = args.csv_file
    layout = args.layout
    repeat = args.repeat
    workload_dict = {"train":[(4,128),
                 (8,128),
                 (16,128),
                 (32,128),
                 (1,512),
                 (2,512),
                 (4,512),
                 (8,512)
                 ],"infer":[(1,128),
                 (1,384),
                 (1,512),
                 (8,32),
                 (8,128),
                 (8,512),
                 (32,512),
                 (256,128),
                 (400,100)
                 ]}
    workloads = workload_dict["train"] if args.train else workload_dict["infer"]
    train_result = []
    fieldnames = ["model_name", "batch_size", "sequence_length","use_jit"]
    with open(csv_file, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames + ["latency"])
        writer.writeheader()

        for workload in tqdm(workloads):
            if layout == "NT":
                batch_size, seq_length = workload
            elif layout == "TN":
                seq_length, batch_size = workload
            else:
                raise ValueError("Invalid Layout")
            for flag in [True]:
                benchmark = FlaxBertBenchmark(workload, model_path="../models/bert-base-uncased", jit=flag, layout = layout, repeat=repeat)
                runtimes = benchmark.run(args.train)
                writer.writerow({"model_name":"bert-base-uncased",
                                     "batch_size":batch_size,
                                     "sequence_length":seq_length,
                                     "use_jit":flag,
                                     "latency":str(runtimes)})