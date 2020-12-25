from bert import FlaxBertForPretrained
from utils import cross_entropy, Memory
import jax
import numpy as np
from jax import numpy as jnp
import argparse
import csv
import timeit
from flax.optim import DynamicScale
from tqdm import tqdm
import ast
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node
import py3nvml.py3nvml as nvml
import os
import pandas as pd
import jax.profiler as profiler

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
class FlaxBertBenchmark:
    def __init__(self, workload, model = None, use_gpu = True, layout="NT", jit=True, repeat = 3, fp16=False):
        self.workload = workload
        if isinstance(model,str):
            self.model = FlaxBertForPretrained.from_pretrained(model, dtype=jnp.float16 if fp16 else jnp.float32)
        elif isinstance(model,FlaxBertForPretrained):
            self.model = model
        else:
            raise ValueError("Invalid model")
        if fp16:
            self.dynamic_scale = DynamicScale()
            self.dtype = jnp.float16
        else:
            self.dtype = jnp.float32
        self.vocab_size = self.model._module.vocab_size
        self.layout = layout
        self.jit = jit
        self.fp16 = fp16
        self.repeat = repeat

        # self.optimizer = Momentum().create(self.model.params)



    def train_speed_memory(self, batch_size, seq_length):
        key = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(key, (batch_size, seq_length), 0, self.vocab_size)
        targets = jax.random.randint(key, (batch_size, seq_length), 0, self.vocab_size)
        labels = jax.random.randint(key, (batch_size, seq_length), 0, 2)
        # input_ids = np.random.randint(0, self.vocab_size, (batch_size, seq_length))
        # targets = np.random.randint(0, self.vocab_size, (batch_size, seq_length))
        # labels = np.random.randint(0,2, (batch_size, seq_length))
        @jax.jit
        def train_step():

            def loss_fn(params):
                token_mask = jnp.where(labels > 0, 1.0, 0.0).astype(self.dtype)
                logits = self.model(input_ids=input_ids, train=True, params=params, dropout_rng=jax.random.PRNGKey(0))[0]
                loss, normalizing_factor = cross_entropy(logits,targets, token_mask)
                jax.profiler.save_device_memory_profile(f"memory/{workload[0]}_{workload[1]}_memory.prof", "gpu")
                return loss / normalizing_factor
            if self.fp16 and jax.local_devices()[0].platform == 'gpu':
                grad_fn = self.dynamic_scale.value_and_grad(loss_fn)
                dyn_scale, is_fin, loss, grad = grad_fn(self.model.params)
            else:
                grad_fn = jax.value_and_grad(loss_fn)
                loss, grad = grad_fn(self.model.params)
            return tree_flatten(grad)[0]


        if jax.local_devices()[0].platform == 'gpu':
            nvml.nvmlInit()
            train_step()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
            max_bytes_in_use = meminfo.used
            memory = Memory(max_bytes_in_use)
            # shutdown nvml
            nvml.nvmlShutdown()
        else:
            memory = None
        # timeit.repeat(train_step,repeat=1,number=2)
        timeit.repeat("for i in train_step():i.block_until_ready()", repeat=1, number=2,globals=locals())
        if self.jit:
            # runtimes = timeit.repeat(train_step,repeat=self.repeat,number=3)
            runtimes = timeit.repeat("for i in train_step():i.block_until_ready()", repeat=self.repeat, number=3,globals=locals())
        else:
            with jax.disable_jit():
                # runtimes = timeit.repeat(train_step, repeat=self.repeat, number=3)
                runtimes = timeit.repeat("for i in train_step():i.block_until_ready()", repeat=self.repeat, number=3,globals=locals())


        return float(np.min(runtimes)/3.0), memory

    def inference_speed_memory(self, batch_size, seq_length):
        # input_ids = np.random.randint(0, self.vocab_size, (batch_size, seq_length))
        key = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(key, (batch_size, seq_length), 0, self.vocab_size)
        @jax.jit
        def ref_step():
            out = self.model(input_ids=input_ids)
            return out[0]
        if jax.local_devices()[0].platform == 'gpu':
            nvml.nvmlInit()
            ref_step().block_until_ready()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
            max_bytes_in_use = meminfo.used
            memory = Memory(max_bytes_in_use)
            # shutdown nvml
            nvml.nvmlShutdown()
        else:
            memory = None
        timeit.repeat("ref_step().block_until_ready()", repeat=1, number=2,globals=locals())
        if self.jit:
            runtimes = timeit.repeat("ref_step().block_until_ready()", repeat=self.repeat,number=3,globals=locals())
        else:
            with jax.disable_jit():
                runtimes = timeit.repeat("ref_step().block_until_ready()",repeat=self.repeat,number=3,globals=locals())
        return float(np.min(runtimes)/3.0), memory


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
    parser.add_argument("--model_name",default="bert-base-uncased",type=str)
    parser.add_argument("--train",default=True,type=ast.literal_eval)
    parser.add_argument("--fp16", default=False, type=ast.literal_eval)
    parser.add_argument("--batch_size",required=True,type=int)
    parser.add_argument("--seq_length", required=True, type=int)
    parser.add_argument("--file",required=True,type=str)
    args = parser.parse_args()
    layout = args.layout
    repeat = args.repeat
    fp16 = args.fp16
    file = args.file
    csv_file = f"{'train' if args.train else 'infer'}/{args.model_name}_{args.batch_size}_{args.seq_length}_{'fp16' if fp16 else 'fp32'}.csv"
    if layout == "NT":
        workload = (args.batch_size,args.seq_length)
    elif layout == "TN":
        workload = (args.seq_length,args.batch_size)
    else:
        raise ValueError("Invalid Layout")
    fieldnames = ["model_name","batch_size","sequence_length","use_jit"]
    for flag in [True]:
        benchmark = FlaxBertBenchmark(workload, model=f"../models/{args.model_name}", jit=flag, layout=layout,
                                      repeat=repeat, fp16=fp16)
        try:
            runtimes, memory = benchmark.run(args.train)
            df = pd.DataFrame({"model_name": [args.model_name],
                               "batch_size": [args.batch_size],
                               "sequence_length": [args.seq_length],
                               "use_jit": [flag],
                               "latency": [str(runtimes)],
                               "peak memory": [str(memory)]})
            df.to_csv(file, mode='a', header=not os.path.exists(file))
        except Exception as e:
            print(e)
            runtimes = None
            memory = None

    # with open(csv_file, 'w') as csv_file:
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames + ["latency", "peak memory"])
    #     writer.writeheader()
    #     for flag in [True]:
    #         benchmark = FlaxBertBenchmark(workload, model=f"../models/{args.model_name}", jit=flag, layout=layout, repeat=repeat, fp16=fp16)
    #         try:
    #             runtimes, memory = benchmark.run(args.train)
    #         except Exception as e:
    #             print(e)
    #             runtimes = None
    #             memory = None
    #         writer.writerow({"model_name": args.model_name,
    #                          "batch_size": args.batch_size,
    #                          "sequence_length": args.seq_length,
    #                          "use_jit": flag,
    #                          "latency": str(runtimes),
    #                          "peak memory": str(memory)})






