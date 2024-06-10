import os
import argparse
import time
import torch
from transformers import OPTConfig, AutoTokenizer
import sys

# print(sys.path)

# Fix src module not found
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.models.gpt_mlp_sparse import GPTLMHeadModel

from flash_attn.models.opt import opt_config_to_gpt2_config
from flash_attn.utils.generation import update_graph_cache


parser = argparse.ArgumentParser(description="OPT generation benchmarking")
parser.add_argument("--promptlen", type=int, default=128)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument(
    "--mlp-K",
    type=int,
    default=49152,
)
parser.add_argument(
    "--model-name",
    type=str,
    default="opt-175b",
    help="model-name",
)
args = parser.parse_args()


os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"
torch.distributed.init_process_group(backend="nccl", init_method="env://")
device = f"cuda:{torch.distributed.get_rank()}"
world_size = torch.distributed.get_world_size()
# Need this, otherwise when we capture the graph the process for GPU 1 would run on both
# GPU0 and GPU1 and things would hang
torch.cuda.set_device(device)

repeats = 3
dtype = torch.float16
device = "cuda"
rtol, atol = 3e-3, 3e-1
fused_ft_kernel = True

if args.model_name == "opt-175b":
    print("Loading 175b config")
    config = OPTConfig.from_pretrained("facebook/opt-66b")
    config.hidden_size = 12 * 1024
    config.word_embed_proj_dim = config.hidden_size
    config.ffn_dim = 4 * config.hidden_size
    config.num_attention_heads = 96
    config.num_hidden_layers = 96
else:
    config = OPTConfig.from_pretrained(args.model_name)
config = opt_config_to_gpt2_config(config)
# Only prenorm supports residual_in_fp32
# config.residual_in_fp32 = getattr(config, 'prenorm', True)
config.use_flash_attn = True
config.fused_bias_fc = True
config.fused_mlp = True
# config.fused_dropout_add_ln = True
config.pad_vocab_size_multiple = 8 * world_size
config.sequence_parallel = False  # Need to set this to False for generation
config.mlp_sparse = False
""" 
sparse mlp config
"""
if args.mlp_K < config.n_inner:
    config.mlp_sparse = True
    config.mlp_K = args.mlp_K
    config.mlp_sp_dim = 1000


from apex.transformer import parallel_state

parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
rank = parallel_state.get_tensor_model_parallel_rank()
process_group = parallel_state.get_tensor_model_parallel_group()

model = GPTLMHeadModel(config, device=device, dtype=dtype, process_group=process_group)
model.eval()

print(
    f"rank {rank}, Num params {sum(p.numel() for p in model.parameters()) * world_size}"
)

torch.manual_seed(0)
# OPT tokenizer requires use_fast=False
# https://huggingface.co/docs/transformers/model_doc/opt
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-66b", use_fast=False)

# input_ids = torch.randint(0, 100, (1, args.promptlen), dtype=torch.long, device="cuda")
input_ids = tokenizer("Hello, my dog is cute and", return_tensors="pt").input_ids.to(
    device=device
)
max_length = input_ids.shape[1] + args.genlen


# Capture graph outside the timing loop
batch_size, seqlen_og = input_ids.shape
# We need to pass tensor_parallel here, otherwise the kv_cache will have the wrong shape
model._decoding_cache = update_graph_cache(
    model, None, batch_size, seqlen_og, max_length, tensor_parallel=world_size
)
out_cg = model.generate(
    input_ids=input_ids,
    max_length=max_length,
    fused_ft_kernel=True,
    cg=True,
    vocab_size=config.vocab_size,
    return_dict_in_generate=True,
    output_scores=True,
    timing=False,
)

torch.cuda.synchronize()
torch.distributed.barrier()
start = time.time()
for _ in range(repeats):
    out_cg = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        fused_ft_kernel=True,
        cg=True,
        vocab_size=config.vocab_size,
        return_dict_in_generate=True,
        output_scores=True,
        timing=False,
    )
torch.cuda.synchronize()
print(
    f"Prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms"
)
# print(tokenizer.batch_decode(out_cg.sequences.tolist()))
if rank == 0:
    print(tokenizer.batch_decode(out_cg.sequences.tolist()))

# If we don't delete the cache, it would hang and not exit. Maybe because the CUDA graph is still
# around and the NCCL connection is not closed?
del model._decoding_cache
