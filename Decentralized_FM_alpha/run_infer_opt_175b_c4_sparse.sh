file=/home/ubuntu/DejaVu-Main-Repo/Decentralized_FM_alpha/c4_train/c4_train.jsonl
output_file=/home/ubuntu/DejaVu-Main-Repo/Decentralized_FM_alpha/c4_train/output_c4_train_sparse.jsonl

export PATH_TO_MODEL_CHECKPOINT=/home/ubuntu/DejaVu-Main-Repo/Decentralized_FM_alpha/pretrained_models1
echo "start running ${file}"
export SPRARSE_PATH=/home/ubuntu/DejaVu-Main-Repo/sparse_predictor/pred_models
export LAYER=86  # Ariel: This doesn't seem to be used anymore in hf_opt_sparse_mlp_attention.py, it's commented out
export ATTN_TOPK_1=24  # TODO Ariel where was this introduced? This is probably what's causing the missing layer errors
export ATTN_TOPK_2=48
export SPARSE_ATT=1

LAYER=86
export TOPK=1024
ATTN_TOPK_1=24
ATTN_TOPK_2=48

ARGS="--model-name $PATH_TO_MODEL_CHECKPOINT \
--model-type opt-ml-att-sparse \
--seed 42 \
--fp16 \
--num-layers 3 \
--max-layers 24 \
--budget 10800 \
--num-iters 1000 \
--dist-url tcp://127.0.0.1:9969 \
--token-micro-batch-size 2 \
--world-size 8 --pipeline-group-size 8 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file} \
--output-path ${output_file}"

(trap 'kill 0' SIGINT; \
python3 dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    &
python3 dist_inference_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    &
python3 dist_inference_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    &
python3 dist_inference_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    &
python3 dist_inference_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
    &
python3 dist_inference_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
    &
python3 dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    &
python3 dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)


python3 -c "import json
import numpy as np

logprobs = []

with open('$output_file') as f:
    for line in f:
        if line.strip() == '':
            continue
        if 'result' not in json.loads(line):
            break
        item = json.loads(line)

        logprobs += item['result']['choices'][0]['logprobs']['token_logprobs'][1:]
mean_logprob = sum(logprobs) / len(logprobs)
perplexity = np.exp(-mean_logprob)
print('sparse model perplexity:', perplexity)" > $eval_file
cat $eval_file
