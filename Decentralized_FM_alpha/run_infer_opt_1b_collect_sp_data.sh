# Ariel - taken from https://github.com/huangwei021230/DejaVu/blob/ed97c0ff53d41206b01066ee95ec1713f1dbfb0f/Decentralized_FM_alpha/run_infer_opt_125m_collect_sp_data.sh

file=./c4_train/c4_train.jsonl

echo "start running ${file}"

ARGS="--model-name /home/user/DejaVu-Speculative-Decoding/Decentralized_FM_alpha/pretrained_models/facebook/opt-1.3b \
--model-type opt-save \
--seed 42 \
--fp16 \
--num-layers 3 \
--max-layers 12 \
--budget 22800 \
--num-iters 2000 \
--dist-url tcp://127.0.0.1:9032 \
--token-micro-batch-size 1 \
--world-size 1 --pipeline-group-size 1 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \

--infer-data ${file}"


python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0
