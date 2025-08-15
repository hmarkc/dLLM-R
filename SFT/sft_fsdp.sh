export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file fsdp_config.yaml \
  --main_process_port 29503 \
  --num_processes 2 \
  sft_train.py \
  --grad_accum_steps 4 \
  --batch_size 1 \
  --num_epochs 20
