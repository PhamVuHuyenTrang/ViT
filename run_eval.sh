###### FOR TRAINING
# CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 10020 --nproc_per_node=4 --use_env main_train.py \
# --model deit_tiny_patch16_224 --batch-size 256 --data-path /root/imagenet/ --output_dir /root/checkpoints/rpc_tiny/ \
# --robust --num_iter 4 --lambd 3 --layer -1 --API_Key f9b91afe90c0f06aa89d2a428bd46dac42640bff \
# --wandb --job_name 4itperlayer_TEST3 --project_name neurips_kpca

# ###### FOR ROBUSTNESS EVAL
# CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 1 --nproc_per_node=4 --use_env eval_OOD.py \
# --model deit_tiny_patch16_224 --data-path /path/to/data/imagenet/ --output_dir /path/to/checkpoints/ \
# --robust --num_iter 4 --lambd 4 --layer 0 --resume /path/to/model/checkpoint/

CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10021 --nproc_per_node=4 --use_env main_train.py \
--model deit_tiny_patch16_224 --batch-size 256 --data-path /root/imagenet/ --output_dir /root/checkpoints/rpc_tiny/ \
--robust --num_iter 4 --lambd 4 --layer 0 --eval --resume /root/repos/RPC/rpc/4itperlayer1.pth 