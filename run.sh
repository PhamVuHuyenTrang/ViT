CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10020 --nproc_per_node=4 --use_env main_train.py \
--model deit_tiny_patch16_224 --batch-size 256 --data-path /root/data/imagenet/ --output_dir /root/checkpoints_trang/rpc_tiny/ --hyperparam_1 1.0 --hyperparam_2 1.0 --method vit  &
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10020 --nproc_per_node=4 --use_env main_train.py \
--model deit_tiny_patch16_224 --batch-size 256 --data-path /root/data/imagenet/ --output_dir /root/checkpoints_trang/rpc_tiny/ --hyperparam_1 0.1 --hyperparam_2 1.0 --method moving_average &
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10020 --nproc_per_node=4 --use_env main_train.py \
--model deit_tiny_patch16_224 --batch-size 256 --data-path /root/data/imagenet/ --output_dir /root/checkpoints_trang/rpc_tiny/ --hyperparam_1 0.3 --hyperparam_2 1.0 --method moving_average &
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10020 --nproc_per_node=4 --use_env main_train.py \
--model deit_tiny_patch16_224 --batch-size 256 --data-path /root/data/imagenet/ --output_dir /root/checkpoints_trang/rpc_tiny/ --hyperparam_1 0.6 --hyperparam_2 1.0 --method moving_average &
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10020 --nproc_per_node=4 --use_env main_train.py \
--model deit_tiny_patch16_224 --batch-size 256 --data-path /root/data/imagenet/ --output_dir /root/checkpoints_trang/rpc_tiny/ --hyperparam_1 0.9 --hyperparam_2 1.0 --method moving_average &
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10020 --nproc_per_node=4 --use_env main_train.py \
--model deit_tiny_patch16_224 --batch-size 256 --data-path /root/data/imagenet/ --output_dir /root/checkpoints_trang/rpc_tiny/ --u 0.3 --s 0.1 --method momentum &
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10020 --nproc_per_node=4 --use_env main_train.py \
--model deit_tiny_patch16_224 --batch-size 256 --data-path /root/data/imagenet/ --output_dir /root/checkpoints_trang/rpc_tiny/ --u 0.1 --s 0.1 --method momentum &
