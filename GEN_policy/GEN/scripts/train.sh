exp_id=GEN_dinov2
config_name=GEN_dinov2

nnodes=1
nproc_per_node=1
node_rank=0
master_addr=127.0.0.1
master_port=29515

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    main.py \
    --config_name $config_name \
    --save_dir ./output/$exp_id \
    --num_nodes $nnodes \
    #--debug \
    #--load_dir outputs/isaac_move_basic_pickput_memory/policy_latest.ckpt \
    #--load_pretrain pretrained/VIRT_droid_pretrain.ckpt \
