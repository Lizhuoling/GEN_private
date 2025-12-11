exp_id=GEN_navdp_concerto_fb_local
config_name=GEN_navdp_concerto_fb

nnodes=1
nproc_per_node=1
node_rank=0
master_addr=127.0.0.1
master_port=29517

CUDA_VISIBLE_DEVICES=1 /home/cvte/miniforge3/envs/GEN/bin/torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    main.py \
    --config_name $config_name \
    --save_dir output/$exp_id \
    --load_dir output/$exp_id/policy_latest.ckpt \
    --num_nodes $nnodes \
    --eval \
    #--real_robot \
    #--save_episode \
    #--debug \