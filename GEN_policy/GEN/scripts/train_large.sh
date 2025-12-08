exp_id=GEN_navdp_concerto_fb
config_name=GEN_navdp_concerto_fb

nnodes=1
nproc_per_node=8
node_rank=0
master_addr=127.0.0.1
master_port=29515

source /root/miniforge3/bin/activate GEN

pwd

cd GEN_policy/GEN

torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    main.py \
    --config_name $config_name \
    --save_dir /result/nave2eresult/gen_result/$exp_id \
    --num_nodes $nnodes \
    #--debug \
    #--load_dir outputs/isaac_move_basic_pickput_memory/policy_latest.ckpt \
    #--load_pretrain pretrained/VIRT_droid_pretrain.ckpt \
