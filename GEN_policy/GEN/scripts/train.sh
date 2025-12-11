exp_id=GEN_navdp_sonata_fb
config_name=GEN_navdp_sonata_fb

nnodes=1
nproc_per_node=2
node_rank=0
master_addr=127.0.0.1
master_port=29515

CUDA_VISIBLE_DEVICES=4,6 /home/cvte/miniforge3/envs/GEN/bin/torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    main.py \
    --config_name $config_name \
    --save_dir /result/nave2eresult/gen_result/$exp_id \
    --num_nodes $nnodes \
    #--load_dir output/GEN_navdp_concerto_fb_local/policy_latest.ckpt \
    #--debug \
    #--load_pretrain pretrained/VIRT_droid_pretrain.ckpt \
