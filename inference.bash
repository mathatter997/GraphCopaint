# /bin/bash

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 1000 \
#  --checkpoint_path models/Community_small/gnn/checkpoint_epoch_200000_t1000_psgn.pth \
#  --scheduler_path models/Community_small/scheduler_config.json \
#  --output_path data/dataset/output_com_small.json

#  CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego_small  --cpu False --num_samples 1000 \
#  --checkpoint_path models/Ego_small/gnn/checkpoint_epoch_200000_t1000_psgn.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/dataset/output_ego_small.json

#  CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego  --cpu False --num_samples 1000 \
#  --checkpoint_path models/Ego/gnn/checkpoint_epoch_1000_t1000_psgn.pth \
#  --scheduler_path models/Ego/scheduler_config.json \
#  --output_path data/dataset/output_ego.json
 
#  CUDA_VISIBLE_DEVICES="0" python inference.py --config_type enzyme  --cpu False --num_samples 1000 \
#  --checkpoint_path models/ENZYMES/gnn/checkpoint_epoch_10000_t1000_psgn.pth \
#  --scheduler_path models/ENZYMES/scheduler_config.json \
#  --output_path data/dataset/output_enyzme.json

python inference.py --config_type community_small  --cpu True --num_samples 1000 \
 --sampler vpsde --num_timesteps 1000 \
 --checkpoint_path models/Community_small/gnn/checkpoint_100.pth \
 --scheduler_path models/Community_small/scheduler_config.json \
 --output_path data/dataset/output_com_small_t25.json

# python inference.py --config_type community_small  --cpu True --num_samples 1000 \
#  --sampler ddpm --num_timesteps 25 \
#  --checkpoint_path models/Community_small/gnn/checkpoint_epoch_200000_t1000_psgn.pth \
#  --scheduler_path models/Community_small/scheduler_config.json \
#  --output_path data/dataset/output_com_small_t25.json

# python inference.py --config_type ego_small  --cpu True --num_samples 1000 \
#  --sampler ddim --num_timesteps 25 \
#  --checkpoint_path models/Ego_small/gnn/checkpoint_epoch_200000_t1000_psgn.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/dataset/output_ego_small_ddim_t25.json

# python inference.py --config_type ego  --cpu True --num_samples 1000 \
#  --sampler ddim --num_timesteps 25 \
#  --checkpoint_path models/Ego/gnn/checkpoint_epoch_1000_t1000_psgn.pth \
#  --scheduler_path models/Ego/scheduler_config.json \
#  --output_path data/dataset/output_ego_ddim_t25.json