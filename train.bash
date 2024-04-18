# /bin/bash

# python train.py --cpu True --config_type community_small 
# CUDA_VISIBLE_DEVICES="0" python train.py --cpu False --config_type community_small_smooth
# CUDA_VISIBLE_DEVICES="0" python train.py --cpu False --config_type community_small --checkpoint_path models/Community_small/gnn/checkpoint_epoch_300000_t1000_psgn.pth
# CUDA_VISIBLE_DEVICES="1" python train.py --cpu False --config_type ego_small
# CUDA_VISIBLE_DEVICES="0" python train.py --cpu False --config_type ego
# CUDA_VISIBLE_DEVICES="0" python train.py --cpu False --config_type enzyme

CUDA_VISIBLE_DEVICES="1" python train.py --cpu True --config_type mnist_zeros
