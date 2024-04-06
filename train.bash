# /bin/bash

python train_ddpm.py --cpu True --config_type community_small
# CUDA_VISIBLE_DEVICES="0" python train_ddpm.py --cpu False --config_type community_small
# CUDA_VISIBLE_DEVICES="0" python train_ddpm.py --cpu False --config_type ego_small
# CUDA_VISIBLE_DEVICES="0" python train_ddpm.py --cpu False --config_type ego
# CUDA_VISIBLE_DEVICES="0" python train_ddpm.py --cpu False --config_type enzyme