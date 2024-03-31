# /bin/bash

CUDA_VISIBLE_DEVICES="1" python train_ddpm.py --cpu False --config_type enzyme
# python train_ddpm.py --cpu True --data_filepath data/dataset/ --data_name Community_small --train_timesteps 1000