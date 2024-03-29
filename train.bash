# /bin/bash

CUDA_VISIBLE_DEVICES="0" python train_ddpm.py --cpu False --data_filepath data/dataset/ --data_name Community_small --train_timesteps 1000
# python train_ddpm.py --cpu True --data_filepath data/dataset/ --data_name Community_small --train_timesteps 1000