# python evaluate.py --dataset 'Community_small' --pred_file 'data/dataset/output_com_small.json'
# python evaluate.py --dataset 'Ego_small' --pred_file 'data/dataset/output_ego_small.json'
# python evaluate.py --dataset 'Ego' --pred_file 'data/dataset/output_ego.json'
# python evaluate.py --dataset 'ENZYMES' --pred_file 'data/dataset/output_enzyme.json'

# python evaluate.py --dataset 'Community_small' --pred_file 'data/dataset/output_com_small_ddim_t25.json'
# python evaluate.py --dataset 'Ego_small' --pred_file 'data/dataset/output_ego_small_ddim_t25.json'
# python evaluate.py --dataset 'Ego' --pred_file 'data/dataset/output_ego_ddim_t25.json'
# python evaluate.py --dataset 'ENZYMES' --pred_file 'data/dataset/output_enzyme_ddim_t25.json'

# python evaluate.py --dataset 'Community_small' --pred_file 'data/dataset/output_com_small_t25.json'
# python evaluate.py --dataset 'Community_small' --pred_file 'data/dataset/output_com_small_copaint.json'
# python evaluate.py --dataset 'Community_small' --pred_file 'data/dataset/output_com_small_copaint.json' \
#     --inpaint_loss True --mask_path 'data/dataset/mask_com_small_copaint.json' \
#     --masked_target_path 'data/dataset/masked_com_small_copaint.json'

# python evaluate.py --dataset 'Community_small' --pred_file 'data/dataset/output_com_small_copaint.json' \
#     --inpaint_loss True --mask_path 'data/dataset/mask_com_small_copaint.json' \
#     --masked_target_path 'data/dataset/masked_com_small_copaint.json'

# python evaluate.py --dataset 'Community_small' --pred_file 'data/dataset/output2_com_small.json' \
    # --inpaint_loss True --mask_path 'data/dataset/mask_com_small_copaint.json' \
    # --masked_target_path 'data/dataset/masked_com_small_copaint.json'

# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/output_com_small_eigen_vpsde_bm5_ep_10000.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/output_com_small_eigen_ddpm_bm5_ep_10000.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/output_com_small_eigen_ddim_bm5_ep_10000.json

# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/output_com_small_eigen_vpsde_bm20_ep_100000.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/output_com_small_eigen_ddpm_bm20_ep_100000.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/output_com_small_eigen_ddim_bm20_ep_100000.json

# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/output_com_small_eigen_vpsde_bm5_ep_100000_clamp.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/output_com_small_eigen_ddpm_bm5_ep_100000_clamp.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/output_com_small_eigen_ddim_bm5_ep_100000_clamp.json


# python evaluate.py --dataset 'Ego_small' --pred_file data/dataset/output_ego_small_ddpm.json
# python evaluate.py --dataset 'Ego_small' --pred_file data/dataset/output_ego_small_ddim.json
# python evaluate.py --dataset 'Ego_small' --pred_file data/dataset/output_ego_small_vpsde.json



python evaluate.py --dataset 'Community_small' --pred_file 'data/dataset/output_com_small_copaint.json' \
    --inpaint_loss True --mask_path 'data/dataset/mask_com_small_copaint.json' \
    --masked_target_path 'data/dataset/masked_com_small_copaint.json'