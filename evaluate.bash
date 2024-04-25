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



# python evaluate.py --dataset 'Community_small' --pred_file 'data/dataset/output_com_small_copaint.json' \
#     --inpaint_loss True --mask_path 'data/dataset/mask_com_small_copaint.json' \
#     --masked_target_path 'data/dataset/masked_com_small_copaint.json'

# lr_xt_paths=(\
#             exp_init25_decay1.0 \
#             exp_init25_decay1.002 \
#             exp_init25_decay1.004 \
#             exp_init25_decay1.006 \
#             exp_init25_decay1.01 \
#             relu_init100_start600_xT200 \
#             relu_init100_start600_xT400 \ 
#             relu_init100_start200_xT200 \ 
#             relu_init100_start200_xT400 \ 
#             relu_init1000_start200_xT200 \ 
#             relu_init1000_start200_xT400 \
#             relu_init100_start200_xT600\
#             relu_init100_start600_xT600\
#             relu_init1000_start200_xT600)

lr_xt_paths=(exp_init25_decay1.0 \
            exp_init25_decay1.002 \
            exp_init25_decay1.004 \
            exp_init25_decay1.006 \
            relu_init25_start100_xT50 \
            relu_init25_start200_xT50
            relu_init25_start100_xT200 \
            relu_init25_start200_xT200 \
            relu_init25_start100_xT400 \
            relu_init25_start200_xT400 \
            )
echo COM
# lr_xt_paths=(exp_init25_decay1.006)
for ((i = 0; i < ${#lr_xt_paths[@]}; i++)); do
    python evaluate.py --dataset 'Community_small' \
    --pred_file data/dataset/ablation/com/output_copaint_noclip_adaptive_${lr_xt_paths[$i]}.json \
    --inpaint_loss True --mask_path data/dataset/ablation/com/mask_copaint_noclip_adaptive_${lr_xt_paths[$i]}.json \
    --masked_target_path data/dataset/ablation/com/masked_output_copain_noclip_adaptivet_${lr_xt_paths[$i]}.json 
    echo ${lr_xt_paths[$i]}
done
echo EGO
for ((i = 0; i < ${#lr_xt_paths[@]}; i++)); do
    python evaluate.py --dataset 'Ego_small' \
    --pred_file data/dataset/ablation/ego/output_copaint_noclip_adaptive_${lr_xt_paths[$i]}.json \
    --inpaint_loss True --mask_path data/dataset/ablation/ego/mask_copaint_noclip_adaptive_${lr_xt_paths[$i]}.json \
    --masked_target_path data/dataset/ablation/ego/masked_output_copaint_noclip_adaptive_${lr_xt_paths[$i]}.json 
    echo ${lr_xt_paths[$i]}
done
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/ddim_control.json 
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/output_copaint_lr0_control.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/output_copaint_lr0_rt_1_tau_2_ttFalse.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/output_copaint_lr0_rt_1_tau_1_ttTrue.json
# for ((i = 0; i < ${#lr_xt_paths[@]}; i++)); do
#     python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/output_copaint_${lr_xt_paths[$i]}.json 
#     echo ${lr_xt_paths[$i]}
# done

# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/output_copaint_lr0_rt_1_tau_1.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/output_copaint_lr0_rt_1_tau_2.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/output_copaint_lr0_rt_1_tau_5.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/output_copaint_lr0_rt_1_tau_10.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/output_copaint_lr0_rt_1_tau_20.json

# for ((i = 0; i<32; i++)); do
#     python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/ddim_control_${i}.json 
# done


# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/output_copaint_lr0_rt_1_tau_1.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/output_copaint_lr0_rt_2_tau_1.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/output_copaint_lr0_rt_5_tau_1.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/output_copaint_lr0_rt_10_tau_1.json
# python evaluate.py --dataset 'Community_small' --pred_file data/dataset/ablation/com/eigen/output_copaint_lr0_rt_20_tau_1.json


