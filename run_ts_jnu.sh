#!/bin/bash
# CUDA_VISIBLE_DEVICES 和基础命令
CUDA_DEVICE=2
BASE_CMD="nohup python train_advanced_ts.py --cuda_device 0 --data_name \"JNUFFT\" --data_dir \"./dataset/JNU\" --save_weights --domain_adversarial True --adversarial_loss DA --self_training True --self_training_criterion confidence --adaptive_confidence_threshold --calibration TempScaling --mcc_loss --sdat --checkpoint_dir output/cat_tempscaling_output"

# 定义 transfer_task 的所有组合（移除 [0],[1] 和 [0],[2]）
TRANSFER_TASKS=(
    "[0],[1]" "[1],[0]"
    "[0],[2]" "[2],[0]"
    "[1],[2]" "[2],[1]"
)

# 循环遍历 seed 值和 transfer_task 组合
for seed in {2023..2027}; do
    for transfer_task in "${TRANSFER_TASKS[@]}"; do
        # 设置日志文件名，包含当前 seed 和 transfer_task
        LOG_FILE="train_log_seed_${seed}_task_${transfer_task//,/}_$(date +%Y%m%d%H%M%S).out"
        
        # 组装完整的命令
        FULL_CMD="export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} && ${BASE_CMD} --seed ${seed} --transfer_task ${transfer_task} > ${LOG_FILE} 2>&1 "
        
        # 打印当前执行的命令（可选）
        echo "Running with seed=${seed}, transfer_task=${transfer_task}, log file: ${LOG_FILE}"
        
        # 执行命令
        eval ${FULL_CMD}
	wait
    done
done

echo "All jobs have been submitted."
