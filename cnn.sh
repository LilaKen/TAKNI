#python train_advanced.py --cuda_device 0 --seed 2025 \
#  --model_name "cnn_features_1d" --data_name "PHM" --data_dir "./dataset/PHM2009" \
#  --transfer_task [0],[3] --last_batch True \
#  --domain_adversarial True --adversarial_loss "CDA"

#!/bin/bash
# 设置模型名称和多个数据集的目录
MODEL_NAME="cnn_features_1d"
DATASETS=("CWRU" "PU" "JNU" "SEU" "PHM2009")

GPUSID0='0'  # Assuming you have 4 GPUs

transfer_task_id1=[0],[1]
transfer_task_id2=[1],[0]
transfer_task_id3=[0],[2]
transfer_task_id4=[2],[0]
transfer_task_id5=[0],[3]
transfer_task_id6=[3],[0]
transfer_task_id7=[1],[2]
transfer_task_id8=[2],[1]
transfer_task_id9=[1],[3]
transfer_task_id10=[3],[1]
transfer_task_id11=[2],[3]
transfer_task_id12=[3],[2]

# 定义每个数据集对应的子集（如 PU 和 PUFFT）
declare -A DATA_SUBSETS
DATA_SUBSETS["CWRU"]="CWRUFFT"
DATA_SUBSETS["PU"]="PUFFT"
DATA_SUBSETS["JNU"]="JNUFFT"
DATA_SUBSETS["SEU"]="SEUFFT"
DATA_SUBSETS["PHM2009"]="PHMFFT"

# 循环遍历每个数据集
for DATASET in "${DATASETS[@]}"
do
  # 获取该数据集对应的子集
  SUBSETS=${DATA_SUBSETS[$DATASET]}

  # 循环遍历每个子集
  for DATA_NAME in $SUBSETS
  do
    DATA_DIR="dataset/$DATASET"

    # 设置 SEED 循环
    for SEED in {2023..2027}
    do
      for i in 1
      do
        if [ $i -eq 0 ]; then
          LOSSES=("DA" "CDA")
          DOMAIN_ADV="--domain_adversarial True"
          DISTANCE_METRIC=""
        else
          LOSSES=("LJMMD")
          DOMAIN_ADV=""
          DISTANCE_METRIC="--distance_metric True"
        fi

        # 根据数据集选择 transfer_task_id
        if [ "$DATASET" = "JNU" ]; then
          VALID_TASKS=(1 2 3 4 7 8)
        elif [ "$DATASET" = "SEU" ]; then
          VALID_TASKS=(1 2)
        else
          VALID_TASKS=(1 2 3 4 5 6 7 8 9 10 11 12)
        fi

        # 运行适用的 transfer_task_id
        for transfer_task_id in "${VALID_TASKS[@]}"
        do
          TASK_VAR="transfer_task_id${transfer_task_id}"
          CURRENT_TASK=${!TASK_VAR}

          for LOSS in "${LOSSES[@]}"
          do
            if [ $i -eq 0 ]; then
              DISTANCE_LOSS="--adversarial_loss $LOSS"
            else
              DISTANCE_LOSS="--distance_loss $LOSS"
            fi

            echo "Running transfer task $CURRENT_TASK with seed $SEED, data $DATA_NAME, loss $LOSS on device $GPUSID0"
            python train_advanced.py --cuda_device $GPUSID0 --seed $SEED \
              --model_name "$MODEL_NAME" --data_name $DATA_NAME --data_dir $DATA_DIR \
              --transfer_task $CURRENT_TASK --last_batch True \
              $DISTANCE_METRIC $DOMAIN_ADV $DISTANCE_LOSS
          done
        done
      done
    done
  done
done
