#!/bin/bash
# JNU 3 有问题
# 设置模型名称和数据集目录
MODEL_NAME="cnn_features_1d"
#DATA_DIR=("dataset/JNU" "dataset/SEU" "dataset/PU" "dataset/PHM")
DATA_DIR=("dataset/CWRU")

GPUSID0='0'  # Assuming you have 4 GPUs
GPUSID1='1'
GPUSID2='2'
GPUSID3='3'
#PU

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

# 0-->1 PU
for DATA_NAME in "CWRUFFT"
do
  for SEED in {2023..2027}
  do
    for i in 1
    do
      if [ $i -eq 0 ]; then
        LOSSES=("DA" "CDA" "CDA+E")
        DOMAIN_ADV="--domain_adversarial True"
        DISTANCE_METRIC=""
      else
        LOSSES=("MK-MMD" "JMMD" )
        DOMAIN_ADV=""
        DISTANCE_METRIC="--distance_metric True"
      fi
      for LOSS in "${LOSSES[@]}"
      do
        if [ $i -eq 0 ]; then
          DISTANCE_LOSS="--adversarial_loss $LOSS"
        else
          DISTANCE_LOSS="--distance_loss $LOSS"
        fi
        echo "Running 0-->1 SEUFFT with seed $SEED, data $DATA_NAME, transfer task $transfer_task_id1, loss $LOSS on device $GPUSID1"
        python train_advanced.py --cuda_device $GPUSID1 --seed $SEED \
          --model_name "$MODEL_NAME" --data_name $DATA_NAME --data_dir $DATA_DIR \
          --transfer_task $transfer_task_id1 --last_batch True \
          $DISTANCE_METRIC $DOMAIN_ADV $DISTANCE_LOSS
      done
    done
  done
done
#


#

# 1-->0 PU
for DATA_NAME in "CWRUFFT"
do
  for SEED in {2023..2027}
  do
    for i in 1
    do
      if [ $i -eq 0 ]; then
        LOSSES=("DA" "CDA" "CDA+E")
        DOMAIN_ADV="--domain_adversarial True"
        DISTANCE_METRIC=""
      else
        LOSSES=("MK-MMD" "JMMD" )
        DOMAIN_ADV=""
        DISTANCE_METRIC="--distance_metric True"
      fi
      for LOSS in "${LOSSES[@]}"
      do
        if [ $i -eq 0 ]; then
          DISTANCE_LOSS="--adversarial_loss $LOSS"
        else
          DISTANCE_LOSS="--distance_loss $LOSS"
        fi
        echo "Running 1-->0 SEUFFT with seed $SEED, data $DATA_NAME, transfer task $transfer_task_id2, loss $LOSS on device $GPUSID1"
        python train_advanced.py --cuda_device $GPUSID1 --seed $SEED \
          --model_name "$MODEL_NAME" --data_name $DATA_NAME --data_dir $DATA_DIR \
          --transfer_task $transfer_task_id2 --last_batch True \
          $DISTANCE_METRIC $DOMAIN_ADV $DISTANCE_LOSS
      done
    done
  done
done


# 0-->2 PU
for DATA_NAME in "CWRUFFT"
do
  for SEED in {2023..2027}
  do
    for i in 1
    do
      if [ $i -eq 0 ]; then
        LOSSES=("DA" "CDA" "CDA+E")
        DOMAIN_ADV="--domain_adversarial True"
        DISTANCE_METRIC=""
      else
        LOSSES=("MK-MMD" "JMMD" )
        DOMAIN_ADV=""
        DISTANCE_METRIC="--distance_metric True"
      fi
      for LOSS in "${LOSSES[@]}"
      do
        if [ $i -eq 0 ]; then
          DISTANCE_LOSS="--adversarial_loss $LOSS"
        else
          DISTANCE_LOSS="--distance_loss $LOSS"
        fi
        echo "Running 0-->2 SEUFFT with seed $SEED, data $DATA_NAME, transfer task $transfer_task_id3, loss $LOSS on device $GPUSID1"
        python train_advanced.py --cuda_device $GPUSID1 --seed $SEED \
          --model_name "$MODEL_NAME" --data_name $DATA_NAME --data_dir $DATA_DIR \
          --transfer_task $transfer_task_id3 --last_batch True \
          $DISTANCE_METRIC $DOMAIN_ADV $DISTANCE_LOSS
      done
    done
  done
done


# 2-->0 PU
for DATA_NAME in "CWRUFFT"
do
  for SEED in {2023..2027}
  do
    for i in 1
    do
      if [ $i -eq 0 ]; then
        LOSSES=("DA" "CDA" "CDA+E")
        DOMAIN_ADV="--domain_adversarial True"
        DISTANCE_METRIC=""
      else
        LOSSES=("MK-MMD" "JMMD" )
        DOMAIN_ADV=""
        DISTANCE_METRIC="--distance_metric True"
      fi
      for LOSS in "${LOSSES[@]}"
      do
        if [ $i -eq 0 ]; then
          DISTANCE_LOSS="--adversarial_loss $LOSS"
        else
          DISTANCE_LOSS="--distance_loss $LOSS"
        fi
        echo "Running 2-->0 SEUFFT with seed $SEED, data $DATA_NAME, transfer task $transfer_task_id4, loss $LOSS on device $GPUSID1"
        python train_advanced.py --cuda_device $GPUSID1 --seed $SEED \
          --model_name "$MODEL_NAME" --data_name $DATA_NAME --data_dir $DATA_DIR \
          --transfer_task $transfer_task_id4 --last_batch True \
          $DISTANCE_METRIC $DOMAIN_ADV $DISTANCE_LOSS
      done
    done
  done
done

## 0-->3 PU
for DATA_NAME in "CWRUFFT"
do
  for SEED in {2023..2027}
  do
    for i in 1
    do
      if [ $i -eq 0 ]; then
        LOSSES=("DA" "CDA" "CDA+E")
        DOMAIN_ADV="--domain_adversarial True"
        DISTANCE_METRIC=""
      else
        LOSSES=("MK-MMD" "JMMD" )
        DOMAIN_ADV=""
        DISTANCE_METRIC="--distance_metric True"
      fi
      for LOSS in "${LOSSES[@]}"
      do
        if [ $i -eq 0 ]; then
          DISTANCE_LOSS="--adversarial_loss $LOSS"
        else
          DISTANCE_LOSS="--distance_loss $LOSS"
        fi
        echo "Running 0-->3 SEUFFT with seed $SEED, data $DATA_NAME, transfer task $transfer_task_id5, loss $LOSS on device $GPUSID1"
        python train_advanced.py --cuda_device $GPUSID1 --seed $SEED \
          --model_name "$MODEL_NAME" --data_name $DATA_NAME --data_dir $DATA_DIR \
          --transfer_task $transfer_task_id5 --last_batch True \
          $DISTANCE_METRIC $DOMAIN_ADV $DISTANCE_LOSS
      done
    done
  done
done

# 3-->0 PU
for DATA_NAME in "CWRUFFT"
do
  for SEED in {2023..2027}
  do
    for i in 1
    do
      if [ $i -eq 0 ]; then
        LOSSES=("DA" "CDA" "CDA+E")
        DOMAIN_ADV="--domain_adversarial True"
        DISTANCE_METRIC=""
      else
        LOSSES=("MK-MMD" "JMMD" )
        DOMAIN_ADV=""
        DISTANCE_METRIC="--distance_metric True"
      fi
      for LOSS in "${LOSSES[@]}"
      do
        if [ $i -eq 0 ]; then
          DISTANCE_LOSS="--adversarial_loss $LOSS"
        else
          DISTANCE_LOSS="--distance_loss $LOSS"
        fi
        echo "Running 3-->0 SEUFFT with seed $SEED, data $DATA_NAME, transfer task $transfer_task_id6, loss $LOSS on device $GPUSID1"
        python train_advanced.py --cuda_device $GPUSID1 --seed $SEED \
          --model_name "$MODEL_NAME" --data_name $DATA_NAME --data_dir $DATA_DIR \
          --transfer_task $transfer_task_id6 --last_batch True \
          $DISTANCE_METRIC $DOMAIN_ADV $DISTANCE_LOSS
      done
    done
  done
done

# 1-->2 PU
for DATA_NAME in "CWRUFFT"
do
  for SEED in {2023..2027}
  do
    for i in 1
    do
      if [ $i -eq 0 ]; then
        LOSSES=("DA" "CDA" "CDA+E")
        DOMAIN_ADV="--domain_adversarial True"
        DISTANCE_METRIC=""
      else
        LOSSES=("LJMMD")
        DOMAIN_ADV=""
        DISTANCE_METRIC="--distance_metric True"
      fi
      for LOSS in "${LOSSES[@]}"
      do
        if [ $i -eq 0 ]; then
          DISTANCE_LOSS="--adversarial_loss $LOSS"
        else
          DISTANCE_LOSS="--distance_loss $LOSS"
        fi
        echo "Running 1-->2 SEUFFT with seed $SEED, data $DATA_NAME, transfer task $transfer_task_id7, loss $LOSS on device $GPUSID1"
        python train_advanced.py --cuda_device $GPUSID1 --seed $SEED \
          --model_name "$MODEL_NAME" --data_name $DATA_NAME --data_dir $DATA_DIR \
          --transfer_task $transfer_task_id7 --last_batch True \
          $DISTANCE_METRIC $DOMAIN_ADV $DISTANCE_LOSS
      done
    done
  done
done

# 2-->1 PU
for DATA_NAME in "CWRUFFT"
do
  for SEED in {2023..2027}
  do
    for i in 1
    do
      if [ $i -eq 0 ]; then
        LOSSES=("DA" "CDA" "CDA+E")
        DOMAIN_ADV="--domain_adversarial True"
        DISTANCE_METRIC=""
      else
        LOSSES=("MK-MMD" "JMMD" )
        DOMAIN_ADV=""
        DISTANCE_METRIC="--distance_metric True"
      fi
      for LOSS in "${LOSSES[@]}"
      do
        if [ $i -eq 0 ]; then
          DISTANCE_LOSS="--adversarial_loss $LOSS"
        else
          DISTANCE_LOSS="--distance_loss $LOSS"
        fi
        echo "Running 2-->1 SEUFFT with seed $SEED, data $DATA_NAME, transfer task $transfer_task_id8, loss $LOSS on device $GPUSID1"
        python train_advanced.py --cuda_device $GPUSID1 --seed $SEED \
          --model_name "$MODEL_NAME" --data_name $DATA_NAME --data_dir $DATA_DIR \
          --transfer_task $transfer_task_id8 --last_batch True \
          $DISTANCE_METRIC $DOMAIN_ADV $DISTANCE_LOSS
      done
    done
  done
done

# 1-->3 PU
for DATA_NAME in "CWRUFFT"
do
  for SEED in {2023..2027}
  do
    for i in 1
    do
      if [ $i -eq 0 ]; then
        LOSSES=("DA" "CDA" "CDA+E")
        DOMAIN_ADV="--domain_adversarial True"
        DISTANCE_METRIC=""
      else
        LOSSES=("MK-MMD" "JMMD" )
        DOMAIN_ADV=""
        DISTANCE_METRIC="--distance_metric True"
      fi
      for LOSS in "${LOSSES[@]}"
      do
        if [ $i -eq 0 ]; then
          DISTANCE_LOSS="--adversarial_loss $LOSS"
        else
          DISTANCE_LOSS="--distance_loss $LOSS"
        fi
        echo "Running 1-->3 SEUFFT with seed $SEED, data $DATA_NAME, transfer task $transfer_task_id9, loss $LOSS on device $GPUSID1"
        python train_advanced.py --cuda_device $GPUSID1 --seed $SEED \
          --model_name "$MODEL_NAME" --data_name $DATA_NAME --data_dir $DATA_DIR \
          --transfer_task $transfer_task_id9 --last_batch True \
          $DISTANCE_METRIC $DOMAIN_ADV $DISTANCE_LOSS
      done
    done
  done
done

# 3-->1 PU
for DATA_NAME in "CWRUFFT"
do
  for SEED in {2023..2027}
  do
    for i in 1
    do
      if [ $i -eq 0 ]; then
        LOSSES=("DA" "CDA" "CDA+E")
        DOMAIN_ADV="--domain_adversarial True"
        DISTANCE_METRIC=""
      else
        LOSSES=("MK-MMD" "JMMD" )
        DOMAIN_ADV=""
        DISTANCE_METRIC="--distance_metric True"
      fi
      for LOSS in "${LOSSES[@]}"
      do
        if [ $i -eq 0 ]; then
          DISTANCE_LOSS="--adversarial_loss $LOSS"
        else
          DISTANCE_LOSS="--distance_loss $LOSS"
        fi
        echo "Running 3-->1 SEUFFT with seed $SEED, data $DATA_NAME, transfer task $transfer_task_id10, loss $LOSS on device $GPUSID1"
        python train_advanced.py --cuda_device $GPUSID1 --seed $SEED \
          --model_name "$MODEL_NAME" --data_name $DATA_NAME --data_dir $DATA_DIR \
          --transfer_task $transfer_task_id10 --last_batch True \
          $DISTANCE_METRIC $DOMAIN_ADV $DISTANCE_LOSS
      done
    done
  done
done

## 2-->3 PU
for DATA_NAME in "CWRUFFT"
do
  for SEED in {2023..2027}
  do
    for i in 1
    do
      if [ $i -eq 0 ]; then
        LOSSES=("DA" "CDA" "CDA+E")
        DOMAIN_ADV="--domain_adversarial True"
        DISTANCE_METRIC=""
      else
        LOSSES=("MK-MMD" "JMMD" )
        DOMAIN_ADV=""
        DISTANCE_METRIC="--distance_metric True"
      fi
      for LOSS in "${LOSSES[@]}"
      do
        if [ $i -eq 0 ]; then
          DISTANCE_LOSS="--adversarial_loss $LOSS"
        else
          DISTANCE_LOSS="--distance_loss $LOSS"
        fi
        echo "Running 2-->3 SEUFFT with seed $SEED, data $DATA_NAME, transfer task $transfer_task_id11, loss $LOSS on device $GPUSID1"
        python train_advanced.py --cuda_device $GPUSID1 --seed $SEED \
          --model_name "$MODEL_NAME" --data_name $DATA_NAME --data_dir $DATA_DIR \
          --transfer_task $transfer_task_id11 --last_batch True \
          $DISTANCE_METRIC $DOMAIN_ADV $DISTANCE_LOSS
      done
    done
  done
done

# 3-->2 PU
for DATA_NAME in "CWRUFFT"
do
  for SEED in {2023..2027}
  do
    for i in 1
    do
      if [ $i -eq 0 ]; then
        LOSSES=("DA" "CDA" "CDA+E")
        DOMAIN_ADV="--domain_adversarial True"
        DISTANCE_METRIC=""
      else
        LOSSES=("MK-MMD" "JMMD" )
        DOMAIN_ADV=""
        DISTANCE_METRIC="--distance_metric True"
      fi
      for LOSS in "${LOSSES[@]}"
      do
        if [ $i -eq 0 ]; then
          DISTANCE_LOSS="--adversarial_loss $LOSS"
        else
          DISTANCE_LOSS="--distance_loss $LOSS"
        fi
        echo "Running 3-->2 SEUFFT with seed $SEED, data $DATA_NAME, transfer task $transfer_task_id12, loss $LOSS on device $GPUSID1"
        python train_advanced.py --cuda_device $GPUSID1 --seed $SEED \
          --model_name "$MODEL_NAME" --data_name $DATA_NAME --data_dir $DATA_DIR \
          --transfer_task $transfer_task_id12 --last_batch True \
          $DISTANCE_METRIC $DOMAIN_ADV $DISTANCE_LOSS
      done
    done
  done
done