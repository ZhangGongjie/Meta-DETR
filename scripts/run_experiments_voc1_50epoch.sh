#!/usr/bin/env bash

EXP_DIR=exps/voc1
BASE_TRAIN_DIR=${EXP_DIR}/base_train
mkdir exps
mkdir ${EXP_DIR}
mkdir ${BASE_TRAIN_DIR}

python -u main.py \
    --dataset_file voc_base1 \
    --backbone resnet101 \
    --num_feature_levels 1 \
    --enc_layers 6 \
    --dec_layers 6 \
    --hidden_dim 256 \
    --num_queries 300 \
    --batch_size 4 \
    --category_codes_cls_loss \
    --epoch 50 \
    --lr_drop_milestones 45 \
    --save_every_epoch 10 \
    --eval_every_epoch 10 \
    --output_dir ${BASE_TRAIN_DIR} \
2>&1 | tee ${BASE_TRAIN_DIR}/log.txt



for fewshot_seed in 01 02 03 04 05 06 07 08 09 10
do
  for num_shot in 01 02 03 05 10
  do
    FS_FT_DIR=${EXP_DIR}/seed${fewshot_seed}_${num_shot}shot
    mkdir ${FS_FT_DIR}

    if [ $num_shot -eq 1 ]
    then
      epoch=700
      lr_drop1=350
      lr_drop2=600
      lr=1.5e-4
      lr_backbone=1.5e-5
    elif [ $num_shot -eq 2 ]
    then
      epoch=600
      lr_drop1=300
      lr_drop2=550
      lr=1e-4
      lr_backbone=1e-5
    elif [ $num_shot -eq 3 ]
    then
      epoch=600
      lr_drop1=300
      lr_drop2=550
      lr=1e-4
      lr_backbone=1e-5
    elif [ $num_shot -eq 5 ]
    then
      epoch=500
      lr_drop1=250
      lr_drop2=450
      lr=5e-5
      lr_backbone=5e-6
    elif [ $num_shot -eq 10 ]
    then
      epoch=500
      lr_drop1=250
      lr_drop2=450
      lr=5e-5
      lr_backbone=5e-6
    else
      exit
    fi

    python -u main.py \
        --dataset_file voc_base1 \
        --backbone resnet101 \
        --num_feature_levels 1 \
        --enc_layers 6 \
        --dec_layers 6 \
        --hidden_dim 256 \
        --num_queries 300 \
        --batch_size 2 \
        --lr ${lr} \
        --lr_backbone ${lr_backbone} \
        --category_codes_cls_loss \
        --resume ${BASE_TRAIN_DIR}/checkpoint.pth \
        --fewshot_finetune \
        --fewshot_seed ${fewshot_seed} \
        --num_shots ${num_shot} \
        --epoch ${epoch} \
        --lr_drop_milestones ${lr_drop1} ${lr_drop2} \
        --warmup_epochs 50 \
        --save_every_epoch ${epoch} \
        --eval_every_epoch ${epoch} \
        --output_dir ${FS_FT_DIR} \
    2>&1 | tee ${FS_FT_DIR}/log.txt
  done
done

