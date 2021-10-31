#!/usr/bin/env bash

EXP_DIR=exps/coco
BASE_TRAIN_DIR=${EXP_DIR}/base_train
mkdir exps
mkdir ${EXP_DIR}
mkdir ${BASE_TRAIN_DIR}

python -u main.py \
    --dataset_file coco_base \
    --backbone resnet101 \
    --num_feature_levels 1 \
    --enc_layers 6 \
    --dec_layers 6 \
    --hidden_dim 256 \
    --num_queries 300 \
    --batch_size 4 \
    --category_codes_cls_loss \
    --epoch 25 \
    --lr_drop_milestones 20 \
    --save_every_epoch 5 \
    --eval_every_epoch 5 \
    --output_dir ${BASE_TRAIN_DIR} \
2>&1 | tee ${BASE_TRAIN_DIR}/log.txt


for fewshot_seed in 01 02 03 04 05
do
  for num_shot in 10 30
  do
    FS_FT_DIR=${EXP_DIR}/seed${fewshot_seed}_${num_shot}shot
    mkdir ${FS_FT_DIR}

    if [ $num_shot -eq 10 ]
    then
      epoch=500
      lr_drop1=300
      lr_drop2=450
    elif [ $num_shot -eq 30 ]
    then
      epoch=500
      lr_drop1=300
      lr_drop2=450
    else
      exit
    fi

    python -u main.py \
        --dataset_file coco_base \
        --backbone resnet101 \
        --num_feature_levels 1 \
        --enc_layers 6 \
        --dec_layers 6 \
        --hidden_dim 256 \
        --num_queries 300 \
        --batch_size 2 \
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
