DATASET_NAME='wikihow'
DATA_PATH='/home/CVPR_2022/data/'${DATASET_NAME}
WEIGHT_PATH='/home/CVPR_2022/data/weights'

CUDA_VISIBLE_DEVICES=1 python train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME}  \
  --logger_name runs/${DATASET_NAME}_wsl_grid_bert_emb300_att_baseline/log --model_name runs/${DATASET_NAME}_wsl_grid_bert_emb300_att_baseline \
  --num_epochs=25 --lr_update=15 --batch_size=8 --learning_rate=5e-4 --precomp_enc_type backbone  --workers 8 --backbone_source wsl \
  --vse_mean_warmup_epochs 1 --backbone_warmup_epochs 25 --embedding_warmup_epochs 0  --optim adam --backbone_lr_factor 0.01  --log_step 200 \
  --input_scale_factor 1.0  --backbone_path ${WEIGHT_PATH}/original_updown_backbone.pth  --resume runs/${DATASET_NAME}_wsl_grid_bert_emb300_att_baseline/checkpoint.pth \
  --embedding_size 300 --no_imgnorm --no_txtnorm --attention_loss ce --attention_loss_weight 0.2
