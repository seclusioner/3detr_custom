# ========== SUNRGBD dataset ==========
# Training
# python main.py --dataset_name "sunrgbd" --max_epoch 1080 \
# --nqueries 128 \
# --base_lr 7e-4 \
# --matcher_giou_cost 3 \
# --matcher_cls_cost 1 \
# --matcher_center_cost 5 \
# --matcher_objectness_cost 5 \
# --loss_giou_weight 0 \
# --loss_no_object_weight 0.1 \
# --save_separate_checkpoint_every_epoch -1 \
# --checkpoint_dir "outputs/sunrgbd_ep1080"

# Testing
# DATANAME="sunrgbd"
# Queries="128"
# MODELPATH="sunrgbd_ep1080.pth"
# python main.py --dataset_name $DATANAME --nqueries $Queries --test_ckpt $MODELPATH --test_only

# Tensorboard (path: /root/Work/3detr/)
# tensorboard --logdir=outputs/custom_ep1080/

# ========== Custom dataset ==========
# Training
python main.py --dataset_name "custom" --max_epoch 1080 \
--nqueries 256 \
--base_lr 7e-4 \
--matcher_giou_cost 3 \
--matcher_cls_cost 1 \
--matcher_center_cost 5 \
--matcher_objectness_cost 5 \
--loss_giou_weight 0 \
--loss_no_object_weight 0.1 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir "outputs/custom_ep1080"

# Testing
# DATANAME="custom"
# Queries="128"
# MODELPATH="custom_ep1080.pth"
# python main.py --dataset_name $DATANAME --nqueries $Queries --test_ckpt $MODELPATH --test_only

# Tensorboard (path: /root/Work/3detr/)
# tensorboard --logdir=outputs/custom_ep1080/


