# Inference (SUN RGBD)
# DATASET="sunrgbd"
# PRE_TRAIN="sunrgbd_ep1080.pth"
# python demo_sunrgbd.py --test_ckpt $PRE_TRAIN

# Inference (Custom)
# python demo_custom.py --test_ckpt "outputs/custom_ep1080/checkpoint_best.pth"
DATASET="custom"
PRE_TRAIN="outputs/custom_ep1080/checkpoint_best.pth"
python demo_custom.py --test_ckpt $PRE_TRAIN
