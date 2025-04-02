#!/bin/bash

# Use CUDA device 0
export CUDA_VISIBLE_DEVICES=0

# Set paths to the dataset directories
REAL_PATH="/l/users/sarim.hashmi/Thesis/NIPS/deep_fake_detection/subset_100/real"
FAKE_PATH="/l/users/sarim.hashmi/Thesis/NIPS/deep_fake_detection/subset_100/fake"

# Run inference with the modified validate.py
python new_inference.py \
  --arch=CLIP:ViT-L/14 \
  --ckpt=pretrained_weights/fc_weights.pth \
  --result_folder=subset_100_results \
  --real_path=$REAL_PATH \
  --fake_path=$FAKE_PATH \
  --batch_size=32