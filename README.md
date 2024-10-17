# MaSSL
Official PyTroch implementation for *Learning from Memory: A Non-Parametric Memory Augmented Self-Supervised Learning of Visual Features*

![Alt Text](https://github.com/sthalles/MaSSL/blob/main/assets/method_video.gif)

Project webpage: https://sthalles.github.io/MaSSL/

# Running scripts

### Run vit-base model on ImageNet

```
torchrun --nproc-per-node=4 main_massl.py --arch vit_base --batch_size_per_gpu 32 --num_workers 6 --local_crops_number 10 --patch_size 16 --weight_decay 0.04 --weight_decay_end 0.4 --lr 0.00075 --min_lr 2e-06 --print_freq 50 --global_crops_scale 0.32 1 --local_crops_scale 0.05 0.32 --epochs 400 --gradient_accumulation 1 --optimizer adamw --momentum_teacher 0.996 --drop_path_rate 0.1 --use_bn_in_head false --out_dim 131072 --partition_size 16384 --warmup_teacher_temp_epochs 50 --warmup_teacher_temp 0.04 --teacher_temp 0.07 --clip_grad 3 --warmup_epochs 10 --koleo_loss_weight 0.0
```

## Abstract

*We present Consistent Assignment of Views over Random Partitions (CARP), a self-supervised clustering method for representation learning of visual features. CARP learns prototypes in an end-to-end online fashion using gradient descent without additional non-differentiable modules to solve the cluster assignment problem. CARP optimizes a new pretext task based on random partitions of prototypes that regularizes the model and enforces consistency between views’ assignments. Additionally, our method improves training stability and prevents collapsed solutions in joint-embedding training. Through an extensive evaluation, we demonstrate that CARP’s representations are suitable for learning downstream tasks. We evaluate CARP’s representations capabilities in 17 datasets across many standard protocols, including linear evaluation, few-shot classification, k-NN, kmeans, image retrieval, and copy detection. We compare CARP performance to 11 existing self-supervised methods. We extensively ablate our method and demonstrate that our proposed random partition pretext task improves the quality of the learned representations by devising multiple random classification tasks. In transfer learning tasks, CARP achieves the best performance on average against many SSL methods trained for a longer time.*