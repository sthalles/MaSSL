# MaSSL
Official PyTroch implementation for *Learning from Memory: A Non-Parametric Memory Augmented Self-Supervised Learning of Visual Features*

![Alt Text](https://github.com/sthalles/MaSSL/blob/main/assets/method_video.gif)

Project webpage: https://sthalles.github.io/MaSSL/

# Running scripts

### Run vit-small model on ImageNet

```
torchrun --nproc-per-node=2 main_massl.py --arch vit_small --batch_size_per_gpu 64 --num_workers 10 --local_crops_number 10 --patch_size 16 --weight_decay 0.04 --weight_decay_end 0.4 --lr 5e-4 --min_lr 1e-06 --print_freq 50 --global_crops_scale 0.32 1 --local_crops_scale 0.05 0.32 --epochs 800 --gradient_accumulation 1 --optimizer adamw --momentum_teacher 0.992 --drop_path_rate 0.1 --use_bn_in_head false --out_dim 131072 --partition_size 16384 --warmup_teacher_temp_epochs 30 --warmup_teacher_temp 0.04 --clip_grad 3 --warmup_epochs 0 --koleo_loss_weight 0.0 --data_path <path/to/imagenet/train>
```

### Run vit-base model on ImageNet

```
torchrun --nproc-per-node=4 main_massl.py --arch vit_base --batch_size_per_gpu 32 --num_workers 6 --local_crops_number 10 --patch_size 16 --weight_decay 0.04 --weight_decay_end 0.4 --lr 0.00075 --min_lr 2e-06 --print_freq 50 --global_crops_scale 0.32 1 --local_crops_scale 0.05 0.32 --epochs 400 --gradient_accumulation 1 --optimizer adamw --momentum_teacher 0.996 --drop_path_rate 0.1 --use_bn_in_head false --out_dim 131072 --partition_size 16384 --warmup_teacher_temp_epochs 50 --warmup_teacher_temp 0.04 --teacher_temp 0.07 --clip_grad 3 --warmup_epochs 10 --koleo_loss_weight 0.0 --data_path <path/to/imagenet/train>
```

## Abstract

*This paper introduces a novel approach to improving the training stability of self-supervised learning (SSL) methods by leveraging a non-parametric memory of seen concepts. The proposed method involves augmenting a neural network with a memory component to stochastically compare current image views with previously encountered concepts. Additionally, we introduce stochastic memory blocks to regularize training and enforce consistency between image views. We extensively benchmark our method on many vision tasks, such as linear probing, transfer learning, low-shot classification, and image retrieval on many datasets. The experimental results consolidate the effectiveness of the proposed approach in achieving stable SSL training without additional regularizers while learning highly transferable representations and requiring less computing time and resources.*

## Reference

```
@inproceedings{silva2024massl,
  title={Learning from Memory: Non-Parametric Memory Augmented Self-Supervised Learning of Visual Features},
  author={Silva, Thalles and Pedrini, Helio and Rivera, Ad{\'\i}n Ram{\'\i}rez},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={1--17},
  month=jul,
  year={2024}
}
```