import argparse
from collections import defaultdict
import os
import shutil
import sys
import datetime
import time
import math
import json
from pathlib import Path
from koleo_loss import KoLeoLoss
import yaml

from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import utils
from head import ProjectionHead
from memory_bank import MemoryBank
from random_partition import RandomPartition
from loss import Criterion
import models
import socket
from datetime import datetime
from pt_distr_env import DistributedEnviron


torchvision_archs = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)


def get_args_parser():
    parser = argparse.ArgumentParser("MaSSL", add_help=False)

    # Model parameters
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "vit_large",
            "deit_tiny",
            "deit_small",
            "swin_tiny",
            "swin_small",
            "swin_base",
            "swin_large",
        ],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""",
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""",
    )
    parser.add_argument(
        "--out_dim",
        default=65536,
        type=int,
        help="""Dimensionality of
        the MaSSL head output. For complex and large datasets large values (like 65k) work well.""",
    )
    parser.add_argument(
        "--norm_last_layer",
        default=True,
        type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the MaSSL head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""",
    )
    parser.add_argument(
        "--momentum_teacher",
        default=0.99,
        type=float,
        help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""",
    )
    parser.add_argument(
        "--use_bn_in_head",
        default=True,
        type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)",
    )

    # Training/Optimization parameters
    parser.add_argument(
        "--use_fp16",
        type=utils.bool_flag,
        default=True,
        help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.000001,
        help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""",
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=0.000001,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=3.0,
        help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="""Number of steps to 
                        accumulate gradients before an optimization step.""",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=32,
        type=int,
        help="Per-GPU batch-size : number of distinct images loaded on one GPU.",
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs of training."
    )
    parser.add_argument(
        "--lr",
        default=0.3,
        type=float,
        help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=10,
        type=int,
        help="Number of epochs for the linear learning-rate warm up.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0048,
        help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""",
    )
    parser.add_argument(
        "--optimizer",
        default="lars",
        type=str,
        choices=["adamw", "sgd", "lars"],
        help="""Type of optimizer. We recommend using adamw with ViTs.""",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        help="""Drop path rate for student network.""",
    )
    parser.add_argument(
        "--partition_size", default=1024, type=int, help="The size of the subgroups."
    )
    parser.add_argument(
        "--bottleneck_dim",
        default=256,
        type=int,
        help="Dimensionality of the embedding vector.",
    )
    parser.add_argument(
        "--student_temp",
        default=0.1,
        type=float,
        help="Temperature for student logits prior to softmax.",
    )
    parser.add_argument(
        "--koleo_loss_weight",
        default=0.0,
        type=float,
        help="Weight for the koleo loss contribution",
    )
    parser.add_argument(
        "--entropy_loss_weight",
        default=0.0,
        type=float,
        help="Weight for the entropy loss contribution",
    )

    # Temperature teacher parameters
    parser.add_argument(
        "--warmup_teacher_temp",
        default=0.04,
        type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""",
    )
    parser.add_argument(
        "--teacher_temp",
        default=0.07,
        type=float,
        help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""",
    )
    parser.add_argument(
        "--warmup_teacher_temp_epochs",
        default=30,
        type=int,
        help="Number of warmup epochs for the teacher temperature (Default: 30).",
    )

    # Multi-crop parameters
    parser.add_argument(
        "--global_crops_scale",
        type=float,
        nargs="+",
        default=(0.2, 1.0),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""",
    )
    parser.add_argument(
        "--local_crops_number",
        type=int,
        default=6,
        help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """,
    )
    parser.add_argument(
        "--local_crops_scale",
        type=float,
        nargs="+",
        default=(0.05, 0.2),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""",
    )

    # Misc
    parser.add_argument(
        "--data_path",
        default="../../../../../../data/ImageNet2012/train",
        type=str,
        help="Please specify path to the ImageNet training data.",
    )
    parser.add_argument(
        "--resume_from_dir",
        default=".",
        type=str,
        help="Path to save logs and checkpoints.",
    )
    parser.add_argument(
        "--saveckp_freq", default=50, type=int, help="Save checkpoint every x epochs."
    )
    parser.add_argument(
        "--print_freq", default=50, type=int, help="Save checkpoint every x epochs."
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--num_workers",
        default=7,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument(
        "--local-rank",
        default=0,
        type=int,
        help="Please ignore and do not set this argument.",
    )
    parser.add_argument(
        "--use_masked_im_modeling",
        default=False,
        type=utils.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)",
    )
    parser.add_argument(
        "--use_mean_pooling",
        default=False,
        type=utils.bool_flag,
        help="Whether to use mean average pooling instead of returning the CLS token (Default: False)",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=0,
        help="Whether to use mean average pooling instead of returning the CLS token (Default: False)",
    )
    return parser


def train_massl(args):

    # init distributed pipeline
    # utils.init_distributed_mode(args)
    distr_env = DistributedEnviron()

    # init distributed pipeline
    # utils.init_distributed_mode(args)
    dist.init_process_group(backend="nccl")
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    args.gpu = distr_env.local_rank
    torch.cuda.set_device(args.gpu)
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    utils.setup_for_distributed(args.rank == 0)

    # fix random seeds
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()), flush=True)
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())),
        flush=True,
    )
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationMaSSL(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.", flush=True)

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is of hierechical features (i.e. swin_tiny, swin_small, swin_base)
    if args.arch in models.__dict__.keys() and "swin" in args.arch:
        student = models.__dict__[args.arch](
            window_size=args.window_size,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            window_size=args.window_size,
            drop_path_rate=0.0,
            return_all_tokens=True,
        )
        embed_dim = student.num_features
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    elif args.arch in models.__dict__.keys():
        student = models.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=False,
            masked_im_modeling=args.use_masked_im_modeling,
            use_mean_pooling=args.use_mean_pooling,
        )
        teacher = models.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=0.0,
            return_all_tokens=False,
            masked_im_modeling=False,
            use_mean_pooling=args.use_mean_pooling,
        )
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(
        student,
        ProjectionHead(
            in_dim=embed_dim,
            use_bn=args.use_bn_in_head,
            bottleneck_dim=args.bottleneck_dim,
        ),
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        ProjectionHead(
            in_dim=embed_dim,
            use_bn=args.use_bn_in_head,
            bottleneck_dim=args.bottleneck_dim,
        ),
    )
    print(f"{student}", flush=True)
    print(f"{teacher}", flush=True)

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(
        f"Student and Teacher are built: they are both {args.arch} network.", flush=True
    )

    # total number of crops = 2 global crops + local_crops_number
    args.ncrops = args.local_crops_number + 2

    # ============ preparing loss ... ============
    criterion = Criterion()
    memory_bank = MemoryBank(args.ncrops, args.out_dim, args.bottleneck_dim)
    memory_bank = memory_bank.cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params_groups, lr=0, momentum=0.9
        )  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches

    # init optimizer
    optimizer.zero_grad()

    # for mixed precision training
    fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr
        * (
            args.gradient_accumulation_steps
            * args.batch_size_per_gpu
            * utils.get_world_size()
        )
        / 256.0,  # linear scaling rule
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    teacher_temp_schedule = utils.cosine_scheduler(
        base_value=args.teacher_temp,
        final_value=args.teacher_temp,
        epochs=args.epochs,
        niter_per_ep=len(data_loader),
        warmup_epochs=args.warmup_teacher_temp_epochs,
        start_warmup_value=args.warmup_teacher_temp,
    )

    ko_weight_schedule = utils.cosine_scheduler(
        base_value=args.koleo_loss_weight,
        final_value=args.koleo_loss_weight,
        epochs=args.epochs,
        niter_per_ep=len(data_loader),
        warmup_epochs=10,
        start_warmup_value=0,
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader)
    )
    print(f"Loss, optimizer and schedulers ready.", flush=True)

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.resume_from_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        memory_bank=memory_bank,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
    )
    start_epoch = to_restore["epoch"]

    summary_writer = None
    root_dir = "/MaSSL"
    if utils.is_main_process():
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        summary_writer = SummaryWriter(
            log_dir=os.path.join(
                root_dir, "runs", current_time + "_" + socket.gethostname()
            )
        )
        shutil.copyfile(
            os.path.join(root_dir, "main_massl.py"),
            os.path.join(summary_writer.log_dir, "main_massl.py"),
        )
        shutil.copyfile(
            os.path.join(root_dir, "utils.py"),
            os.path.join(summary_writer.log_dir, "utils.py"),
        )
        shutil.copyfile(
            os.path.join(root_dir, "head.py"),
            os.path.join(summary_writer.log_dir, "head.py"),
        )
        shutil.copyfile(
            os.path.join(root_dir, "loss.py"),
            os.path.join(summary_writer.log_dir, "loss.py"),
        )
        shutil.copyfile(
            os.path.join(root_dir, "memory_bank.py"),
            os.path.join(summary_writer.log_dir, "memory_bank.py"),
        )
        shutil.copyfile(
            os.path.join(root_dir, "random_partition.py"),
            os.path.join(summary_writer.log_dir, "random_partition.py"),
        )
        shutil.copyfile(
            os.path.join(root_dir, "./models/vision_transformer.py"),
            os.path.join(summary_writer.log_dir, "vision_transformer.py"),
        )
        shutil.copyfile(
            os.path.join(root_dir, "koleo_loss.py"),
            os.path.join(summary_writer.log_dir, "koleo_loss.py"),
        )
        stats_file = open(
            os.path.join(summary_writer.log_dir, "stats.txt"), "a", buffering=1
        )
        print(" ".join(sys.argv), flush=True)
        print(" ".join(sys.argv), file=stats_file, flush=True)
        with open(os.path.join(summary_writer.log_dir, "metadata.txt"), "a") as f:
            yaml.dump(args, f, allow_unicode=True)
            f.write(str(student))
            f.write(str(teacher))

    random_partitioning = RandomPartition(args).cuda()

    start_time = time.time()
    print("Starting MaSSL training !", flush=True)
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of MaSSL ... ============
        train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            criterion,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            teacher_temp_schedule,
            ko_weight_schedule,
            epoch,
            fp16_scaler,
            random_partitioning,
            memory_bank,
            summary_writer,
            args,
        )

        # ============ writing logs ... ============
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "memory_bank": memory_bank.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        if summary_writer is not None:
            utils.save_on_master(
                save_dict, os.path.join(summary_writer.log_dir, "checkpoint.pth")
            )
        if args.saveckp_freq and (epoch + 1) % args.saveckp_freq == 0:
            if summary_writer is not None:
                utils.save_on_master(
                    save_dict,
                    os.path.join(summary_writer.log_dir, f"checkpoint{epoch:04}.pth"),
                )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str), flush=True)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [
            correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk
        ]


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    criterion_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    teacher_temp_schedule,
    ko_weight_schedule,
    epoch,
    fp16_scaler,
    random_partitioning,
    memory_bank,
    summary_writer,
    args,
):

    koleo_loss = KoLeoLoss()
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    learning_rates = AverageMeter("LR", ":.4e")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    for i, (images, _) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        it = len(data_loader) * epoch + i  # global training iteration

        lr = lr_schedule[it]
        m = momentum_schedule[it]
        teacher_temp = teacher_temp_schedule[it]

        learning_rates.update(lr)

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        # update learning rate according to schedule
        for j, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr
            if j == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
            student_output = student(images)
            teacher_output = teacher(
                images[:2]
            )  # only the 2 global views pass through the teacher

            # koleo loss must come before random partition
            ko = 0
            for p in student_output.chunk(args.ncrops)[:2]:
                ko += koleo_loss(p)
            ko /= 2

            student_output = memory_bank(student_output)
            teacher_output = memory_bank(teacher_output, update=True)

            student_output /= args.student_temp
            teacher_output /= teacher_temp

            ## random Parition strategy
            student_output, teacher_output = random_partitioning(
                student_output, teacher_output, args.partition_size
            )

            student_probs = torch.cat(student_output, dim=1).flatten(
                0, 1
            )  # [N_CROPS * N_BLOCKS * BS, BLOCK_SIZE)
            student_probs = torch.softmax(student_probs, dim=-1)

            ce = criterion_loss(student_output, teacher_output)
            ce /= args.gradient_accumulation_steps

        loss = ce + args.koleo_loss_weight * ko

        optimizer.zero_grad()

        # clip gradients
        fp16_scaler.scale(loss).backward()
        if args.clip_grad:
            fp16_scaler.unscale_(
                optimizer
            )  # unscale the gradients of optimizer's assigned params in-place
            param_norms = utils.clip_gradients(student, args.clip_grad)
        fp16_scaler.step(optimizer)
        fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            for param_q, param_k in zip(
                student.module.parameters(), teacher_without_ddp.parameters()
            ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        losses.update(loss.item(), images[0].size(0))

        if summary_writer is not None and it % args.print_freq == 0:
            acc1, acc5 = accuracy(
                student_output[0][0],
                torch.argmax(teacher_output[1][0], dim=1),
                topk=(1, 5),
            )

            teacher_probs = torch.cat(teacher_output, dim=1).flatten(0, 1)
            teacher_probs = torch.softmax(teacher_probs, dim=-1)

            # summary_writer.add_scalar(f"metric/student/avg_max_score", torch.max(student_probs, dim=-1).values.mean().item(), it)
            # summary_writer.add_scalar(f"metric/student/avg_min_score", torch.min(student_probs, dim=-1).values.mean().item(), it)

            # summary_writer.add_scalar(f"metric/teacher/avg_max_score", torch.max(teacher_probs, dim=-1).values.mean().item(), it)
            # summary_writer.add_scalar(f"metric/teacher/avg_min_score", torch.min(teacher_probs, dim=-1).values.mean().item(), it)

            summary_writer.add_scalar(f"metric/ko/weight", ko_weight_schedule[it], it)

            summary_writer.add_scalar("loss/total", loss.item(), it)
            summary_writer.add_scalar("loss/koleo", ko.item(), it)
            summary_writer.add_scalar("loss/ce", ce.item(), it)
            # summary_writer.add_scalar("loss/entropy", entropy.item(), it)

            # summary_writer.add_scalar("metric/momentum", m, it)
            # summary_writer.add_scalar("metric/lr", lr, it)
            summary_writer.add_scalar("acc/top1", acc1, it)
            summary_writer.add_scalar("acc/top5", acc5, it)
            # summary_writer.add_scalar("metric/teacher_temp", teacher_temp, it)

            n_protos = student_probs.shape[1]
            summary_writer.add_histogram(
                f"dist/probs/blocks_{n_protos}", torch.argmax(student_probs, dim=-1), it
            )
            summary_writer.add_histogram(
                f"dist/targets/blocks_{n_protos}",
                torch.argmax(teacher_probs, dim=-1),
                it,
            )
            # summary_writer.add_histogram(f"dist/targets/memory", torch.argmax(teacher_mem_output, dim=-1), it)
            # summary_writer.add_histogram(f"dist/probs/memory", torch.argmax(student_mem_output, dim=-1), it)

            progress.display(i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class DataAugmentationMaSSL(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                transforms.RandomApply([utils.GaussianBlur([0.1, 2.0])], p=1.0),
                normalize,
            ]
        )
        # second global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                transforms.RandomApply([utils.GaussianBlur([0.1, 2.0])], p=0.1),
                transforms.RandomApply([utils.Solarize()], p=0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    96, scale=local_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                transforms.RandomApply([utils.GaussianBlur([0.1, 2.0])], p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MaSSL", parents=[get_args_parser()])
    args = parser.parse_args()
    train_massl(args)
