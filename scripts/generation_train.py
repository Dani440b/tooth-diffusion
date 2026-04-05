"""
A script for training a diffusion model to unconditional image generation.
"""

import argparse
import numpy as np
import random
import sys
import torch as th

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import logger
from guided_diffusion.toothloader import ToothVolumes
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          args_to_dict,
                                          add_dict_to_argparser)
from guided_diffusion.train_util import TrainLoop
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import os


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize distributed process group. Choose backend based on CUDA availability.
    use_cuda = th.cuda.is_available()
    backend = 'nccl' if use_cuda else 'gloo'
    dist.init_process_group(backend=backend, init_method='env://')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set device (only set CUDA device when available)
    if use_cuda:
        th.cuda.set_device(local_rank)
        device = th.device(f'cuda:{local_rank}')
        th.backends.cudnn.benchmark = True
        th.backends.cuda.matmul.allow_tf32 = True
        th.backends.cudnn.allow_tf32 = True
        try:
            th.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        device = th.device('cpu')

    if args.auto_vram and use_cuda:
        local_total_gb = th.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        total_gb_tensor = th.tensor([local_total_gb], device=device, dtype=th.float32)
        if dist.is_initialized():
            dist.all_reduce(total_gb_tensor, op=dist.ReduceOp.MIN)
        total_gb = float(total_gb_tensor.item())

        # Tune once from the smallest visible GPU across ranks so every process builds the same model.
        if total_gb <= 16:
            args.num_channels = 16
            args.channel_mult = "1,1,2,2,4,4"
            args.batch_size = 1
            args.microbatch = 1
        elif total_gb <= 24:
            args.num_channels = 32
            args.channel_mult = "1,2,2,4,4,4"
            args.batch_size = 1
            args.microbatch = 1
        elif total_gb <= 40:
            args.num_channels = 64
            args.channel_mult = "1,2,2,4,4,4"
            args.batch_size = 1
            args.microbatch = 1
        else:
            args.num_channels = 64
            args.channel_mult = "1,2,2,4,4,4"
        args.use_checkpoint = True
        if rank == 0:
            logger.log(
                f"Auto VRAM tuning: {total_gb:.1f} GB -> "
                f"channels={args.num_channels}, channel_mult={args.channel_mult}, "
                f"batch_size={args.batch_size}, microbatch={args.microbatch}, "
                f"fp16={args.use_fp16}, checkpoint={args.use_checkpoint}"
            )
    
    summary_writer = None
    if args.use_tensorboard and rank == 0:
        logdir = None
        if args.tensorboard_path:
            logdir = args.tensorboard_path
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()

    # Initialize wandb if requested (only on rank 0)
    wandb_run = None
    if args.use_wandb and rank == 0:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project or None,
                                   entity=args.wandb_entity or None,
                                   name=args.wandb_run_name or None,
                                   config=vars(args))
        except Exception as e:
            logger.log(f"Failed to initialize wandb: {e}")

    logger.log(f"Rank {rank}/{world_size}: Creating model and diffusion...")

    # Compute model input channels from active conditioning configuration.
    if args.metadata_as_channels:
        cond_ch = 8 if args.conditioning_image != "none" else 0
        args.in_channels = 8 + cond_ch + int(args.metadata_channels_dim)
        logger.log(
            f"Using metadata channel conditioning: in_channels={args.in_channels} "
            f"(x_t=8, cond={cond_ch}, metadata={args.metadata_channels_dim})"
        )

    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    
    # Model and diffusion creation
    model, diffusion = create_model_and_diffusion(**arguments)
    model = model.to(device)
    # Wrap model for distributed training. For CUDA use device_ids, for CPU let DDP handle CPU tensors.
    if use_cuda:
        model = th.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:
        model = th.nn.parallel.DistributedDataParallel(model)

    # logger.log("Number of trainable parameters: {}".format(np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()))
    logger.log(f"Rank {rank}: Creating schedule sampler...")
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, maxt=args.diffusion_steps)

    if args.dataset == 'mri':
        assert args.image_size in [256], "We currently just support image sizes 256"
        ds = ToothVolumes(
            directory=args.data_dir,
            metadata_path=args.meta_data,
            test_flag=False,
            normalize=(lambda x: 2 * x - 1) if args.renormalize else None,
            mode='train',
            img_size=args.image_size,
            noisy_dir=args.noisy_dir or None,
            noisy_meta_data=args.noisy_meta_data or None,
        )

    else:
        print("We currently just support the datasets: mri")

    logger.log(f"Rank {rank}: Creating dataset...")
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
    )
    if args.num_workers > 0:
        dataloader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )
    datal = DataLoader(ds, **dataloader_kwargs)

    logger.log(f"Rank {rank}: Start training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        wandb_run=wandb_run,
        mode='default',
        target=args.target,
        training_mode=args.training_mode,
        conditioning_image=args.conditioning_image,
        lambda_mask=args.lambda_mask,
        lambda_quality=args.lambda_quality,
        lambda_quality_overall=args.lambda_quality_overall,
    ).run_loop()
    
    dist.destroy_process_group()

def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        meta_data="",
        noisy_dir="",        # path to torchio_preproc/output/train for DCP-Diff style conditioning
        noisy_meta_data="",  # path to augmentation_metadata.csv (defaults to noisy_dir/augmentation_metadata.csv)
        target="",
        training_mode="train",
        conditioning_image="none",
        schedule_sampler="uniform",
        lr=1e-4,
        lambda_mask=10.0,
        lambda_quality=1.0,
        lambda_quality_overall=1.0,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        beta_min=0.1,
        beta_max=20.0,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        auto_vram=False,
        dataset='tooth',
        use_tensorboard=True,
        tensorboard_path='',  # set path to existing logdir for resuming
        use_wandb=False,
        wandb_project='tooth-diffusion',
        wandb_entity='',
        wandb_run_name='',
        devices=[0],
        dims=3,
        learn_sigma=False,
        num_groups=32,
        channel_mult="1,2,2,4,4",
        in_channels=8,
        metadata_as_channels=True,
        metadata_channels_dim=13,
        out_channels=8,
        bottleneck_attention=False,
        num_workers=0,
        mode='default',
        renormalize=True,
        additive_skips=False,
        use_freq=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
