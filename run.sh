# general settings
GPU=0,1;                  # gpu to use
NUMBERofGPUS=2            # number of gpus to use, should reflect what you specify on GPU variable
MASTERPORT=${MASTERPORT:-12345};  # master port for distributed training
SEED=42;                  # randomness seed for sampling
CHANNELS=32;              # number of model base channels (lower to reduce VRAM)
DATASET='mri';            
MODEL='ours_wnet_256';    # 'ours_wnet_256' currently only supported for wnet_256

# general settings for training and sampling specified when running the script
MODE=${1:-train}          # train vs sample
TARGET=${2:-mri}        
RESUME_CHECKPOINT=${3:-}  
CONDITIONING_IMAGE='dcp'

# settings for sampling/inference
ITERATIONS=1200;        # training iteration (as a multiple of 1k) checkpoint to use for sampling
SAMPLING_STEPS=2        # number of steps for accelerated sampling, 0 for the default 1000
RUN_DIR="";             # tensorboard dir to be set for the evaluation

# detailed settings, currently only for wnet_256 
if [[ $MODEL == 'ours_wnet_256' ]]; then
  echo "MODEL: WDM (WavU-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE=256;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=1;
  IN_CHANNELS=8;  
else
  echo "MODEL TYPE NOT FOUND -> Check the supported configurations again";
fi

# If conditioning image is none, then 8 or else 16 
if [[ $CONDITIONING_IMAGE != 'none' ]]; then
  IN_CHANNELS=16
fi

# some information and overwriting batch size for sampling
# (overwrite in case you want to sample with a higher batch size)
# no need to change for reproducing
if [[ $MODE == 'sample' ]]; then
  echo "MODE: sample"
  BATCH_SIZE=1;
  DATA_DIR=prep_data/train;
  META_DATA=prep_data/metadata.csv
elif [[ $MODE == 'train' ]]; then
  if [[ $DATASET == 'mri' ]]; then
    echo "MODE: training";
    echo "DATASET: MRI"
    DATA_DIR=prep_data/train;
    META_DATA=prep_data/metadata.csv
    VAL_DATA_DIR=${VAL_DATA_DIR:-}
    VAL_NOISY_DIR=${VAL_NOISY_DIR:-}
    VAL_NOISY_META_DATA=${VAL_NOISY_META_DATA:-}
    VAL_SPLIT=${VAL_SPLIT:-0.1}
    # DCP-Diff style conditioning: noisy MRI as conditioning guide, clean MRI as diffusion target.
    NOISY_DIR=${NOISY_DIR:-prep_data/train_augmented}
  else
    echo "DATASET NOT FOUND -> Check the supported datasets again";
  fi
fi

COMMON="
--beta_min=0.1
--beta_max=20.0
--dataset=${DATASET}
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=2
--num_heads=1
--timestep_respacing=
--learn_sigma=False
--use_scale_shift_norm=True
--attention_resolutions=
--channel_mult=${CHANNEL_MULT}
--diffusion_steps=2
--noise_schedule=vp_sde
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=${BATCH_SIZE}
--num_groups=32
--in_channels=${IN_CHANNELS}
--out_channels=8
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=${ADDITIVE_SKIP}
--use_freq=${USE_FREQ}
--predict_xstart=True
"
TRAIN="
--data_dir=${DATA_DIR}
--meta_data=${META_DATA}
--noisy_dir=${NOISY_DIR}
--val_data_dir=${VAL_DATA_DIR}
--val_noisy_dir=${VAL_NOISY_DIR}
--val_noisy_meta_data=${VAL_NOISY_META_DATA}
--val_split=${VAL_SPLIT}
--target=${TARGET}
--training_mode=${MODE}
--conditioning_image=${CONDITIONING_IMAGE}
--resume_checkpoint=${RESUME_CHECKPOINT}
--resume_step=0
--image_size=${IMAGE_SIZE}
--use_fp16=False
--use_checkpoint=True
--microbatch=1
--auto_vram=True
--lr=1e-5
--lambda_mask=10.0
--lambda_quality=1.0
--lambda_quality_overall=1.0
--save_interval=5000
--validation_interval=5000
--early_stop_patience=10
--num_workers=4
--devices=${GPU}
"
SAMPLE="
--data_dir=${DATA_DIR}
--meta_data=${META_DATA}
--data_mode=${DATA_MODE}
--seed=${SEED}
--image_size=${IMAGE_SIZE}
--use_fp16=False
--model_path=${RESUME_CHECKPOINT}
--devices=${GPU}
--output_dir=./results/
--num_samples=1000
--use_ddim=False
--sampling_steps=${SAMPLING_STEPS}
--clip_denoised=True
"
# forward any extra args passed after the first three positional args
EXTRA_ARGS="${@:4}"
# run the python scripts
if [[ $MODE == 'train' ]]; then
  echo "Mode: $MODE";
  echo "Target: $TARGET";
  echo "Condition image: $CONDITIONING_IMAGE";
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 \
  torchrun --nproc_per_node=$NUMBERofGPUS --master_port=$MASTERPORT \
    scripts/generation_train.py $TRAIN $COMMON $EXTRA_ARGS
else
  python scripts/reconstruction/generation_sample_add.py $SAMPLE $COMMON $EXTRA_ARGS;
fi
