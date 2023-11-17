In order to run the scripts, make sure that all of thes eenvironment variables are provided:

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0" &&
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix" &&
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

export HF_DATASETS_CACHE=/future/u/suppakit/hf-cache
export WANDB_DIR=future/u/suppakit/wandblogs
export HF_HOME=/future/u/suppakit/hf-home
export TRANSFORMERS_CACHE=/future/u/suppakit/transformers-cache
export WANDB_CONFIG_DIR=/future/u/suppakit/wandb_config
