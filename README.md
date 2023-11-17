In order to run the scripts, make sure that all of thes eenvironment variables are provided:

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0" &&
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix" &&
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

export HF_DATASETS_CACHE=/future/u/suppakit/hf-cache
export WANDB_DIR=future/u/suppakit/wandblogs
export HF_HOME=/future/u/suppakit/hf-home
export TRANSFORMERS_CACHE=/future/u/suppakit/transformers-cache
export WANDB_CONFIG_DIR=/future/u/suppakit/wandb_config

# To run training:

accelerate launch train_text_to_image_lora_sdxl.py \
--pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=1024 \
  --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="a cute Sundar Pichai creature" \
  --validation_epochs 5 \
  --checkpointing_steps=5000 \
  --output_dir="./model-output" \
  --rank 256 \
    --cache_dir data-cache