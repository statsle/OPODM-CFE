"""
Online Preference Optimization for Diffusion Models with Classifier-Free Exploration
Main script for alignment (fine-tuning) of Stable Diffusion 1.5.
"""

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
import argparse
from copy import deepcopy
import datasets
import diffusers
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
import logging
import math
import numpy as np
from omegaconf import OmegaConf
import os
from pathlib import Path
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
# import re
import shutil
import subprocess
import torch
from torch.amp import autocast
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import transformers
import wandb
import warnings

from diffusers_patch import SDPipeline_Patch
from preference_pick_score import PreferenceFromPickScore
from prompt_dataset import DiffusionDBPrompts
from validate import validation_loop

logger = get_logger(__name__, log_level="INFO")

def nvidia_smi_gpu_memory_stats():
    """
    Parse the nvidia-smi output and extract the memory used stats.
    """
    out_dict = {}
    sp = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
    )
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split("\n")
    out_dict = {}
    for item in out_list:
        if " MiB" in item:
            gpu_idx, mem_used = item.split(',')
            gpu_key = f"gpu_{gpu_idx}_mem_used_gb"
            out_dict[gpu_key] = int(mem_used.strip().split(" ")[0]) / 1024

    return out_dict


def get_nvidia_smi_gpu_memory_stats_str():
    return f"nvidia-smi stats: {nvidia_smi_gpu_memory_stats()}"

def parse_args():
    """Parses command-line arguments for configuration overrides."""
    parser = argparse.ArgumentParser(description="Stable Diffusion Fine-tuning with Accelerate & WandB")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--data_path", type=str, default="", help="Path to parquet dataset.")
    parser.add_argument("--truncate_dataset", type=int, default=None, help="Whether to make dataset shorter.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--snr_gamma", type=float, default=None, help="Min-SNR gamma hparameter.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="DPO",
        choices=("DPO", "IPO", "SLiC", "exp"),
        help=("Loss function f to be applied to \\beta\\phi_\\theta within the GPO framework.")
    )
    parser.add_argument("--gpo_beta", type=float, default=0.1, help="gpo_beta hyperparameter for loss function")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--quality_threshold", default=0.5, type=float, help="Threshold for minimum preference in training.")
    parser.add_argument("--validation_threshold", default=0.55, type=float, help="Threshold for minimum preference in validation.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=False,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--write_dir",
        type=str,
        default=None,
        help="Output directory.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="Checkpoint path or 'latest'. If empty no checkpoint is resumed.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution of the images (single int because we use square resolution).",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        default=False,
        help="If enabled, images used to train will have a center crop, otherwise it will be a random one.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        default=False,
        help="If enabled, images used to train will have the chance of being horizontally flipped.",
    )
    return parser.parse_args()

def load_config():
    """Loads configuration from file and applies command-line overrides."""
    args = parse_args()
    config = OmegaConf.load(args.config)  # Load YAML file
    cli_args = OmegaConf.from_cli()       # Get command-line arguments
    return OmegaConf.merge(config, cli_args)  # Merge, with CLI args taking priority

def main():

    config = load_config()

    assert(not config.quality_threshold or 0.5 <= config.quality_threshold < 1)
    assert(not config.validation_threshold or 0.5 <= config.validation_threshold < 1)

    if config.write_dir is not None:
        os.makedirs(config.write_dir, exist_ok=True)  

    logging_dir = Path(config.write_dir) / "logs"
    
    project_config = ProjectConfiguration(project_dir=config.write_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with="wandb",
        project_config=project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        # warnings.filterwarnings(
        #     'ignore', 
        #     message=r'\n?The following part of your input was truncated because CLIP can only handle sequences up to 77 tokens:.*',
        # )
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if config.seed is not None:
        set_seed(config.seed, device_specific=True) 

    logger.info(f"Loading models...")

    preference_model = PreferenceFromPickScore(device=accelerator.device)
    
    logger.info(f"Loaded preference model.")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load pre-trained SD pipeline
    pipeline = SDPipeline_Patch.from_pretrained(
        config.model_name, 
        torch_dtype=weight_dtype,
        local_files_only=True,
    ).to(accelerator.device)

    logger.info(f"Loaded {config.model_name}.")

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)

    pipeline.register_modules(
        new_unet = deepcopy(pipeline.unet)
    )
    pipeline.new_unet.requires_grad_(False)

    # Disable progress bar for training
    pipeline.set_progress_bar_config(disable=True)

    # Disable safety checker
    pipeline.safety_checker = lambda images, clip_input: (images, [False] * images.shape[0])

    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipeline.new_unet.add_adapter(lora_config)

    if config.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(pipeline.new_unet, dtype=torch.float32)

    lora_layers = filter(lambda p: p.requires_grad, pipeline.new_unet.parameters())

    # if config.gradient_checkpointing:
    #     pipeline.new_unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.lr = (
            config.lr * config.gradient_accumulation_steps * config.batch_size * accelerator.num_processes
        )   

    # Optimizer
    optimizer = torch.optim.AdamW(
        lora_layers, 
        lr=config.lr,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    # Load dataset
    prompts = DiffusionDBPrompts(config.data_path)
    if config.truncate_dataset is not None:
        prompts.truncate(config.truncate_dataset)

    train_ds, valid_ds = random_split(prompts, [.98, .02])
    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True) # missing num_workers
    valid_dataloader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False)

    train_transforms = transforms.Compose(
        [
            transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(config.resolution) if config.center_crop else transforms.RandomCrop(config.resolution),
            transforms.RandomHorizontalFlip() if config.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = config.lr_warmup_steps * accelerator.num_processes
    if config.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / config.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            config.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = config.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Use Accelerate to prepare models, optimizer, and dataloader
    pipeline.new_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pipeline.new_unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != config.max_train_steps * accelerator.num_processes:
            raise ValueError(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
        
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        # Initialize WandB logging with accelerate
        accelerator.init_trackers(
            project_name=config.wandb_project, 
            config=OmegaConf.to_container(config),
            init_kwargs={'wandb': {
                # 'mode': 'online',
                'resume': config.resume_run,
                'id': config.run_id,
                'dir': config.wandb_dir,
            }}
        )

    total_batch_size = config.batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_ds)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(config.write_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.write_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Training loop
    for epoch in range(first_epoch, config.num_train_epochs):
        pipeline.new_unet.train()
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        # pref_hist, pref_bins = np.histogram([], bins=np.arange(0.5,1.01,0.01)) # Used to se the distribution of preferences.
        for prompts in train_dataloader:
            local_batch_size = 0
            with accelerator.accumulate(pipeline.new_unet):
                with torch.no_grad():
                    # Generate an image from π_θ
                    with autocast(accelerator.device.type):
                        images_0 = pipeline(prompts, unet_name="new_unet").images
                        images_0 = pipeline(prompts, unet_name="new_unet").images
                        images_ref = pipeline(prompts, unet_name="unet").images
                        images_1 = pipeline(prompts, unet_name="new_unet", image=images_ref, guidance_scale=7.5).images

                        del images_ref

                        # Get preference probability
                        preference_probs = preference_model(prompts, images_0, images_1)
                
                filtered_prompts = []
                images_tensor_w = []
                images_tensor_l = []
                for i, (img_0, img_1) in enumerate(zip(images_0, images_1)):
                    if preference_probs[i] <= 1 - config.quality_threshold:
                        images_tensor_w.append(train_transforms(img_1))
                        images_tensor_l.append(train_transforms(img_0))
                        filtered_prompts.append(prompts[i])
                        preference_probs[i] = 1 - preference_probs[i]
                        local_batch_size += 1
                    elif preference_probs[i] >= config.quality_threshold:
                        images_tensor_w.append(train_transforms(img_0))
                        images_tensor_l.append(train_transforms(img_1))
                        filtered_prompts.append(prompts[i])
                        local_batch_size += 1

                del images_0
                del images_1
                del prompts
                torch.cuda.empty_cache()

                local_batches = accelerator.gather(torch.tensor([local_batch_size], dtype=torch.int8, device=accelerator.device))

                if not all(local_batches):
                    del images_tensor_w
                    del images_tensor_l
                    del filtered_prompts
                    torch.cuda.empty_cache()
                    continue

                images_tensor_w = torch.stack(images_tensor_w, dim=0).to(dtype=weight_dtype).to(accelerator.device)
                images_tensor_l = torch.stack(images_tensor_l, dim=0).to(dtype=weight_dtype).to(accelerator.device)

                with autocast(accelerator.device.type):
                    latents_w = pipeline.vae.encode(images_tensor_w).latent_dist.sample()
                    latents_w = latents_w * pipeline.vae.config.scaling_factor
                    latents_l = pipeline.vae.encode(images_tensor_l).latent_dist.sample()
                    latents_l = latents_l * pipeline.vae.config.scaling_factor

                    del images_tensor_w
                    del images_tensor_l
                    
                    latents = torch.cat((latents_w, latents_l),dim=0).to(accelerator.device)
                    # latents = latents * pipeline.scheduler.init_noise_sigma

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents_w)
                    bsz = latents_w.shape[0]

                    del latents_w
                    del latents_l
                    torch.cuda.empty_cache()

                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    noise = noise.repeat(2,1,1,1)
                    timesteps = timesteps.repeat(2)

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
                    
                    # Get the text embedding for conditioning
                    encoder_hidden_states, _ = pipeline.encode_prompt(
                        prompt=filtered_prompts,
                        device=accelerator.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False
                    )

                    encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)

                    model_pred = pipeline.new_unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                    ).sample

                model_losses = (model_pred - noise).pow(2).mean(dim=(1,2,3))
                model_losses_w, model_losses_l = model_losses.chunk(2)
                model_diff = model_losses_w - model_losses_l
                raw_model_loss = model_losses.mean()

                with torch.no_grad():
                    with autocast(accelerator.device.type):
                        ref_pred = pipeline.unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states,
                        ).sample.detach()
                    
                    ref_losses = (ref_pred - noise).pow(2).mean(dim=(1,2,3))
                    ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                    ref_diff = ref_losses_w - ref_losses_l
                    raw_ref_loss = ref_losses.mean()

                inside_term = -0.5 * config.gpo_beta * (model_diff - ref_diff)
                implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)

                if config.snr_gamma is not None:
                    timesteps, _ = timesteps.chunk(2)
                    snr = compute_snr(pipeline.scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    mse_loss_weights /= snr
                    inside_term *= mse_loss_weights

                if config.loss == "DPO":
                    loss = - F.logsigmoid(inside_term).mean()
                elif config.loss == "IPO":
                    loss = torch.square(inside_term - 1).mean()
                elif config.loss == "SLiC":
                    loss = torch.clamp(1 - inside_term, min=0).mean()
                elif config.loss == "exp":
                    loss = torch.exp(-inside_term).mean()
                else: 
                    raise ValueError("The provided loss is not impemented. Make sure to use 'DPO', 'IPO', 'SLiC' or 'exp'.")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.unsqueeze(0) * local_batch_size).sum() / local_batches.sum()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps
                
                avg_model_mse = accelerator.gather(raw_model_loss.unsqueeze(0) * local_batch_size).sum() / local_batches.sum()
                avg_ref_mse = accelerator.gather(raw_ref_loss.unsqueeze(0) * local_batch_size).sum() / local_batches.sum()
                avg_acc = accelerator.gather(implicit_acc.unsqueeze(0) * local_batch_size).sum() / local_batches.sum()
                implicit_acc_accumulated += avg_acc / config.gradient_accumulation_steps
                # all_preferences = accelerator.gather(preference_probs).cpu().tolist()
                # pref_hist += np.histogram(all_preferences, bins=pref_bins)[0]

                # Backpropagate and optimize
                accelerator.backward(loss)  # Compute gradients for all parameters

                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)

                optimizer.step()  # Update model parameters
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"diff_mse_unaccumulated": avg_model_mse - avg_ref_mse}, step=global_step)
                accelerator.log({"model_mse_unaccumulated": avg_model_mse}, step=global_step)
                accelerator.log({"implicit_acc_accumulated": implicit_acc_accumulated}, step=global_step)
                train_loss = 0.0
                implicit_acc_accumulated = 0.0

                if global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(config.write_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(config.write_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(config.write_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        unwrapped_unet = accelerator.unwrap_model(pipeline.new_unet)
                        unwrapped_unet = unwrapped_unet._orig_mod if is_compiled_module(unwrapped_unet) else unwrapped_unet
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0], 
                "implicit_acc": avg_acc
            }
            progress_bar.set_postfix(**logs)

            if global_step >= config.max_train_steps:
                break

        # Once we finish our epoch, we may do validation
        if accelerator.is_main_process and epoch % config.validation_epochs == 0:
            avg_pref, avg_wins, avg_draws = validation_loop(
                prompt_dataloader=valid_dataloader, 
                preference_model=preference_model, 
                pipeline=pipeline, 
                threshold=config.validation_threshold
            )
            accelerator.log({'average_preference': avg_pref, 'average_wins': avg_wins, 'average_draws': avg_draws}, step=epoch)

    # Save the lora layers once full training is finished
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline.new_unet = pipeline.new_unet.to(torch.float32)

        unwrapped_unet = accelerator.unwrap_model(pipeline.new_unet)
        unwrapped_unet = unwrapped_unet._orig_mod if is_compiled_module(unwrapped_unet) else unwrapped_unet
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=config.write_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

        # Final inference
        avg_pref, avg_wins, avg_draws = validation_loop(
            prompt_dataloader=valid_dataloader, 
            preference_model=preference_model, 
            pipeline=pipeline, 
            threshold=config.validation_threshold
        )
        accelerator.log({'final_average_preference': avg_pref, 'final_average_wins': avg_wins, 'final_average_draws': avg_draws})

    
    accelerator.end_training()

if __name__ == "__main__":
    main()
