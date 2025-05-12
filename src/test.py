"""
Script for testing models head-to-head or just doing inference.
See the input arguments for more details on the options.
"""

import argparse
import datasets
from datetime import datetime
import diffusers
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import numpy as np
import os
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers

from preference_pick_score import PreferenceFromPickScore
from prompt_dataset import DiffusionDBPrompts

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_images_batch(write_dir, prompts_file, images, prompts, batch_id, batch_size, model_name=""):
    with open(prompts_file, 'a') as f:
        for it, (img, prompt) in enumerate(zip(images,prompts)):
            image_id = batch_id * batch_size + it
            image_id = f"M{model_name}I{image_id}"
            img.save(Path(write_dir) / f"images/img_{image_id}.png")
            f.write(f"{image_id} <sep> {prompt}\n")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Testing script. Select between one of the test modes."
    )

    parser.add_argument(
        "--write_dir",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory containing the model to evaluate.",
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        choices=["inference", "inference_ref", "inference_dpo", "vs_ref", "vs_model", "vs_dpo", "dpo_vs_ref"],
        default="inference",
        help="Testing mode. Can be 'inference' if you just want to generate some images, 'vs_ref' if you want to compare it against the base model."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computing device.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="None",
        choices=["fp16"],
        help="Whether to use fp16.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Computing device.",
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default="./test_prompts.parquet",
        help="Path to the *TEST* prompts file.",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Maximum number of validation prompts.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--win_threshold",
        type=float,
        default=0.55,
        help="If the preference score is at least this, we consider it a win. Has to be between 0.5 and 1.",
    )
    parser.add_argument(
        "--second_model",
        type=str,
        default=None,
        help="Directory containing the model to compare against.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=12,
        help="The generated images will be saved each x iterations."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    if args.test_mode == "vs_model":
        assert args.second_model is not None

    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

    os.makedirs(args.write_dir, exist_ok=True)  
    os.makedirs(Path(args.write_dir) / "images", exist_ok=True)
    prompts_file = Path(args.write_dir) / "prompts.txt"

    # TODO!
    # Check if this directory was used for inference previously
    # if prompts_file.is_file():
    #     pass

    if args.seed is not None:
        set_seed(args.seed) 

    if "vs" in args.test_mode:
        assert 0.5 <= args.win_threshold < 1
        preference_model = PreferenceFromPickScore(device=args.device)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_name, 
        torch_dtype=weight_dtype,
        local_files_only=True,
    ).to(args.device)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.safety_checker = lambda images, clip_input: (images, [False] * images.shape[0])

    if args.test_mode == "inference_dpo" or args.test_mode == "dpo_vs_ref":
        unet_id = "mhdang/dpo-sd1.5-text2image-v1"
        unet = UNet2DConditionModel.from_pretrained(unet_id, subfolder="unet", torch_dtype=torch.float16).to(args.device)
        pipeline.unet = unet
    elif args.test_mode != "inference_ref":
        assert args.model_dir is not None
        pipeline.load_lora_weights(args.model_dir)

    if "vs" in args.test_mode:
        pipeline_2 = StableDiffusionPipeline.from_pretrained(
            args.model_name, 
            torch_dtype=weight_dtype,
            local_files_only=True,
        ).to(args.device)

        pipeline_2.vae.requires_grad_(False)
        pipeline_2.text_encoder.requires_grad_(False)
        pipeline_2.unet.requires_grad_(False)
        pipeline_2.set_progress_bar_config(disable=True)
        pipeline_2.safety_checker = lambda images, clip_input: (images, [False] * images.shape[0])
        
        if args.test_mode == "vs_model":
            pipeline_2.load_lora_weights(args.second_model)
        elif args.test_mode == "vs_dpo":
            unet_id = "mhdang/dpo-sd1.5-text2image-v1"
            unet = UNet2DConditionModel.from_pretrained(unet_id, subfolder="unet", torch_dtype=torch.float16).to(args.device)
            pipeline_2.unet = unet


    prompts = DiffusionDBPrompts(args.prompts_path)
    if args.max_prompts is not None:
        prompts.truncate(args.max_prompts)

    dataloader = DataLoader(prompts, batch_size=args.batch_size, shuffle=False)

    if "inference" in args.test_mode:
        for batch_id, prompt_batch in enumerate(tqdm(dataloader)):
            images = pipeline(prompt_batch).images
            save_images_batch(
                args.write_dir, 
                prompts_file,
                images,
                prompt_batch,
                batch_id,
                args.batch_size
            )
    
    elif "vs" in args.test_mode:
        acc_preference = 0
        acc_wins = 0
        acc_draws = 0

        for it, prompts in enumerate(tqdm(dataloader)):
            images_1 = pipeline(prompts).images
            images_2 = pipeline_2(prompts).images
            preference_probs = preference_model(prompts, images_1, images_2).cpu()
            acc_preference += preference_probs.mean().item()
            acc_wins += (preference_probs > args.win_threshold).sum().item() / preference_probs.shape[0]
            acc_draws += ((preference_probs <= args.win_threshold).sum().item() - (preference_probs < 1-args.win_threshold).sum().item()) / preference_probs.shape[0]
            
            if it % args.save_interval == 0:
                save_images_batch(
                    args.write_dir, 
                    prompts_file,
                    images_1,
                    prompts,
                    it,
                    args.batch_size,
                    "1"
                )
                save_images_batch(
                    args.write_dir, 
                    prompts_file,
                    images_2,
                    prompts,
                    it,
                    args.batch_size,
                    "2"
                )

        avg_preference = acc_preference / len(dataloader)
        avg_wins = acc_wins / len(dataloader)
        avg_draws = acc_draws / len(dataloader)

        with open(Path(args.write_dir) / "vs_logs.txt", "a") as f:
            f.write(f"Date: {datetime.now().date()}\n")
            f.write(f"Model directory: {args.model_dir}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Win threshold: {args.win_threshold}\n")
            f.write(f"Average preference: {avg_preference}\n")
            f.write(f"Win %: {round(avg_wins*100, 2)}\n")
            f.write(f"Draw %: {round(avg_draws*100, 2)}\n")
            f.write("-"*25)
            f.write("\n")
    

if __name__ == "__main__":
    main()