def validation_loop(prompt_dataloader, preference_model, pipeline, threshold = 0.6):
    acc_preference = 0.0
    acc_wins = 0
    acc_draws = 0

    for prompts in prompt_dataloader:
        images_ref = pipeline(prompts).images
        images_new = pipeline(prompts, unet="new_unet").images
        preference_probs = preference_model(prompts, images_new, images_ref).cpu()
        acc_preference += preference_probs.mean().item()
        acc_wins += (preference_probs > threshold).sum().item() / preference_probs.shape[0]
        acc_draws += ((preference_probs <= threshold).sum().item() - (preference_probs < 1-threshold).sum().item()) / preference_probs.shape[0]

    return acc_preference / len(prompt_dataloader), acc_wins / len(prompt_dataloader), acc_draws / len(prompt_dataloader)

