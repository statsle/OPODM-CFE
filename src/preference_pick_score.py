"""
Class containing the preference model, i.e. PickScore.
"""

from transformers import AutoProcessor, AutoModel
import torch

from PIL import Image
from io import BytesIO


def bytes2image(bytes: bytes) -> Image:
    return Image.open(BytesIO(bytes)).convert("RGB")


class PreferenceFromPickScore:
    def __init__(self, device=None):
        self.device = device or "cuda"

        self.processor_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.model_pretrained_name = "yuvalkirstain/PickScore_v1"

        self.processor = AutoProcessor.from_pretrained(self.processor_name, local_files_only=True)
        self.model = (
            AutoModel.from_pretrained(self.model_pretrained_name, local_files_only=True).eval().to(self.device)
        )

    @torch.no_grad()
    def calc_probs(self, prompts, images):
        # preprocess
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        batch_size = len(prompts)

        # embed
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = self.model.logit_scale.exp() * (text_embs.unsqueeze(1) @ image_embs.view(2, batch_size, -1).permute(1, 2, 0)).squeeze(1)

        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)[:, 0]

        return probs

    def __call__(self, prompts, images_0, images_1):
        return self.calc_probs(prompts, images_0 + images_1)
