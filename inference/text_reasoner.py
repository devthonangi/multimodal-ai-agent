import torch
from transformers import BlipForQuestionAnswering, BlipProcessor


DEFAULT_MODEL = "Salesforce/blip-vqa-base"


class TextReasoner:
    """A practical image-question-answering model for local inference."""

    def __init__(self, device, model_name=DEFAULT_MODEL):
        self.device = device
        self.model_name = model_name
        print(f"[model] Loading {model_name} on {device}")
        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
        self.model.eval()

    def generate(self, query, context, image):
        question = query.strip()
        if context:
            question = f"{question} Context: {context}"
        inputs = self.processor(images=image, text=question, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=40)
        return self.processor.decode(generated[0], skip_special_tokens=True).strip()
