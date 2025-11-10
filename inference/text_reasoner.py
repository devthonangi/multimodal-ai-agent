import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class TextReasoner:
    def __init__(self, device):
        self.device = device
        print("[INFO] Loading BLIP-2 for multimodal reasoning...")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16
        ).to(device)

    def generate(self, query, context, vision_features):
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        inputs = self.processor(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=128)
        return self.processor.decode(generated[0], skip_special_tokens=True)
