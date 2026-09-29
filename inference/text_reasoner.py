import os

from utils.rag_utils import RetrievalEngine


DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_CAPTION_MODEL = "Salesforce/blip2-opt-2.7b"
DEFAULT_FALLBACK_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"


class TextReasoner:
    """BLIP-2 visual extraction, FAISS retrieval, and LLaVA reasoning."""

    def __init__(self, device, model_name=DEFAULT_MODEL):
        import torch
        from transformers import AutoProcessor, Blip2ForConditionalGeneration, LlavaForConditionalGeneration

        self.device = device
        self.model_name = model_name
        self.caption_model_name = os.getenv("CAPTION_MODEL", DEFAULT_CAPTION_MODEL)
        self.torch = torch
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.lightweight = model_name == DEFAULT_FALLBACK_MODEL

        if self.lightweight:
            try:
                from transformers import AutoModelForImageTextToText as AutoVisionModel
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoVisionModel

            print(f"[model] Loading lightweight {model_name} on {device}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoVisionModel.from_pretrained(
                model_name, torch_dtype=self.dtype, low_cpu_mem_usage=True
            ).to(device)
            self.model.eval()
            return

        print(f"[model] Loading {self.caption_model_name} on {device}")
        self.caption_processor = AutoProcessor.from_pretrained(self.caption_model_name)
        self.caption_model = Blip2ForConditionalGeneration.from_pretrained(
            self.caption_model_name, torch_dtype=self.dtype
        ).to(device)
        self.caption_model.eval()

        print(f"[model] Loading {model_name} on {device}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=self.dtype, low_cpu_mem_usage=True
        ).to(device)
        self.model.eval()
        self.retrieval = RetrievalEngine()

    def generate(self, query, context, image):
        if self.lightweight:
            return self._generate_lightweight(query, context, image)

        caption = self._caption(image)
        sources = [caption]
        if context and context.strip():
            sources.append(context.strip())
        evidence = self.retrieval.retrieve(query, sources)
        prompt = self._build_prompt(query, caption, evidence)

        inputs = self._move_inputs(self.processor(images=image, text=prompt, return_tensors="pt"))
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=220, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1] :]
        answer = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return answer or caption

    def _generate_lightweight(self, query, context, image):
        instruction = (
            "Answer the question using only visible evidence in the image. "
            "Be concise and say when something cannot be determined.\n"
        )
        if context and context.strip():
            instruction += f"Additional context: {context.strip()}\n"
        instruction += f"Question: {query.strip()}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._move_inputs(
            self.processor(text=prompt, images=[image], return_tensors="pt")
        )
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=160, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def _caption(self, image):
        inputs = self._move_inputs(self.caption_processor(images=image, return_tensors="pt"))
        with self.torch.inference_mode():
            output = self.caption_model.generate(**inputs, max_new_tokens=80)
        return self.caption_processor.batch_decode(output, skip_special_tokens=True)[0].strip()

    def _move_inputs(self, inputs):
        return {
            key: value.to(self.device, dtype=self.dtype) if value.is_floating_point() else value.to(self.device)
            for key, value in inputs.items()
        }

    @staticmethod
    def _build_prompt(query, caption="", retrieved=""):
        return (
            "USER: <image>\n"
            "Answer using the image and the retrieved evidence. "
            "Be concise, specific, and do not invent details.\n"
            f"Visual extraction: {caption or 'No caption available.'}\n"
            f"Retrieved evidence: {retrieved or 'No additional evidence.'}\n"
            f"Question: {query.strip()}\n"
            "ASSISTANT:"
        )
