import os
import tempfile
from pathlib import Path

from utils.rag_utils import RetrievalEngine


DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_CAPTION_MODEL = "Salesforce/blip2-opt-2.7b"
DEFAULT_FALLBACK_MODEL = "Salesforce/blip-vqa-base"
DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"


class TextReasoner:
    """BLIP-2 visual extraction, FAISS retrieval, and LLaVA reasoning."""

    def __init__(self, device, model_name=DEFAULT_MODEL):
        self.device = device
        self.model_name = model_name
        self.mlx = device == "mlx"
        if self.mlx:
            from mlx_vlm import load

            print(f"[model] Loading {model_name} with MLX")
            self.model, self.processor = load(model_name)
            return

        import torch
        from transformers import AutoProcessor, Blip2ForConditionalGeneration, LlavaForConditionalGeneration

        self.caption_model_name = os.getenv("CAPTION_MODEL", DEFAULT_CAPTION_MODEL)
        self.torch = torch
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.lightweight = model_name == DEFAULT_FALLBACK_MODEL

        if self.lightweight:
            from transformers import BlipForQuestionAnswering, BlipProcessor

            print(f"[model] Loading lightweight {model_name} on {device}")
            try:
                self.processor = BlipProcessor.from_pretrained(
                    model_name, local_files_only=True
                )
                self.model = BlipForQuestionAnswering.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                ).to(device)
            except OSError:
                self.processor = BlipProcessor.from_pretrained(model_name)
                self.model = BlipForQuestionAnswering.from_pretrained(
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
        if self.mlx:
            return self._generate_mlx(query, context, image)
        if self.lightweight:
            return self._generate_lightweight(query, context, image)

        caption = self._caption(image)
        sources = [caption]
        if context and context.strip():
            sources.append(context.strip())
        evidence = self.retrieval.retrieve(query, sources)
        prompt = self._build_prompt(query, caption, evidence, context or "")

        inputs = self._move_inputs(self.processor(images=image, text=prompt, return_tensors="pt"))
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=220, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1] :]
        answer = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return answer or caption

    def _generate_mlx(self, query, context, image):
        from mlx_vlm import apply_chat_template, generate

        history = context.strip() if context and context.strip() else "No previous turns."
        prompt = (
            "You are a conversational visual assistant. Answer only from visible image evidence. "
            "Use the conversation history to understand follow-ups, but independently verify every "
            "claim against the image. Never invent objects, text, actions, emotions, or thoughts. "
            "If something is unclear or not visible, say so. If asked for a description, give a "
            "complete description covering the subject, appearance, pose, clothing, background, "
            "visible objects, and readable text. If a question is unrelated to the image, politely "
            "redirect the user. Respond naturally in complete sentences.\n\n"
            f"Conversation history:\n{history}\n\nCurrent question: {query.strip()}"
        )
        formatted = apply_chat_template(
            self.processor, self.model.config, prompt, num_images=1
        )
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                image.convert("RGB").save(temp_file, format="PNG")
            output = generate(
                self.model,
                self.processor,
                formatted,
                image=[str(temp_path)],
                max_tokens=260,
                temperature=0.0,
                verbose=False,
            )
        finally:
            if temp_path:
                temp_path.unlink(missing_ok=True)
        return getattr(output, "text", output).strip()

    def _generate_lightweight(self, query, context, image):
        normalized = query.lower().strip(" .,!?;:")
        previous_answer = self._last_history_value(context, "Assistant")

        if len(normalized) <= 3 and normalized not in {"who", "why"}:
            return "I didn’t understand that. Please ask a clear question about the image."
        if normalized in {"u asking me", "you asking me", "who are you", "how are you"}:
            return "I can only help with questions about the selected image."
        if any(term in normalized for term in ("thinking", "thought", "mind", "feeling inside")):
            return (
                "I can’t determine what someone is thinking from an image. "
                "I can only describe their visible expression, pose, and surroundings."
            )
        if previous_answer and normalized in {"what", "what do you mean", "explain", "explain that"}:
            return f"To clarify, {previous_answer[0].lower() + previous_answer[1:]}"
        if previous_answer and ("are you sure" in normalized or normalized == "why"):
            return (
                f"Based on the visible details, my previous answer was: {previous_answer} "
                "I can’t be certain about anything that is not directly shown."
            )

        inputs = self._move_inputs(self.processor(images=image, text=query, return_tensors="pt"))
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
        raw_answer = self.processor.decode(output[0], skip_special_tokens=True).strip()
        return self._conversational_answer(query, raw_answer)

    @staticmethod
    def _last_history_value(context, role):
        if not context:
            return ""
        prefix = f"{role}: "
        values = [line[len(prefix) :].strip() for line in context.splitlines() if line.startswith(prefix)]
        return values[-1] if values else ""

    @staticmethod
    def _conversational_answer(query, answer):
        answer = answer.strip().rstrip(".")
        if not answer:
            return "I can’t determine that from this image."
        normalized = query.lower()
        if "doing" in normalized:
            return f"The person appears to be {answer}."
        if "wearing" in normalized:
            return f"The person is wearing {answer}."
        if any(phrase in normalized for phrase in ("what is in", "what's in", "whats in", "what do you see")):
            if answer.lower() in {"man", "woman", "boy", "girl", "person", "dog", "cat"}:
                answer = f"a {answer}"
            return f"The image shows {answer}."
        return answer[0].upper() + answer[1:] + "."

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
    def _build_prompt(query, caption="", retrieved="", history=""):
        return (
            "USER: <image>\n"
            "Answer using the image and the retrieved evidence. "
            "This is an image-only conversation. If the request is unrelated to the selected image "
            "or its conversation history, politely redirect the user to ask about the image. "
            "Sound natural and conversational. Answer directly in one or two complete sentences, "
            "and do not respond with an unrelated follow-up question. Be specific and do not "
            "invent details.\n"
            f"Visual extraction: {caption or 'No caption available.'}\n"
            f"Retrieved evidence: {retrieved or 'No additional evidence.'}\n"
            f"Conversation history: {history or 'No previous turns.'}\n"
            "Use the conversation history to resolve follow-up questions and remain consistent.\n"
            f"Question: {query.strip()}\n"
            "ASSISTANT:"
        )
