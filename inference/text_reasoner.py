import tempfile
from pathlib import Path

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"


class TextReasoner:
    """Free local vision-language reasoning accelerated by Apple MLX."""

    def __init__(self, device, model_name=DEFAULT_MODEL):
        self.device = "mlx-metal"
        self.model_name = model_name
        print(f"[model] Loading {model_name} with MLX")
        self.model, self.processor = load(model_name)
        self.config = load_config(model_name)

    def generate(self, query, context, image):
        prompt = self._build_prompt(query, context)
        formatted = apply_chat_template(self.processor, self.config, prompt, num_images=1)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                image.convert("RGB").save(temp_file, format="JPEG", quality=92)
            output = generate(
                self.model,
                self.processor,
                formatted,
                [str(temp_path)],
                max_tokens=220,
                temperature=0.0,
                verbose=False,
            )
        finally:
            if temp_path:
                temp_path.unlink(missing_ok=True)
        return getattr(output, "text", output).strip()

    @staticmethod
    def _build_prompt(query, context=None):
        normalized = query.strip().lower().rstrip("?.!")
        if normalized in {
            "what is in this image",
            "what do you see",
            "describe this image",
            "describe the image",
        }:
            prompt = (
                "Describe this image accurately in 2 to 4 complete sentences. Identify the main subjects, "
                "setting, actions, visible text, spatial relationships, and notable details. Do not guess."
            )
        else:
            prompt = (
                f"Answer this question about the image clearly and completely: {query.strip()} "
                "Use only visible evidence. If the answer cannot be determined, say so plainly."
            )
        if context:
            prompt += f" User context: {context.strip()}"
        return prompt
