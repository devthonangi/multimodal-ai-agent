import base64
import io
import os

import requests


DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
DEFAULT_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


class TextReasoner:
    """High-quality multimodal reasoning through NVIDIA NIM."""

    def __init__(self, device, model_name=DEFAULT_MODEL):
        self.device = "nvidia-nim"
        self.model_name = model_name
        self.api_url = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL)
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not configured. Copy .env.example to .env and add a key from build.nvidia.com."
            )

    def generate(self, query, context, image):
        prompt = self._build_prompt(query, context)
        image_data = self._encode_image(image)
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.2,
                "top_k": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=120,
        )
        if not response.ok:
            try:
                detail = response.json().get("detail") or response.json().get("message")
            except ValueError:
                detail = response.text[:300]
            raise RuntimeError(f"NVIDIA API returned {response.status_code}: {detail or 'request failed'}")
        payload = response.json()
        try:
            answer = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("NVIDIA API returned an unexpected response") from error
        return answer.strip()

    @staticmethod
    def _encode_image(image):
        output = io.BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=92, optimize=True)
        return base64.b64encode(output.getvalue()).decode("ascii")

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
                "setting, actions, visible text, spatial relationships, and notable details. Do not guess "
                "about anything that is not visually supported."
            )
        else:
            prompt = (
                f"Answer this question about the image: {query.strip()}\n"
                "Give a clear, complete answer grounded only in visible evidence. Explain briefly when useful. "
                "If the answer cannot be determined from the image, say so plainly."
            )
        if context:
            prompt += f"\nUser-provided context: {context.strip()}"
        return prompt
