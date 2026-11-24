"""Choose an inference backend without changing the agent job lifecycle."""

from agent.config import AgentConfig
from agent.ollama_client import run_inference as run_ollama
from agent.parallax_client import run_inference as run_parallax
from common.inference import InferenceError, InferenceResult


def run_inference(
    config: AgentConfig, model_name: str, prompt: str, max_tokens: int, timeout_s: float,
) -> InferenceResult:
    if config.inference_backend == "ollama":
        return run_ollama(config.ollama_url, model_name, prompt, max_tokens, timeout_s)
    if config.inference_backend == "parallax":
        return run_parallax(config.parallax_url, model_name, prompt, max_tokens, timeout_s)
    raise InferenceError(f"Unsupported inference backend: {config.inference_backend}")
