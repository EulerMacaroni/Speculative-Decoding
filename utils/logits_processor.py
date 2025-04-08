import abc

import torch
from torch import Tensor
from torch.nn import functional as F


class LogitsProcessor(abc.ABC):
    """Logits processors for sampling."""

    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, logits: Tensor) -> Tensor:
        proc = self._process(logits)
        return F.softmax(proc / self.temperature, dim=-1)

    @abc.abstractmethod
    def _process(self, logits: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def sample(self, probs: Tensor) -> Tensor:
        pass


class GreedyProcessor(LogitsProcessor):
    """Greedy: Most probable token."""

    def __init__(self, temperature: float = 1):
        super().__init__(temperature)

    def _process(self, logits: Tensor) -> Tensor:
        return logits

    def sample(self, probs: Tensor) -> Tensor:
        return torch.argmax(probs, dim=-1).unsqueeze(-1)


class MultinomialProcessor(LogitsProcessor):
    """Multinomial: Random sampling."""

    def __init__(self, temperature: float):
        super().__init__(temperature)

    def _process(self, logits: Tensor) -> Tensor:
        return logits

    def sample(self, probs: Tensor) -> Tensor:
        return torch.multinomial(probs, num_samples=1)


class TopKProcessor(MultinomialProcessor):
    """Top-k: Top-k sampling."""

    def __init__(self, temperature: float, top_k: int):
        super().__init__(temperature)
        self.top_k = top_k

    def _process(self, logits: Tensor) -> Tensor:
        logits = logits.to(torch.float32)
        top_k = min(self.top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = -1e4
        return logits


class NucleusProcessor(MultinomialProcessor):
    """Nucleus: Top-p sampling."""

    def __init__(self, temperature: float, top_p: float):
        super().__init__(temperature)
        self.top_p = top_p

    def _process(self, logits: Tensor) -> Tensor:
        logits = logits.to(torch.float32)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        sorted_logits[sorted_indices_to_remove] = -1e4
        logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))
        return logits


class TopKNucleusProcessor(MultinomialProcessor):
    """Top-k and nucleus: Top-k sampling with top-p fallback."""

    def __init__(self, temperature: float, top_k: int, top_p: float):
        super().__init__(temperature)
        self.top_k = top_k
        self.top_p = top_p

    def _process(self, logits: Tensor) -> Tensor:
        logits = logits.to(torch.float32)
        top_k = min(self.top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = -1e4
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        sorted_logits[sorted_indices_to_remove] = -1e4
        logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))
        return logits


class TwoStageSamplingProcessor(MultinomialProcessor):
    """Top-k selection followed by noise injection for diversity."""

    def __init__(
        self, temperature: float = 1, top_k: int = 10, noise_scale: float = 0.3
    ):
        """
        top_k	10-50	Controls candidate pool size (quality vs diversity balance)
        noise_scale	0.1-0.5	Governs exploration magnitude (higher = more diversity)
        temperature	0.7-1.2	Final probability sharpening (inherited from base processor)

        """
        super().__init__(temperature)
        self.top_k = top_k
        self.noise_scale = noise_scale

    def _process(self, logits: Tensor) -> Tensor:
        logits = logits.to(torch.float32)
        # Stage 1: Select top-k
        top_k = min(self.top_k, logits.size(-1))
        top_logits, top_indices = torch.topk(logits, top_k, dim=-1)

        # Stage 2: Add controlled noise
        noise = torch.randn_like(top_logits) * self.noise_scale
        perturbed_logits = top_logits + noise

        # Reconstruct full logits tensor
        perturbed_logits_full = torch.full_like(logits, -1e4)
        return perturbed_logits_full.scatter(-1, top_indices, perturbed_logits)
