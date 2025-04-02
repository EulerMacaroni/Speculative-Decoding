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
        top_k = min(self.top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = -1e20
        return logits


class NucleusProcessor(MultinomialProcessor):
    """Nucleus: Top-p sampling."""

    def __init__(self, temperature: float, top_p: float):
        super().__init__(temperature)
        self.top_p = top_p

    def _process(self, logits: Tensor) -> Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        sorted_logits[sorted_indices_to_remove] = -1e20
        logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))
        return logits


class TopKNucleusProcessor(MultinomialProcessor):
    """Top-k and nucleus: Top-k sampling with top-p fallback."""

    def __init__(self, temperature: float, top_k: int, top_p: float):
        super().__init__(temperature)
        self.top_k = top_k
        self.top_p = top_p

    def _process(self, logits: Tensor) -> Tensor:
        top_k = min(self.top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = -1e20
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        sorted_logits[sorted_indices_to_remove] = -1e20
        logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))
        return logits


class MCMCProcessor(LogitsProcessor):
    """MCMC: Markov Chain Monte Carlo token filtering."""

    def __init__(self, temperature: float = 1.0, num_steps: int = 100):
        super().__init__(temperature)
        self.num_steps = num_steps

    def _process(self, logits: Tensor) -> Tensor:
        """
        All others are suppressed with large negative values (like -1e20).
        """
        logits = logits.view(
            -1, logits.size(-1)
        )  # Support inputs like [1, gamma, vocab]
        batch_size, vocab_size = logits.shape
        device = logits.device

        probs = F.softmax(logits / self.temperature, dim=-1)

        current_tokens = torch.randint(0, vocab_size, (batch_size,), device=device)

        for _ in range(self.num_steps):
            noise = torch.normal(
                mean=0.0, std=vocab_size / 10.0, size=(batch_size,), device=device
            ).long()
            proposed_tokens = torch.clamp(current_tokens + noise, 0, vocab_size - 1)

            p_current = probs[torch.arange(batch_size), current_tokens]
            p_proposed = probs[torch.arange(batch_size), proposed_tokens]

            accept_ratio = (p_proposed / (p_current + 1e-9)).clamp(max=1.0)
            accept = torch.rand(batch_size, device=device) < accept_ratio

            current_tokens = torch.where(accept, proposed_tokens, current_tokens)

        new_logits = torch.full_like(logits, fill_value=-1e20)
        new_logits[torch.arange(batch_size), current_tokens] = logits[
            torch.arange(batch_size), current_tokens
        ]

        return new_logits.view(*logits.shape)  # Return with the same shape

    def sample(self, probs: Tensor) -> int:
        """Standard multinomial sampling from filtered probabilities.

        Supports both 1D [vocab_size] and 2D [1, vocab_size] inputs.
        """
        if probs.dim() == 1:
            # If probs is [vocab_size]
            return torch.multinomial(probs, num_samples=1).item()
        elif probs.dim() == 2 and probs.size(0) == 1:
            # If probs is [1, vocab_size]
            return torch.multinomial(probs, num_samples=1).squeeze(0).item()
        else:
            raise ValueError(f"Expected input shape [vocab_size] or [1, vocab_size], got {probs.shape}")

