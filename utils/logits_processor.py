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


class TypicalProcessor(LogitsProcessor):
    """Typical Sampling: Select tokens based on how close their information content is to the expected information."""

    def __init__(self, temperature: float = 1.0, mass: float = 0.9):
        super().__init__(temperature)
        self.mass = mass

    def _process(self, logits: Tensor) -> Tensor:
        """
        Process logits using Typical Sampling.
        All tokens with atypical information content are suppressed.
        """
        logits = logits.view(
            -1, logits.size(-1)
        )  # Support inputs like [1, gamma, vocab]
        batch_size, vocab_size = logits.shape
        device = logits.device

        # Create a new tensor to store the processed logits
        new_logits = logits.clone()

        # Process each item in the batch
        for b in range(batch_size):
            # Calculate token probabilities
            probs = F.softmax(logits[b] / self.temperature, dim=-1)

            # Calculate entropy
            log_probs = torch.log(probs + 1e-10)
            expected_entropy = -torch.sum(probs * log_probs, dim=-1)

            # Calculate each token's contribution to entropy
            token_entropies = -log_probs

            # Calculate how far each token is from the expected entropy
            token_divergence = torch.abs(token_entropies - expected_entropy)

            # Sort by divergence
            sorted_divergence, sorted_indices = torch.sort(token_divergence, dim=-1)
            sorted_probs = probs.gather(-1, sorted_indices)

            # Keep tokens until we reach the desired probability mass
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            indices_to_keep = cumulative_probs <= self.mass

            # If nothing is kept, keep at least one token
            if not torch.any(indices_to_keep):
                indices_to_keep[0] = True

            # Create a mask for the tokens to keep
            masked_divergence = torch.full_like(token_divergence, float("inf"))

            # Gather indices to keep and map back to original positions
            keep_indices = sorted_indices[indices_to_keep]
            masked_divergence.scatter_(-1, keep_indices, 0)

            # Apply the mask to the logits
            new_logits[b][masked_divergence == float("inf")] = -1e20

        return new_logits.view(*logits.shape)  # Return with the same shape

    def sample(self, probs: Tensor) -> Tensor:
        """Standard multinomial sampling from filtered probabilities."""
        return torch.multinomial(probs, num_samples=1)


class TwoStageSamplingProcessor(MultinomialProcessor):
    """Top-k selection followed by noise injection for diversity."""

    def __init__(
        self, temperature: float = 0.8, top_k: int = 10, noise_scale: float = 0.4
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
        # Stage 1: Select top-k
        top_k = min(self.top_k, logits.size(-1))
        top_logits, top_indices = torch.topk(logits, top_k, dim=-1)

        # Stage 2: Add controlled noise
        noise = torch.randn_like(top_logits) * self.noise_scale
        perturbed_logits = top_logits + noise

        # Reconstruct full logits tensor
        perturbed_logits_full = torch.full_like(logits, -1e20)
        return perturbed_logits_full.scatter(-1, top_indices, perturbed_logits)
