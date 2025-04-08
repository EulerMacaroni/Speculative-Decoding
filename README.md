The Speculative-Decoding repository provides a PyTorch implementation of the speculative decoding technique, as introduced in the paper "Fast Inference from Transformers via Speculative Decoding" by Leviathan et al., 2023. This method aims to enhance the efficiency of transformer-based model inference by generating multiple token predictions simultaneously and then verifying them, rather than generating tokens sequentially.

**Key Features:**

- **Decoding Strategies:** The repository implements three primary text generation strategies:
  1. **Auto-regressive Decoding:** Traditional token-by-token generation.
  2. **Speculative Decoding:** Generates multiple tokens in parallel and verifies them, aiming to speed up the inference process.
  3. **N-Gram-Assisted Speculative Decoding:** Incorporates a fast drafting phase using an adaptive N-gram module, followed by a verification step by the target model. Based on the ANPD framework (Adaptive N-gram Parallel Decoding), this approach dynamically adapts the N-gram module to context and introduces a multi-level architecture for improved draft precision. Based on the paper, it achieves acceleration without compromising output quality, retraining, or extra GPU memory. In experiments on models like LLaMA, it has shown speedups up to 3.67x.

- **Sampling Techniques:** Both auto-regressive and speculative decoding support:
  - **Greedy Sampling:** Selects the most probable next token. Greedy decoding prioritizes determinism and speed, but may miss out on diversity or more creative continuations.

  - **Multinomial Sampling:** Introduces controlled randomness by sampling from the probability distribution over tokens. It respects the full softmax distribution, allowing for diverse but valid outputs, especially with temperature tuning.

  - **Top-k Sampling:** Narrows down the sampling pool to the top-k most likely tokens. This improves quality by removing low-probability noise while retaining some diversity.

  - **Top-k with Noise (Two-Stage Sampling):** Selects top-k logits and then adds Gaussian noise to introduce further diversity in sampling. This method is useful when you want diversity but still ensure quality tokens are considered first.

**Repository Structure:**

```
Speculative-Decoding/
├── data/                   # Datasets for training/evaluation
├── ngram_assisted/         # N-gram-assisted decoding logic
├── results/                # Generated outputs and results
├── sampling/               # Sampling strategy implementations
├── utils/                  # Helper and Sampling technique
├── infer.py                # Main script for running inference
├── requirements.txt        # Python dependency list
├── test.py                 # Unit tests for decoding methods
└── test_optimized_for_colab.ipynb  # Notebook for Colab
```

*References:**

- Leviathan, Y., et al. (2023). Fast Inference from Transformers via Speculative Decoding. [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)
- Vyas, Y., et al. (2024). Efficient and Reliable Speculative Decoding with N-Gram Expert Guidance. [arXiv:2404.08698](https://arxiv.org/abs/2404.08698)
