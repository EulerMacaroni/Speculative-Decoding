import random
import re
import string
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

from ngram_assisted.ngram_assisted import ngram_assisted_speculative_generate
from ngram_assisted.ngram_storage import NGramStorage
from sampling.base_decoding import autoregressive_generate
from sampling.speculative_decoding import speculative_generate
from utils.logits_processor import *


def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Calculate F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, ground_truth):
    """Check if prediction matches ground truth exactly."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# ------------------ Setup ------------------
# Seed for reproducibility
def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
target_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
draft_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
target_quantize = QuantoConfig(weights="int8")
drafter_quantize = QuantoConfig(weights="int8")
tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
drafter = AutoModelForCausalLM.from_pretrained(
    draft_model,
    quantization_config=drafter_quantize,
    device_map=device,
    trust_remote_code=True,
)
target = AutoModelForCausalLM.from_pretrained(
    target_model,
    quantization_config=target_quantize,
    device_map=device,
    trust_remote_code=True,
)
drafter.eval()
target.eval()
# End tokens
end_tokens = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

# Load dataset
df = pd.read_csv("QAwithTemp.csv")
df = df.head(2)
# ------------------ Evaluation Function ------------------


def run_model(model_name, prompt, answer_text, temperature, input_ids, max_len):
    # processor = TwoStageSamplingProcessor(
    #     temperature=temperature,
    #     top_k=10,
    #     noise_scale=0.6
    # )
    processor = TwoStageSamplingProcessor()

    start = time.time()
    print(model_name)
    print(prompt)

    if model_name == "speculative":
        output_ids, acc_rate = speculative_generate(
            inputs=input_ids,
            drafter=drafter,
            target=target,
            tokenizer=tokenizer,
            logits_processor=processor,
            gamma=4,
            max_gen_len=max_len,
            eos_tokens_id=end_tokens,
            use_cache=False,
            debug=False,
        )
    elif model_name == "ngram":
        ngram = NGramStorage(n=3, vocab_size=target.config.vocab_size)
        ngram.reset()
        output_ids, acc_rate = ngram_assisted_speculative_generate(
            inputs=input_ids,
            ngramstorage=ngram,
            target=target,
            tokenizer=tokenizer,
            gamma=4,
            filler_top_k=3,
            logits_processor=processor,
            max_gen_len=max_len,
            eos_tokens_id=end_tokens,
            use_cache=False,
            debug=False,
            stop_if_unknown=True,
        )
    elif model_name == "autoregressive":
        output_ids = autoregressive_generate(
            inputs=input_ids,
            model=target,
            logits_processor=processor,
            max_gen_len=max_len,
            eos_tokens_id=end_tokens,
            use_cache=False,
            debug=False,
        )
        acc_rate = None  # Not applicable to AR

    duration = time.time() - start
    tps = len(output_ids) / duration if duration > 0 else float("inf")
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(output_text)
    f1 = f1_score(output_text, answer_text)
    em = exact_match(output_text, answer_text)
    score = 0.6 * f1 + 0.4 * em

    return {
        "model_answer": output_text,
        "score": score,
        "acceptance_rate": acc_rate,
        "tokens_per_sec": round(tps, 2),
    }


# ------------------ Apply to Data ------------------


def evaluate_row(row):
    prompt = row["question"]

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    answer = row["answer"]
    temp = 1
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    max_len = 40

    results = {
        "speculative": run_model(
            "speculative", prompt, answer, temp, input_ids, max_len
        ),
        "ngram": run_model("ngram", prompt, answer, temp, input_ids, max_len),
        "autoregressive": run_model(
            "autoregressive", prompt, answer, temp, input_ids, max_len
        ),
    }

    return pd.Series(
        {
            "spec_output": results["speculative"]["model_answer"],
            "spec_score": results["speculative"]["score"],
            "spec_accept": results["speculative"]["acceptance_rate"],
            "spec_tps": results["speculative"]["tokens_per_sec"],
            "ngram_output": results["ngram"]["model_answer"],
            "ngram_score": results["ngram"]["score"],
            "ngram_accept": results["ngram"]["acceptance_rate"],
            "ngram_tps": results["ngram"]["tokens_per_sec"],
            "ar_output": results["autoregressive"]["model_answer"],
            "ar_score": results["autoregressive"]["score"],
            "ar_tps": results["autoregressive"]["tokens_per_sec"],
        }
    )


# Run evaluation on all rows
df_results = df.apply(evaluate_row, axis=1)

# Combine with original data
df_final = pd.concat([df, df_results], axis=1)

print(df_final)
# Uncomment to save results  for {sampling_method}
# df_final.to_csv("QAwithTemp_{process}.csv", index=False)
