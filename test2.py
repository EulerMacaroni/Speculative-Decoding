import random
import re
import string
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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


def embedding_similarity(pred, truth, model):
    emb1 = model.encode([pred], convert_to_tensor=True)
    emb2 = model.encode([truth], convert_to_tensor=True)

    # Move to CPU for sklearn
    if emb1.device.type != "cpu":
        emb1 = emb1.cpu()
        emb2 = emb2.cpu()

    return float(cosine_similarity(emb1.numpy(), emb2.numpy())[0][0])


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

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

end_tokens = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
df = pd.read_csv("data/QAwithTemp.csv")
# df = df.head(1)

processors = {
    "greedy": lambda temp: GreedyProcessor(temperature=temp),
    "multinomial": lambda temp: MultinomialProcessor(temperature=temp),
    "topk": lambda temp: TopKProcessor(temperature=temp, top_k=10),
    "nucleus": lambda temp: NucleusProcessor(temperature=temp, top_p=0.95),
    "topknucleus": lambda temp: TopKNucleusProcessor(
        temperature=temp, top_k=10, top_p=0.95
    ),
    "TwoStage": lambda temp: TwoStageSamplingProcessor(
        temperature=temp, top_k=10, noise_scale=0.4
    ),
}

model_types = ["speculative", "ngram", "autoregressive"]

all_rows = []

for proc_name, proc_builder in processors.items():
    for idx, row in df.iterrows():
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        answer = row["answer"]
        temp = 0.8
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        max_len = 40

        for model_name in model_types:
            processor = (
                proc_builder(temp) if proc_name != "greedy" else GreedyProcessor()
            )

            try:
                start = time.time()
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
                else:
                    output_ids = autoregressive_generate(
                        inputs=input_ids,
                        model=target,
                        logits_processor=processor,
                        max_gen_len=max_len,
                        eos_tokens_id=end_tokens,
                        use_cache=False,
                        debug=False,
                    )
                    acc_rate = None

                duration = time.time() - start
                tps = len(output_ids) / duration if duration > 0 else float("inf")
                output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                # f1 = f1_score(output_text, answer)
                # em = exact_match(output_text, answer)
                sim = embedding_similarity(output_text, answer, embedding_model)
                # score = 0.6 * f1 + 0.4 * em
                score = sim

                all_rows.append(
                    {
                        "question": row["question"],
                        "answer": answer,
                        "temperature": temp,
                        "model": model_name,
                        "processor": proc_name,
                        "output": output_text,
                        "score": score,
                        "acceptance_rate": acc_rate,
                        "tokens_per_sec": round(tps, 2),
                    }
                )

                print(
                    f"Model: {model_name}, Processor: {proc_name}, Score: {score}, tokens/s: {tps}, Acceptance Rate: {acc_rate}"
                )
            except Exception as e:
                print(
                    f" Error in {model_name} with processor {proc_name} on question: '{row['question']}'\n{e}"
                )
                continue

# Save to CSV
final_df = pd.DataFrame(all_rows)
print(final_df)
# final_df.to_csv("results/all_model_results.csv", index=False)
# print("Evaluation complete. Results saved to results/all_model_results.csv")
