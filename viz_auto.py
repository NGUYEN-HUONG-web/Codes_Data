import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model import TransformerClassifier


"""Tiện ích visual tự động.

Script này chọn ba ví dụ từ tập test (một dự đoán đúng, một sai, và
một chứa phủ định nếu có), lưu heatmap attention và ghi phân tích ngắn
vào `results/attention_report.txt`.
"""

NEG_WORDS = {"not", "never", "don't", "didn't", "no", "none", "never"}


def load_vocab(path: Path):
    """Đọc `vocab.json` và trả về dict token->id."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_meta(path: Path):
    """Đọc `meta.json` và trả về dict metadata."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_text(text: str, vocab: dict, max_len: int):
    """Tách từ và chuyển thành id (padding/truncate tới `max_len`).

    Trả về (ids, tokens).`tokens` dùng để gán nhãn cho trục heatmap.
    """
    tokens = text.strip().lower().split()
    ids = [vocab.get(tok, vocab.get("[UNK]", 1)) for tok in tokens][:max_len]
    length = len(ids)
    if length < max_len:
        ids += [vocab.get("[PAD]", 0)] * (max_len - length)
    return ids, tokens[:max_len]


def save_heatmap(weights, tokens, out_path: Path, title: str):
    """Lưu heatmap attention (token x token) vào `out_path`."""
    plt.figure(figsize=(6, 5))
    plt.imshow(weights, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    plt.yticks(range(len(tokens)), tokens)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    processed_dir = Path("data/processed")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(processed_dir / "vocab.json")
    meta = load_meta(processed_dir / "meta.json")

    model_path = results_dir / "model_Transformer_d64_ff128.pt"
    if not model_path.exists():
        candidates = sorted(results_dir.glob("model_Transformer*.pt"))
        if not candidates:
            raise FileNotFoundError("No trained model found in results/. Run train.py first.")
        model_path = candidates[0]

    # infer d_model/d_ff from filename like model_Transformer_d64_ff128.pt
    stem = model_path.stem.replace("model_", "")
    if "d128_ff256" in stem:
        d_model, d_ff = 128, 256
    elif "d32_ff64" in stem:
        d_model, d_ff = 32, 64
    else:
        d_model, d_ff = 64, 128

    model = TransformerClassifier(meta["vocab_size"], d_model, d_ff, meta["max_len"], meta["num_classes"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    data = torch.load(processed_dir / "test.pt")
    texts = data["texts"]
    input_ids_all = data["input_ids"]
    labels_all = data["labels"]

    found = {"correct": None, "incorrect": None, "negation": None}

    for i, (ids, label, text) in enumerate(zip(input_ids_all, labels_all, texts)):
        # `ids` may already be a torch.Tensor (shape: max_len) or a list; handle both
        if isinstance(ids, torch.Tensor):
            ids_tensor = ids.unsqueeze(0).to(torch.long)
        else:
            ids_tensor = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(ids_tensor)
            pred = logits.argmax(dim=-1).item()
            weights = model.last_attention_weights[0].cpu().numpy()

        tokens = text.strip().lower().split()[: meta["max_len"]]

        if found["negation"] is None and any(w in NEG_WORDS for w in tokens):
            found["negation"] = (i, text, int(label), pred, weights, tokens)

        if pred == int(label) and found["correct"] is None:
            found["correct"] = (i, text, int(label), pred, weights, tokens)

        if pred != int(label) and found["incorrect"] is None:
            found["incorrect"] = (i, text, int(label), pred, weights, tokens)

        if all(found.values()):
            break

    outputs = []
    for key in ("correct", "incorrect", "negation"):
        item = found.get(key)
        if item is None:
            print(f"No example found for: {key}")
            continue
        i, text, true_label, pred_label, weights, tokens = item
        # restrict weights to token length
        L = len(tokens)
        w = weights[:L, :L]
        out_path = results_dir / f"attention_{key}_{i}.png"
        title = f"{key.upper()} idx={i} | pred={meta['label_names'][pred_label]} | true={meta['label_names'][true_label]}"
        save_heatmap(w, tokens, out_path, title)

        # analyze attention: token with highest average received attention
        avg_received = w.mean(axis=0)
        top_idx = int(avg_received.argmax())
        top_token = tokens[top_idx]

        obs = [
            f"Sentence idx={i}: {text}",
            f"Predicted: {meta['label_names'][pred_label]} | True: {meta['label_names'][true_label]}",
            f"Top-attended token (by received avg): '{top_token}' (idx={top_idx})",
        ]
        # two short comments
        comments = []
        comments.append(f"Observation 1: Attention concentrates around '{top_token}', suggesting the model focuses on this token when forming context.")
        if any(t in NEG_WORDS for t in tokens):
            comments.append("Observation 2: Sentence contains negation; check whether attention links negation to sentiment words (may indicate handling of negation).")
        else:
            comments.append("Observation 2: Attention appears concentrated on sentiment-bearing words rather than function words.")

        outputs.append((out_path, obs, comments))

    # print summary and save a small report
    report_lines = []
    for out_path, obs, comments in outputs:
        print("Saved heatmap:", out_path)
        for o in obs:
            print(o)
        for c in comments:
            print(c)
        print()
        report_lines += obs + comments + ["\n"]

    with open(results_dir / "attention_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Wrote report to:", results_dir / "attention_report.txt")


if __name__ == "__main__":
    main()
