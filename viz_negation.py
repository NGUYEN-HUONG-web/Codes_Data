import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model import TransformerClassifier


def main():
    """Tạo heatmap cho câu chứa phủ định và thêm phân tích ngắn vào báo cáo.

    Sử dụng câu mẫu "i do not like this movie", lưu `results/attention_negation.png`
    và nối kết quả vào `results/attention_report.txt`.
    """
    processed_dir = Path("data/processed")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab = json.load(open(processed_dir / "vocab.json", "r", encoding="utf-8"))
    meta = json.load(open(processed_dir / "meta.json", "r", encoding="utf-8"))

    model_path = results_dir / "model_Transformer_d64_ff128.pt"
    if not model_path.exists():
        candidates = sorted(results_dir.glob("model_Transformer*.pt"))
        if not candidates:
            raise FileNotFoundError("No trained model found in results/. Run train.py first.")
        model_path = candidates[0]

    # build model
    model = TransformerClassifier(meta["vocab_size"], 64, 128, meta["max_len"], meta["num_classes"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    sentence = "i do not like this movie"
    tokens = sentence.strip().lower().split()[: meta["max_len"]]
    ids = [vocab.get(t, vocab.get("[UNK]", 1)) for t in tokens]
    if len(ids) < meta["max_len"]:
        ids += [vocab.get("[PAD]", 0)] * (meta["max_len"] - len(ids))

    with torch.no_grad():
        logits = model(torch.tensor([ids], dtype=torch.long))
        pred = logits.argmax(dim=-1).item()
        weights = model.last_attention_weights[0].cpu().numpy()

    L = len(tokens)
    w = weights[:L, :L]

    out_path = results_dir / "attention_negation.png"
    plt.figure(figsize=(6, 5))
    plt.imshow(w, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(L), tokens, rotation=45, ha="right")
    plt.yticks(range(L), tokens)
    plt.title(f"Negation example | pred={meta['label_names'][pred]}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    # analysis
    avg_received = w.mean(axis=0)
    top_idx = int(avg_received.argmax())
    top_token = tokens[top_idx]

    lines = [
        f"Sentence (negation): {sentence}",
        f"Predicted: {meta['label_names'][pred]}",
        f"Top-attended token: '{top_token}' (idx={top_idx})",
        "Observation 1: Check whether attention links 'not' to the sentiment word to capture negation.",
        "Observation 2: If attention highlights the negation token and sentiment token together, model may handle negation better.",
        "",
    ]

    with open(results_dir / "attention_report.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Saved negation heatmap:", out_path)
    print("Appended analysis to:", results_dir / "attention_report.txt")


if __name__ == "__main__":
    main()
