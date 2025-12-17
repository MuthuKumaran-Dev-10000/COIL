#!/usr/bin/env python3
"""
Enhanced token & cost analysis for multiple tokenizer models
"""

import math
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd
import os

# -----------------------------
# TOKENIZER SUPPORT
# -----------------------------
try:
    import tiktoken
    HAVE_TIKTOKEN = True
except Exception:
    HAVE_TIKTOKEN = False

TOKENIZER_MODELS = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "o1": "o1",
    "o1-mini": "o1-mini",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "claude-3.7": "cl100k_base",
    "claude-3.5": "cl100k_base",
    "gemini-1.5-pro": "cl100k_base",
    "gemini-1.5-flash": "cl100k_base",
    "llama3": "cl100k_base",
    "llama3.1": "cl100k_base",
    "mistral-large": "cl100k_base",
    "mistral-small": "cl100k_base",
    "qwen2.5": "cl100k_base",
    "qwen1.5": "cl100k_base",
    "deepseek-v3": "cl100k_base",
    "deepseek-chat": "cl100k_base",
}

# Cost per 1k tokens in USD (set realistic estimates for popular models)
PRICE_PER_1K = {
    "gpt-4o": 0.005,
    "gpt-4o-mini": 0.00015,
    "gpt-4.1": 0.01,
    "gpt-4.1-mini": 0.002,
    "gpt-3.5-turbo": 0.0005,
    "o1": 0.003,
    "o1-mini": 0.0003,
    "claude-3.7": 0.003,
    "claude-3.5": 0.0015,
    "gemini-1.5-pro": 0.0025,
    "gemini-1.5-flash": 0.001,
    "llama3": 0.0001,
    "llama3.1": 0.0001,
    "mistral-large": 0.0002,
    "mistral-small": 0.0001,
    "qwen2.5": 0.0002,
    "qwen1.5": 0.0001,
    "deepseek-v3": 0.00015,
    "deepseek-chat": 0.0001,
}

INR_PER_USD = 83  # exchange rate

# -----------------------------
# HELPERS
# -----------------------------
def load_text(path: Path) -> str:
    return path.read_text(encoding="utf8")

def estimate_tokens_fallback(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))

def tokenize_tiktoken(text, model_name):
    if not HAVE_TIKTOKEN:
        return estimate_tokens_fallback(text)
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def token_count_for_all_models(text: str):
    results = {}
    for model, enc_name in TOKENIZER_MODELS.items():
        try:
            results[model] = tokenize_tiktoken(text, enc_name)
        except Exception:
            results[model] = estimate_tokens_fallback(text)
    return results

def write_log(message: str):
    with open("token_report.log", "a", encoding="utf8") as f:
        f.write(message + "\n")

def ensure_chart_dir():
    if not os.path.exists("charts"):
        os.makedirs("charts")

def save_chart(name: str):
    plt.tight_layout()
    plt.savefig(f"charts/{name}.png")
    plt.close()

# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--encoded", required=True)
    args = ap.parse_args()

    orig_text = load_text(Path(args.original))
    enc_text = load_text(Path(args.encoded))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_log(f"\n=== TOKEN REPORT GENERATED AT {timestamp} ===\n")

    orig_counts = token_count_for_all_models(orig_text)
    enc_counts = token_count_for_all_models(enc_text)

    df = pd.DataFrame({
        "model": list(TOKENIZER_MODELS.keys()),
        "original": list(orig_counts.values()),
        "encoded": list(enc_counts.values()),
    })

    df["savings_pct"] = 100 - (df["encoded"] / df["original"] * 100)
    df["compression_factor"] = df["original"] / df["encoded"]
    df["difference"] = df["original"] - df["encoded"]

    # Cost calculation in USD and INR
    df["orig_cost_usd"] = df.apply(lambda r: (r["original"] / 1000) * PRICE_PER_1K.get(r["model"], 0), axis=1)
    df["enc_cost_usd"] = df.apply(lambda r: (r["encoded"] / 1000) * PRICE_PER_1K.get(r["model"], 0), axis=1)
    df["savings_usd"] = df["orig_cost_usd"] - df["enc_cost_usd"]
    df["savings_inr"] = df["savings_usd"] * INR_PER_USD

    ensure_chart_dir()
    sns.set(style="whitegrid", font="Arial", font_scale=1.1)

    # -----------------------------
    # Token comparison bar chart
    # -----------------------------
    plt.figure(figsize=(12, 6))
    df_m = df.melt(id_vars="model", value_vars=["original", "encoded"], var_name="type", value_name="tokens")
    sns.barplot(data=df_m, x="tokens", y="model", hue="type")
    plt.title("Token Count Comparison (Original vs Encoded)")
    plt.xlabel("Representation Length (Tokens)")
    plt.ylabel("Model")
    save_chart("token_comparison")

    # -----------------------------
    # Token savings percentage ranking
    # -----------------------------
    df_sorted = df.sort_values("savings_pct", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_sorted, x="savings_pct", y="model", palette="viridis")
    plt.title("Token Savings % Ranking")
    plt.xlabel("Savings (%)")
    plt.ylabel("Model")
    save_chart("token_savings_pct_ranking")

    # -----------------------------
    # Cost savings chart (USD & INR)
    # -----------------------------
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df.sort_values("savings_usd", ascending=False), 
                x="savings_usd", y="model", palette="rocket")
    plt.title("Cost Savings per Model (USD)")
    plt.xlabel("Savings (USD)")
    plt.ylabel("Model")
    save_chart("cost_savings_usd")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df.sort_values("savings_inr", ascending=False), 
                x="savings_inr", y="model", palette="rocket")
    plt.title("Cost Savings per Model (INR)")
    plt.xlabel("Savings (INR)")
    plt.ylabel("Model")
    save_chart("cost_savings_inr")

    print("✔ Charts saved in /charts")
    print("✔ Log file saved as token_report.log")

if __name__ == "__main__":
    main()
