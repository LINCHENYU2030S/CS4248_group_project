"""
Fine-tuning script for helinivan/english-sarcasm-detector
- Input:  your labeled Excel file
- Model:  BERT-based sarcasm classifier from Hugging Face
- Device: CPU-friendly (small batch size, fewer epochs)
- Goal:   improve accuracy + adapt to your domain/style
- Splits: 70% train | 10% validation (during training) | 20% test (final evaluation)
- Charts: training/validation loss & accuracy curves + confusion matrix saved as PNG
"""

import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────

EXCEL_PATH   = "./cleaned_dataset.csv"          # ← change to your file path
TEXT_COL     = "headline"                # column containing the text
LABEL_COL    = "is_sarcastic"            # column containing 0/1 labels

MODEL_NAME   = "helinivan/english-sarcasm-detector"
OUTPUT_DIR   = "./finetuned_sarcasm_model"
CHARTS_DIR   = "./training_charts"
RANDOM_NO    = random.randint(0,100)

MAX_LEN      = 128      # max tokens per headline
BATCH_SIZE   = 32        
EPOCHS       = 5        # increase if val loss still dropping
LR           = 2e-6     # standard learning rate for BERT fine-tuning
TEST_SIZE    = 0.20     # 20% held out as final test set (never seen during training)
VAL_SIZE     = 0.10     # 10% of total used for validation during training
RANDOM_SEED  = 42

# ── DATASET CLASS ─────────────────────────────────────────────────────────────

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── LOAD & PREPARE DATA ───────────────────────────────────────────────────────

def load_data(path):
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)

    assert TEXT_COL  in df.columns, f"Column '{TEXT_COL}' not found in Excel file"
    assert LABEL_COL in df.columns, f"Column '{LABEL_COL}' not found in Excel file"

    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df[TEXT_COL]  = df[TEXT_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    print(f"  Total rows:        {len(df)}")
    print(f"  Sarcastic (1):     {df[LABEL_COL].sum()}")
    print(f"  Not sarcastic (0): {(df[LABEL_COL] == 0).sum()}")
    return df

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        preds_all.extend(preds)
        labels_all.extend(labels.cpu().numpy())

    return total_loss / len(loader), accuracy_score(labels_all, preds_all)

# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    total_loss, preds_all, labels_all = 0, [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs    = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels.cpu().numpy())

    return total_loss / len(loader), accuracy_score(labels_all, preds_all), preds_all, labels_all

# ── VISUALISATION ─────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_dir: str):
    """
    Saves a 2-panel chart:
      Left  — training vs validation LOSS per epoch
      Right — training vs validation ACCURACY per epoch
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")

    # Loss
    ax1.plot(epochs, history["train_loss"], "o-",  color="#E05C5C", label="Train loss")
    ax1.plot(epochs, history["val_loss"],   "o--", color="#5C9BE0", label="Val loss")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(list(epochs))

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "o-",  color="#E05C5C", label="Train acc")
    ax2.plot(epochs, history["val_acc"],   "o--", color="#5C9BE0", label="Val acc")
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(list(epochs))

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Training curves saved      → {path}")


def plot_confusion_matrix(labels, preds, save_dir: str, split_name: str = "Test"):
    """
    Saves a confusion matrix heatmap for the given split.
    """
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Sarcastic", "Sarcastic"],
        yticklabels=["Not Sarcastic", "Sarcastic"],
        ax=ax,
    )
    ax.set_title(f"{split_name} Set — Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    fname = f"confusion_matrix_{split_name.lower()}.png"
    path  = os.path.join(save_dir, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Confusion matrix saved     → {path}")


def plot_prediction_confidence(test_labels, test_preds, confidences, save_dir: str):
    """
    Saves a strip-plot showing model confidence for correct vs incorrect predictions.
    Helps you see where the model is uncertain or overconfident.
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame({
        "confidence": confidences,
        "correct":    [p == l for p, l in zip(test_preds, test_labels)],
        "true_label": ["Sarcastic" if l == 1 else "Not Sarcastic" for l in test_labels],
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {True: "#4CAF50", False: "#E05C5C"}
    for correct, grp in df.groupby("correct"):
        ax.scatter(
            grp["confidence"],
            grp["true_label"],
            alpha=0.5,
            color=colors[correct],
            label="Correct" if correct else "Wrong",
            s=40,
        )
    ax.set_title("Prediction Confidence on Test Set", fontsize=13, fontweight="bold")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("True Label")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    path = os.path.join(save_dir, "prediction_confidence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Confidence plot saved      → {path}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # 1. Load data
    df = load_data(EXCEL_PATH)
    texts  = df[TEXT_COL].tolist()
    labels = df[LABEL_COL].tolist()

    # 2. Three-way split: 70% train | 10% val | 20% test
    #    First carve out the 20% test set, then split remainder into train/val
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        texts, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)   # val fraction of the remaining 80%
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        random_state=RANDOM_SEED,
        stratify=y_trainval,
    )
    print(f"\nData split:")
    print(f"  Train : {len(X_train):>5} samples  (~70%)")
    print(f"  Val   : {len(X_val):>5} samples  (~10%)")
    print(f"  Test  : {len(X_test):>5} samples  (~20%) ← held out until after training")

    # 3. Tokenizer + model
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2,)
    model.to(device)

    # 4. Datasets + DataLoaders
    train_dataset = SarcasmDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset   = SarcasmDataset(X_val,   y_val,   tokenizer, MAX_LEN)
    test_dataset  = SarcasmDataset(X_test,  y_test,  tokenizer, MAX_LEN)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # 5. Optimizer + scheduler
    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # 6. Training loop — collect history for charts
    print("\n── Starting training ──")
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc        = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, _, _      = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  Train — loss: {train_loss:.4f} | acc: {train_acc:.4f}")
        print(f"  Val   — loss: {val_loss:.4f}   | acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  ✅ Best model saved → {OUTPUT_DIR}")

    # 7. Plot training curves (loss + accuracy per epoch)
    print("\n── Generating charts ──")
    plot_training_curves(history, CHARTS_DIR)

    # 8. Final evaluation on the held-out 20% TEST set using best saved model
    print("\n── Evaluating on held-out 20% test set ──")
    best_model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
    best_model.to(device)

    test_loss, test_acc, test_preds, test_labels_out = evaluate(best_model, test_loader, device)
    print(f"  Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")
    print("\n" + classification_report(
        test_labels_out, test_preds,
        target_names=["Not Sarcastic", "Sarcastic"]
    ))

    # 9. Confusion matrix on test set
    plot_confusion_matrix(test_labels_out, test_preds, CHARTS_DIR, split_name="Test")

    # 10. Confidence scatter plot on test set
    best_model.eval()
    confidences = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits         = best_model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs          = torch.softmax(logits, dim=1).cpu().numpy()
            for p, pred in zip(probs, np.argmax(probs, axis=1)):
                confidences.append(float(p[pred]))

    plot_prediction_confidence(test_labels_out, test_preds, confidences, CHARTS_DIR)

    print(f"\n✅ Done! All outputs saved:")
    print(f"   Model  → {OUTPUT_DIR}/")
    print(f"   Charts → {CHARTS_DIR}/")
    print(f"     • training_curves.png      (loss & accuracy per epoch)")
    print(f"     • confusion_matrix_test.png (true vs predicted labels)")
    print(f"     • prediction_confidence.png (correct vs wrong by confidence)")

# ── INFERENCE HELPER (use after training) ─────────────────────────────────────

def predict(texts: list[str], model_dir: str = OUTPUT_DIR) -> list[dict]:
    """
    Load the fine-tuned model and run inference on a list of headlines.

    Usage:
        results = predict(["Scientists discover cure for Mondays"])
        print(results)
        # [{'text': 'Scientists discover cure for Mondays', 'label': 1, 'confidence': 0.97}]
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    results = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=1).squeeze()
        label  = int(torch.argmax(probs))
        results.append({
            "text":       text,
            "label":      label,           # 0 = not sarcastic, 1 = sarcastic
            "confidence": round(float(probs[label]), 4),
        })
    return results


if __name__ == "__main__":
    main()