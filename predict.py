"""Standalone inference on new FASTA sequences.

Accepts a FASTA file of protein sequences, extracts ESM-2 embeddings using the
frozen pre-trained model, and runs the trained classifier to produce per-sequence
probabilities. Outputs a CSV with columns: sequence_id, probability, prediction.

Usage:
    python predict.py -i sequences.fasta -o results.csv
    python predict.py -i sequences.fasta --esm-model facebook/esm2_t12_35M_UR50D
"""

import argparse
import os
import torch
import pandas as pd
from Bio import SeqIO

from config import (
    CHECKPOINT_DIR,
    ESM_MODEL_NAME,
    PRED_BATCH_SIZE,
    MAX_LEN,
    get_esm_embed_dim,
    get_model_info,
    ensure_dirs,
)
from utils import load_esm_model, extract_esm2_embeddings, load_classifier_from_checkpoint


def predict_fasta(input_fasta, output_csv, checkpoint=None, esm_model_name=None):
    """Predict lasso peptide probabilities for all sequences in a FASTA file.

    Pipeline:
      1. Parse FASTA file and validate it contains sequences.
      2. Load the ESM-2 frozen encoder and extract per-token embeddings.
      3. Load the trained classifier checkpoint.
      4. Run batched inference to produce probability per sequence.
      5. Output a CSV with columns: sequence_id, probability, prediction.

    Args:
        input_fasta: path to input FASTA file.
        output_csv: path to write results CSV.
        checkpoint: path to model checkpoint. Defaults to checkpoints/best_model.pt.
        esm_model_name: HuggingFace ESM-2 model ID. Default from config.
    """
    ensure_dirs()
    if esm_model_name is None:
        esm_model_name = ESM_MODEL_NAME

    info = get_model_info(esm_model_name)
    embed_dim = get_esm_embed_dim(esm_model_name)
    batch_size = info["batch"]
    print(f"[*] ESM-2: {info['label']} | embed_dim={embed_dim}")

    # Parse and validate input
    print(f"[*] Reading sequences from {input_fasta}")
    records = list(SeqIO.parse(input_fasta, "fasta"))
    if not records:
        print("[-] No sequences found in input file")
        return

    print(f"[*] Found {len(records)} sequences, extracting ESM-2 embeddings...")

    # Extract frozen ESM-2 embeddings (seq_ids are from the same parse, ensuring alignment)
    esm_model, tokenizer, device = load_esm_model(esm_model_name)
    seq_ids, embeddings = extract_esm2_embeddings(
        input_fasta, esm_model_name, esm_model, tokenizer, device, batch_size, MAX_LEN
    )

    # Load trained classifier
    ckpt_path = checkpoint or os.path.join(CHECKPOINT_DIR, "best_model.pt")
    print(f"[*] Loading model from {ckpt_path}")
    model = load_classifier_from_checkpoint(ckpt_path, embed_dim, device)

    # Batched inference (can use larger batch than ESM-2 extraction)
    with torch.no_grad():
        all_probs = []
        for i in range(0, len(embeddings), PRED_BATCH_SIZE):
            batch = embeddings[i : i + PRED_BATCH_SIZE].to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(-1)  # (batch,) probabilities
            all_probs.append(probs.cpu())
        probabilities = torch.cat(all_probs).numpy()

    # Build output DataFrame
    results = pd.DataFrame({
        "sequence_id": seq_ids,
        "probability": probabilities,
        "prediction": ["positive" if p >= 0.5 else "negative" for p in probabilities],
    })
    results.to_csv(output_csv, index=False)
    print(f"[+] Results saved to {output_csv}")
    print(results.head())


def main():
    """CLI entry point for predict.py."""
    parser = argparse.ArgumentParser(description="Predict lasso peptide probability from FASTA")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    parser.add_argument("-o", "--output", default="predictions.csv", help="Output CSV file")
    parser.add_argument("-c", "--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--esm-model", default=None, help="ESM-2 model name")
    args = parser.parse_args()
    predict_fasta(args.input, args.output, args.checkpoint, args.esm_model)


if __name__ == "__main__":
    main()
