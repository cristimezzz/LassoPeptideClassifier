import argparse
import torch
import pandas as pd
from Bio import SeqIO
from transformers import EsmTokenizer, EsmModel

from config import (
    CHECKPOINT_DIR,
    ESM_MODEL_NAME,
    ESM_BATCH_SIZE,
    MAX_LEN,
    ESM_EMBED_DIM,
    CNN_CHANNELS,
    CNN_KERNELS,
    ATTENTION_HEADS,
    MLP_HIDDEN,
    DROPOUT,
)
from model import LassoPeptideClassifier


def extract_esm2_embeddings_inference(
    sequences, model_name=ESM_MODEL_NAME, batch_size=ESM_BATCH_SIZE, max_len=MAX_LEN
):
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding="max_length",
                max_length=max_len,
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            last_hidden = last_hidden * mask
            all_embeddings.append(last_hidden.cpu())

    return torch.cat(all_embeddings, dim=0)


def predict_fasta(input_fasta, output_csv, checkpoint=None):
    print(f"[*] Reading sequences from {input_fasta}")
    records = list(SeqIO.parse(input_fasta, "fasta"))
    if not records:
        print("[-] No sequences found")
        return

    seq_ids = [r.id for r in records]
    sequences = [str(r.seq) for r in records]

    print(f"[*] Extracting ESM-2 embeddings for {len(sequences)} sequences...")
    embeddings = extract_esm2_embeddings_inference(sequences)

    ckpt_path = checkpoint or CHECKPOINT_DIR + "/best_model.pt"
    print(f"[*] Loading model from {ckpt_path}")
    model = LassoPeptideClassifier(
        embed_dim=ESM_EMBED_DIM,
        cnn_channels=CNN_CHANNELS,
        cnn_kernels=CNN_KERNELS,
        attention_heads=ATTENTION_HEADS,
        mlp_hidden=MLP_HIDDEN,
        dropout=DROPOUT,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        batch_size = 64
        all_probs = []
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size]
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(-1)
            all_probs.append(probs)
        probabilities = torch.cat(all_probs).numpy()

    results = pd.DataFrame(
        {
            "sequence_id": seq_ids,
            "probability": probabilities,
            "prediction": ["positive" if p >= 0.5 else "negative" for p in probabilities],
        }
    )
    results.to_csv(output_csv, index=False)
    print(f"[+] Results saved to {output_csv}")
    print(results.head())


def main():
    parser = argparse.ArgumentParser(description="Predict lasso peptide probability from FASTA")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    parser.add_argument("-o", "--output", default="predictions.csv", help="Output CSV file")
    parser.add_argument("-c", "--checkpoint", default=None, help="Model checkpoint path")
    args = parser.parse_args()

    predict_fasta(args.input, args.output, args.checkpoint)


if __name__ == "__main__":
    main()
