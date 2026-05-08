"""Data preparation pipeline: download → dedup → ESM-2 extraction → dataset split.

This module consolidates the full data workflow:
  1. Sequence deduplication — CD-HIT binary with pure-Python greedy fallback.
  2. Data download — positive samples from LassoPred API, negatives from UniProt REST API.
  3. ESM-2 inference — frozen embedding extraction for all sequences.
  4. Dataset split — stratified 80/10/10 train/val/test split, saved as .pt tensors.

Can run standalone or be imported by run_experiment.py.
"""

import argparse
import os
import torch
import requests
import subprocess
from sklearn.model_selection import train_test_split

from config import (
    CLEAN_POS,
    CLEAN_NEG,
    DATASET_DIR,
    ESM_MODEL_NAME,
    MAX_LEN,
    CD_HIT_THRESHOLD,
    RANDOM_SEED,
    RAW_POS,
    RAW_NEG,
    UNIPROT_NEG_LIMIT,
    get_esm_embed_dim,
    get_model_info,
    ensure_dirs,
)
from utils import extract_esm2_embeddings, load_esm_model


# ═══════════════════════════════════════════════════════════
#  Pure-Python sequence deduplication (CD-HIT fallback)
# ═══════════════════════════════════════════════════════════

def _get_aligner():
    """Configure a BioPython PairwiseAligner for ungapped sequence identity.

    Uses local alignment mode with effectively infinite gap penalties,
    so the score reduces to match_count = number of identical positions
    in the maximal ungapped overlap.
    """
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1e9
    aligner.extend_gap_score = -1e9
    return aligner


def _sequence_identity(a, b, aligner):
    """Compute sequence identity as matches / min(len(a), len(b)) with ungapped alignment.

    Uses the same metric as CD-HIT: number of matches in the best ungapped local
    alignment divided by the shorter sequence length.
    """
    if a == b:
        return 1.0
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    score = aligner.align(a, b).score  # match_count (match=1, mismatch=0)
    return score / min_len


def deduplicate_fasta(input_fasta, output_fasta, threshold=0.5):
    """Pure-Python sequence deduplication using the CD-HIT greedy clustering algorithm.

    Pipeline:
      1. Exact duplicate removal via sequence-string hashing.
      2. Sort by length descending (CD-HIT heuristic: longest representative first).
      3. Greedy clustering: for each sequence, compare against current representatives;
         if identity >= threshold, discard; otherwise add as a new cluster representative.

    Args:
        input_fasta: path to input FASTA file.
        output_fasta: path to write deduplicated FASTA.
        threshold: sequence identity threshold (0.0–1.0). Default 0.5 (50% identity).
    """
    from Bio import SeqIO

    if os.path.exists(output_fasta):
        n = sum(1 for _ in SeqIO.parse(output_fasta, "fasta"))
        print(f"[~] CD-HIT cached: {output_fasta} ({n} seqs)")
        return

    records = list(SeqIO.parse(input_fasta, "fasta"))
    n_input = len(records)
    if n_input == 0:
        with open(output_fasta, "w") as f:
            pass
        print(f"[+] CD-HIT: 0 → 0 sequences")
        return

    # 1. Exact dedup by sequence string
    seen = {}
    for rec in records:
        seq = str(rec.seq).upper()
        if seq not in seen:
            seen[seq] = rec
    unique = list(seen.values())
    n_exact_removed = n_input - len(unique)

    # 2. If threshold >= 1.0, we are done after exact dedup
    if threshold >= 1.0:
        with open(output_fasta, "w") as f:
            SeqIO.write(unique, f, "fasta")
        n_kept = len(unique)
        print(f"[+] CD-HIT: {n_input} → {n_kept} sequences  "
              f"(exact dup: {n_exact_removed}) → {output_fasta}")
        return

    # 3. Greedy clustering by length descending (same heuristic as CD-HIT)
    indexed = [(str(r.seq).upper(), r) for r in unique]
    indexed.sort(key=lambda x: len(x[0]), reverse=True)
    aligner = _get_aligner()

    representatives = []

    for seq_str, rec in indexed:
        keep = True
        for rep_str, _ in representatives:
            identity = _sequence_identity(seq_str, rep_str, aligner)
            if identity >= threshold:
                keep = False
                break
        if keep:
            representatives.append((seq_str, rec))

    n_near_removed = len(unique) - len(representatives)
    with open(output_fasta, "w") as f:
        for _, rec in representatives:
            SeqIO.write(rec, f, "fasta")

    print(f"[+] CD-HIT: {n_input} → {len(representatives)} sequences  "
          f"(exact dup: {n_exact_removed}, near dup: {n_near_removed}) → {output_fasta}")


# ═══════════════════════════════════════════════════════════
#  Data download
# ═══════════════════════════════════════════════════════════

def fetch_lassopred_to_fasta(output_fasta):
    """Download positive (lasso peptide) sequences from the LassoPred API.

    Fetches all lasso peptide records with their precursor sequences,
    filtering out entries with empty sequences. Outputs a FASTA file
    with LP_ID as the header.

    Args:
        output_fasta: path to write the downloaded FASTA.

    Returns:
        Number of valid sequences written.

    Raises:
        requests.exceptions.RequestException: on network failure.
    """
    print("[+] Downloading positive samples (LassoPred)...")
    api_url = "https://lassopred.accre.vanderbilt.edu/api/data/?page=1&size=4029"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        data_list = response.json()["data"]

        valid_count = 0
        with open(output_fasta, "w", encoding="utf-8") as f:
            for item in data_list:
                seq_id = item.get("LP_ID", "")
                sequence = item.get("Precursor_Sequence", "").strip()
                if sequence:
                    f.write(f">{seq_id}\n{sequence}\n")
                    valid_count += 1

        print(f"[+] Downloaded {valid_count} positive sequences → {output_fasta}")
        return valid_count

    except requests.exceptions.RequestException as e:
        print(f"[-] Network error: {e}")
        raise
    except (KeyError, TypeError, IndexError) as e:
        print(f"[-] Parse error: {e}")
        raise


def fetch_uniprot_negatives(output_fasta, limit=UNIPROT_NEG_LIMIT):
    """Download negative (non-lasso) bacterial protein sequences from UniProt REST API.

    Query criteria:
      - Taxonomy: bacteria (taxonomy_id:2)
      - Length: 40–100 amino acids (matching lasso peptide size range)
      - Reviewed: Swiss-Prot entries only
      - Excludes: entries annotated with "lasso peptide" family

    UniProt's streaming endpoint returns one FASTA entry per line pair
    (header, sequence). The counting logic writes exactly `limit` complete
    entries by checking the header count before starting a new entry.

    Args:
        output_fasta: path to write the downloaded FASTA.
        limit: max number of sequences to download. Default from config.

    Returns:
        Number of sequences written.

    Raises:
        requests.exceptions.RequestException: on network failure.
    """
    print("[+] Downloading negative samples (UniProt)...")
    query = (
        'taxonomy_id:2 AND length:[40 TO 100] AND reviewed:true '
        'NOT family:"lasso peptide"'
    )
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {"query": query, "format": "fasta", "size": limit}
    headers = {
        "User-Agent": "LassoPeptideClassifier/1.0"
    }

    try:
        response = requests.get(url, params=params, headers=headers, stream=True,
                               timeout=(30, 60))
        response.raise_for_status()

        count = 0
        pending_sequence = False  # True after header, False after first sequence line
        with open(output_fasta, "w", encoding="utf-8") as f:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith(">"):
                    if count >= limit:
                        break                    # already wrote limit complete entries
                    count += 1
                    pending_sequence = True
                elif pending_sequence:
                    pending_sequence = False     # first sequence line after header
                f.write(line + "\n")

        print(f"[+] Downloaded {count} negative sequences → {output_fasta}")
        return count

    except requests.exceptions.RequestException as e:
        print(f"[-] Network error: {e}")
        raise


# ═══════════════════════════════════════════════════════════
#  CD-HIT wrapper
# ═══════════════════════════════════════════════════════════

def run_cd_hit(input_fasta, output_fasta, threshold=0.5):
    """Run CD-HIT sequence clustering, with automatic pure-Python fallback.

    Tries the native `cd-hit` binary first. If the binary is not found or fails
    (e.g. on Windows where it has no native build), falls back to the pure-Python
    greedy clustering in deduplicate_fasta().

    Args:
        input_fasta: path to raw FASTA file.
        output_fasta: path to write deduplicated FASTA.
        threshold: sequence identity threshold (0.0–1.0). Default 0.5.
    """
    if os.path.exists(output_fasta):
        from Bio import SeqIO
        n = sum(1 for _ in SeqIO.parse(output_fasta, "fasta"))
        print(f"[~] CD-HIT cached: {output_fasta} ({n} seqs)")
        return

    print(f"[*] Running CD-HIT (threshold={threshold})...")
    try:
        cmd = ["cd-hit", "-i", input_fasta, "-o", output_fasta,
               "-c", str(threshold), "-n", "3", "-M", "0", "-d", "0"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, shell=False)
        from Bio import SeqIO
        n_before = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))
        n_after = sum(1 for _ in SeqIO.parse(output_fasta, "fasta"))
        print(f"[+] CD-HIT ok: {n_before} → {n_after} sequences → {output_fasta}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[!] CD-HIT not available ({e}), using pure-Python fallback...")
        deduplicate_fasta(input_fasta, output_fasta, threshold)


# ═══════════════════════════════════════════════════════════
#  ESM extraction + dataset split
# ═══════════════════════════════════════════════════════════

def create_and_split_dataset(pos_fasta, neg_fasta, output_dir=DATASET_DIR, esm_model_name=None, seed=None):
    """Full data pipeline: load FASTA → downsample → ESM-2 extract → stratified split → save.

    Steps:
      1. Load positive and negative FASTA files.
      2. Downsample negatives to at most 3× positives (class balancing).
      3. Write records to temp FASTA files for ESM-2 extraction.
      4. Extract frozen ESM-2 embeddings (GPU-accelerated, batch-wise).
      5. Stratified 80/10/10 train/val/test split (stratify by label).
      6. Save {train, val, test}.pt to output_dir.

    Args:
        pos_fasta: path to cleaned positive FASTA.
        neg_fasta: path to cleaned negative FASTA.
        output_dir: directory to write .pt tensor files. Default DATASET_DIR.
        esm_model_name: HuggingFace ESM-2 model ID. Default from config.
        seed: random seed for downsampling and splitting. Default from config.
    """
    ensure_dirs()
    if esm_model_name is None:
        esm_model_name = ESM_MODEL_NAME
    if seed is not None:
        import numpy as np
        torch.manual_seed(seed)
        np.random.seed(seed)

    info = get_model_info(esm_model_name)
    batch_size = info["batch"]

    esm_model, tokenizer, device = load_esm_model(esm_model_name)
    embed_dim = get_esm_embed_dim(esm_model_name)
    print(f"    ESM-2: {info['label']} | embed_dim={embed_dim}")

    from Bio import SeqIO
    pos_records = list(SeqIO.parse(pos_fasta, "fasta"))
    neg_records = list(SeqIO.parse(neg_fasta, "fasta"))

    # Downsample negatives to 3× positives for class balance
    if len(neg_records) > len(pos_records) * 3:
        rng_seed = seed if seed is not None else RANDOM_SEED
        rng = torch.Generator().manual_seed(rng_seed)
        idx = torch.randperm(len(neg_records), generator=rng)[: len(pos_records) * 3]
        neg_records = [neg_records[i] for i in idx]
        print(f"[i] Downsampled negatives: {len(neg_records)} (x{len(pos_records)*3})")

    # Write to temp files for extract_esm2_embeddings (expects file paths)
    import tempfile
    from Bio.SeqIO import write
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f_pos:
        write(pos_records, f_pos, "fasta")
        tmp_pos = f_pos.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f_neg:
        write(neg_records, f_neg, "fasta")
        tmp_neg = f_neg.name

    # Extract frozen ESM-2 embeddings
    try:
        pos_ids, pos_features = extract_esm2_embeddings(
            tmp_pos, esm_model_name, esm_model, tokenizer, device, batch_size, MAX_LEN
        )
        pos_labels = torch.ones(pos_features.size(0), dtype=torch.float32)

        neg_ids, neg_features = extract_esm2_embeddings(
            tmp_neg, esm_model_name, esm_model, tokenizer, device, batch_size, MAX_LEN
        )
        neg_labels = torch.zeros(neg_features.size(0), dtype=torch.float32)
    finally:
        # Clean up temp files even on exception
        os.remove(tmp_pos)
        os.remove(tmp_neg)

    del esm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Concatenate all data
    X = torch.cat([pos_features, neg_features], dim=0)
    y = torch.cat([pos_labels, neg_labels], dim=0)
    all_ids = pos_ids + neg_ids

    print(f"[*] Feature tensor shape: {X.shape}")
    print(f"[*] Pos: {len(pos_ids)}, Neg: {len(neg_ids)}")

    # Stratified split: 80% train, 10% val, 10% test
    split_seed = seed if seed is not None else RANDOM_SEED
    X_temp, X_test, y_temp, y_test, id_temp, id_test = train_test_split(
        X, y, all_ids, test_size=0.1, random_state=split_seed, stratify=y
    )
    # 0.111 * 0.9 ≈ 0.1 of total → val split
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_temp, y_temp, id_temp, test_size=0.111, random_state=split_seed, stratify=y_temp
    )

    for name, x, yy, ids in [
        ("train", X_train, y_train, id_train),
        ("val", X_val, y_val, id_val),
        ("test", X_test, y_test, id_test),
    ]:
        torch.save({"X": x, "y": yy, "ids": ids}, os.path.join(output_dir, f"{name}.pt"))

    print(f"[+] Dataset saved → {output_dir}/")


# ═══════════════════════════════════════════════════════════
#  CLI entry points
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data preparation pipeline: download, CD-HIT dedup, ESM-2 extraction, dataset split"
    )
    parser.add_argument("--esm-model", default=None, help="ESM-2 model name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download data, skip CD-HIT and ESM extraction")
    args = parser.parse_args()

    ensure_dirs()

    if args.download_only:
        fetch_lassopred_to_fasta(RAW_POS)
        fetch_uniprot_negatives(RAW_NEG)
    else:
        run_cd_hit(RAW_POS, CLEAN_POS, threshold=CD_HIT_THRESHOLD)
        run_cd_hit(RAW_NEG, CLEAN_NEG, threshold=CD_HIT_THRESHOLD)
        create_and_split_dataset(CLEAN_POS, CLEAN_NEG, esm_model_name=args.esm_model, seed=args.seed)
