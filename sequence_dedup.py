import os
from Bio import SeqIO


def _get_aligner():
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1e9
    aligner.extend_gap_score = -1e9
    return aligner


def _sequence_identity(a, b, aligner):
    if a == b:
        return 1.0
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    score = aligner.align(a, b).score
    return score / min_len


def deduplicate_fasta(input_fasta, output_fasta, threshold=0.5):
    """Pure-Python CD-HIT fallback: greedy clustering on global sequence identity.

    Uses the same metric as CD-HIT: matches / min_len over ungapped global alignment.
    """
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

    representatives = []  # list of (seq_str, record)

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
