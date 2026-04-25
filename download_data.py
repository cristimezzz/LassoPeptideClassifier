import os
import requests

from config import DATA_DIR, UNIPROT_NEG_LIMIT


def ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def fetch_lassopred_to_fasta(output_fasta):
    print("[+] Downloading positive samples (LassoPred)...")
    api_url = "https://lassopred.accre.vanderbilt.edu/api/data/?page=1&size=4029"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(api_url, headers=headers)
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
    except Exception as e:
        print(f"[-] Parse error: {e}")
        raise


def fetch_uniprot_negatives(output_fasta, limit=UNIPROT_NEG_LIMIT):
    print("[+] Downloading negative samples (UniProt)...")
    query = (
        'taxonomy_id:2 AND length:[40 TO 100] AND reviewed:true '
        'NOT family:"lasso peptide"'
    )
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {"query": query, "format": "fasta", "size": limit}

    try:
        response = requests.get(url, params=params, stream=True)
        response.raise_for_status()

        count = 0
        with open(output_fasta, "w") as f:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                f.write(line + "\n")
                if line.startswith(">"):
                    count += 1
                if count >= limit:
                    break

        print(f"[+] Downloaded {count} negative sequences → {output_fasta}")
        return count

    except requests.exceptions.RequestException as e:
        print(f"[-] Network error: {e}")
        raise


if __name__ == "__main__":
    from config import RAW_POS, RAW_NEG

    ensure_dir()
    fetch_lassopred_to_fasta(RAW_POS)
    fetch_uniprot_negatives(RAW_NEG)
