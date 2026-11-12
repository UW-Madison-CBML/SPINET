#!/bin/bash
# Reports the GPU and whether the execute node can reach the ColabFold MSA server.
nvidia-smi
python3 - <<'PY'
import urllib.request
for url in ("https://api.colabfold.com", "https://storage.googleapis.com/alphafold/"):
    try:
        urllib.request.urlopen(url, timeout=20)
        print("reachable:", url)
    except urllib.error.HTTPError as e:
        print("reachable:", url, f"(HTTP {e.code})")
    except Exception as e:
        print("NOT reachable:", url, e)
PY
colabfold_batch --help | head -3
