import argparse
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

from tqdm import tqdm

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
DEFAULT_CACHE_DIR = "pdbs"


def download_pdb(pdb_id: str, timeout: int = 10) -> str:
    max_tries = 3
    backoff = 1
    for _ in range(max_tries):
        try:
            if not isinstance(pdb_id, str) or not re.fullmatch(r"[0-9A-Za-z]{4}", pdb_id):
                raise ValueError(f"Invalid PDB ID '{pdb_id}'. Must be 4 alphanumeric characters.")

            url = RCSB_URL.format(pdb_id=pdb_id.upper())

            try:
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    if response.status != 200:
                        raise urllib.error.HTTPError(url, response.status, "HTTP error", response.headers, None)
                    data = response.read()
            except urllib.error.HTTPError as e:
                raise urllib.error.HTTPError(e.url, e.code, f"Failed to download PDB file: {e.reason}", e.headers, e.fp)
            except urllib.error.URLError as e:
                raise urllib.error.URLError(f"Network error while downloading PDB file: {e.reason}")

            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("Downloaded file is not valid UTF-8 text.")

            if not text.strip():
                raise ValueError(f"PDB file for ID '{pdb_id}' is empty.")

            return text
        except ValueError:
            sleep(backoff)
            backoff *= 2
    raise ValueError(f"reached max backoff downloading '{pdb_id}'")


def cache_path(pdb_id: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"{pdb_id.upper()}.cif")


def download_pdb_cached(pdb_id: str, cache_dir: str = DEFAULT_CACHE_DIR, timeout: int = 10) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(pdb_id, cache_dir)
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    text = download_pdb(pdb_id, timeout=timeout)
    with open(path, "w") as f:
        f.write(text)
    return text


def download_all(pdb_ids, cache_dir: str = DEFAULT_CACHE_DIR, max_workers: int = 16, timeout: int = 10):
    unique_ids = sorted({pid[:4].upper() for pid in pdb_ids})
    os.makedirs(cache_dir, exist_ok=True)

    succeeded, failed = [], {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(download_pdb_cached, pid, cache_dir, timeout): pid for pid in unique_ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="downloading"):
            pid = futures[fut]
            try:
                fut.result()
                succeeded.append(pid)
            except Exception as e:
                failed[pid] = str(e)

    if failed:
        print(f"\n{len(failed)} structure(s) failed to download:")
        for pid, err in failed.items():
            print(f"  {pid}: {err}")

    return succeeded, failed


def _load_ids_from_file(path: str):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-download and cache .cif files from RCSB.")
    parser.add_argument("--ids", required=True, help="Text file with one PDB ID (or PPPP_C) per line.")
    parser.add_argument("--out", default=DEFAULT_CACHE_DIR, help="Cache directory to write .cif files into.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel download threads.")
    parser.add_argument("--timeout", type=int, default=10, help="Per-request timeout in seconds.")
    args = parser.parse_args()

    ids = _load_ids_from_file(args.ids)
    download_all(ids, cache_dir=args.out, max_workers=args.workers, timeout=args.timeout)
