#!/bin/bash
# Step 2: fold every sequence in a FASTA file with ColabFold (AlphaFold2).
#
# Usage: fold.sh seqs.fasta [name ...]
# With names, only the FASTA records with those names are folded.
#
# MSAs come from the public ColabFold MMseqs2 server, so the job needs outbound
# internet and the sequences leave the cluster.
#
# Output, in ./fold:
#   <name>_af2.pdb      Rank-1 model (AMBER-relaxed when the relaxation worked).
#   fold_scores.json    Mean pLDDT and pTM of each rank-1 model.
#   raw/                Everything colabfold_batch wrote.
#
# AF2 weights: if ./colabfold_params.tar.gz exists the script unpacks it into
# ./params. Otherwise colabfold_batch downloads the weights (about 4 GB).
set -euo pipefail

ALL=$1
shift
FASTA=fold_input.fasta
if [ $# -gt 0 ]; then
    for name in "$@"; do grep -A1 "^>${name}\$" "$ALL"; done > "$FASTA"
else
    cp "$ALL" "$FASTA"
fi
export HOME=$PWD
export XDG_CACHE_HOME=$PWD/.cache
export MPLCONFIGDIR=$PWD/.mpl
mkdir -p fold/raw params "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

nvidia-smi --query-gpu=name,memory.total --format=csv || true
if [ -f colabfold_params.tar.gz ]; then
    tar -xf colabfold_params.tar.gz -C params      # -xf detects the compression (or none)
fi
# ColabFold reads the weights from <data>/params/*.npz. Find that folder wherever
# the archive put it; without weights, <data> stays ./params and ColabFold downloads.
DATA=$PWD/params
NPZ=$(find "$PWD/params" -name 'params_model_*.npz' | head -n 1)
if [ -n "$NPZ" ]; then
    WEIGHTS=$(dirname "$NPZ")
    if [ "$(basename "$WEIGHTS")" != params ]; then
        mkdir -p "$PWD/af2_data"
        ln -s "$WEIGHTS" "$PWD/af2_data/params"
        WEIGHTS=$PWD/af2_data/params
    fi
    DATA=$(dirname "$WEIGHTS")
    echo "AF2 weights: $(ls "$WEIGHTS" | wc -l) files in $WEIGHTS"
    ls "$WEIGHTS"/download*finished.txt >/dev/null 2>&1 || \
        echo "No download_finished.txt marker: ColabFold will download the weights again."
fi

colabfold_batch --data "$DATA" \
    --msa-mode mmseqs2_uniref_env \
    --num-models 5 --num-recycle 3 --rank plddt \
    --amber --num-relax 1 --use-gpu-relax \
    "$FASTA" fold/raw

python3 - "$FASTA" <<'EOF'
import glob, json, shutil, sys

names = [line[1:].split()[0] for line in open(sys.argv[1]) if line.startswith(">")]
scores = {}
for name in names:
    relaxed = sorted(glob.glob(f"fold/raw/{name}_relaxed_rank_001_*.pdb"))
    unrelaxed = sorted(glob.glob(f"fold/raw/{name}_unrelaxed_rank_001_*.pdb"))
    if not relaxed and not unrelaxed:
        sys.exit(f"ColabFold wrote no rank-1 model for {name}")
    model = (relaxed or unrelaxed)[0]
    shutil.copy(model, f"fold/{name}_af2.pdb")
    s = json.load(open(sorted(glob.glob(f"fold/raw/{name}_scores_rank_001_*.json"))[0]))
    scores[name] = dict(model=model.split("/")[-1], relaxed=bool(relaxed),
                        mean_plddt=sum(s["plddt"]) / len(s["plddt"]), ptm=s.get("ptm"))
json.dump(scores, open("fold/fold_scores.json", "w"), indent=2)
print(json.dumps(scores, indent=2))
EOF
