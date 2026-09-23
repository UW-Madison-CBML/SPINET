# AF3 Pullback Workflow v2

## 0. Starting point

Inputs:
- PCA CHTC outputs: `../full_pca/*.tar.gz`
- Sheaf selected sequences: `../predicted_sequence_preprocessing/selected/sheaf_320_selected.csv`
- AF3 weights: `af3.bin.zst`

---

## 1. Build Sheaf AF3 packages locally

Test first:

```bash
python build_pullback_batch.py \
  --pca-tar-dir ../full_pca \
  --sequences ../predicted_sequence_preprocessing/selected/sheaf_320_selected.csv \
  --temperature 320 \
  --method sheaf \
  --max-domains 3 \
  --out af3_batch_320_sheaf_test
```

Full:

```bash
python build_pullback_batch.py \
  --pca-tar-dir ../full_pca \
  --sequences ../predicted_sequence_preprocessing/selected/sheaf_320_selected.csv \
  --temperature 320 \
  --method sheaf \
  --out af3_batch_320_sheaf
```

Before CHTC, expected local tree:

```text
af3_batch_320_sheaf_test/
├── packages/
│   ├── 320__sheaf__12asA00.tar.gz
│   ├── 320__sheaf__1aa7A02.tar.gz
│   └── 320__sheaf__1aepA00.tar.gz
├── references/
│   ├── 12asA00/
│   │   ├── 12asA00__state_00__target.pdb
│   │   └── ...
│   └── ...
└── manifests/
    ├── packages_320_sheaf.tsv
    ├── af3_jobs_320_sheaf.tsv
    └── build_failures_320_sheaf.csv
```

Check build failures:

```bash
cat af3_batch_320_sheaf_test/manifests/build_failures_320_sheaf.csv
```

---

## 2. Build native AF3 packages locally

No native CSV is needed. Native sequence is parsed from each PCA representative-state PDB.

Test:

```bash
python build_native_pullback_batch.py \
  --pca-tar-dir ../full_pca \
  --temperature 320 \
  --max-domains 3 \
  --out af3_batch_320_native_test
```

Full:

```bash
python build_native_pullback_batch.py \
  --pca-tar-dir ../full_pca \
  --temperature 320 \
  --out af3_batch_320_native
```

Expected:

```text
af3_batch_320_native_test/
├── packages/
│   ├── 320__native__12asA00.tar.gz
│   └── ...
├── references/
│   └── <domain>/
└── manifests/
    ├── packages_320_native.tsv
    ├── af3_jobs_320_native.tsv
    └── build_failures_320_native.csv
```

Each domain has `1 native sequence × 5 states = 5 AF3 folds`.

---

## 3. Make a headerless HTCondor queue file

This step fixes the previous bug where the header/title line of `packages_*.tsv` was submitted as an extra job.

Sheaf:

```bash
python make_condor_queue.py \
  --packages-manifest af3_batch_320_sheaf_test/manifests/packages_320_sheaf.tsv \
  --out af3_batch_320_sheaf_test/manifests/queue_320_sheaf.tsv
```

Native:

```bash
python make_condor_queue.py \
  --packages-manifest af3_batch_320_native_test/manifests/packages_320_native.tsv \
  --out af3_batch_320_native_test/manifests/queue_320_native.tsv
```

The queue file must have NO header:

```text
320__sheaf__12asA00    packages/320__sheaf__12asA00.tar.gz
320__sheaf__1aa7A02    packages/320__sheaf__1aa7A02.tar.gz
320__sheaf__1aepA00    packages/320__sheaf__1aepA00.tar.gz
```

---

## 4. Upload category-specific files to CHTC

Do not upload `references/`; they stay local for scoring.

Example CHTC tree before submit:

```text
pullback_320_sheaf_test/
├── af3.bin.zst
├── af3_domain_batch_v2.sub
├── run_af3_domain_batch.sh
├── manifests/
│   └── queue_320_sheaf.tsv
├── packages/
│   ├── 320__sheaf__12asA00.tar.gz
│   ├── 320__sheaf__1aa7A02.tar.gz
│   └── 320__sheaf__1aepA00.tar.gz
├── logs/
└── results/
```

For native:

```text
pullback_320_native_test/
├── af3.bin.zst
├── af3_domain_batch_v2.sub
├── run_af3_domain_batch.sh
├── manifests/
│   └── queue_320_native.tsv
├── packages/
│   ├── 320__native__12asA00.tar.gz
│   └── ...
├── logs/
└── results/
```

Prepare:

```bash
mkdir -p logs results
chmod +x run_af3_domain_batch.sh
```

---

## 5. Submit on CHTC

Sheaf:

```bash
condor_submit af3_domain_batch_v2.sub \
  queue_file=manifests/queue_320_sheaf.tsv
```

Native:

```bash
condor_submit af3_domain_batch_v2.sub \
  queue_file=manifests/queue_320_native.tsv
```

One Condor GPU job = one domain.

Expected JSON counts:
- Sheaf: 25 JSONs/domain for 5 sequences × 5 states
- Native: 5 JSONs/domain for 1 sequence × 5 states

Monitor:

```bash
condor_q
tail -f logs/<package_id>.out
```

A healthy Sheaf job prints:

```text
Input JSON count: 25
Template CIF count: 5
```

A healthy native job prints:

```text
Input JSON count: 5
Template CIF count: 5
```

---

## 6. CHTC tree after completion

```text
pullback_320_sheaf_test/
├── logs/
│   ├── 320__sheaf__12asA00.log
│   ├── 320__sheaf__12asA00.out
│   └── 320__sheaf__12asA00.err
└── results/
    ├── af3_results_320__sheaf__12asA00.tar.gz
    ├── af3_results_320__sheaf__1aa7A02.tar.gz
    └── af3_results_320__sheaf__1aepA00.tar.gz
```

Download the `results/*.tar.gz` files back into the corresponding local batch root.

---

## 7. Unpack AF3 results locally

Sheaf:

```bash
python unpack_af3_results.py \
  --results-dir af3_batch_320_sheaf_test/results \
  --out af3_batch_320_sheaf_test/af3_output \
  --skip-existing
```

Native:

```bash
python unpack_af3_results.py \
  --results-dir af3_batch_320_native_test/results \
  --out af3_batch_320_native_test/af3_output \
  --skip-existing
```

Check fold counts:

```bash
find af3_batch_320_sheaf_test/af3_output \
  -mindepth 1 -maxdepth 1 -type d | wc -l
```

Expected for 3-domain tests:
- Sheaf: 75
- Native: 15

---

## 8. Score locally

Sheaf:

```bash
python score_pullback_batch.py \
  --manifest af3_batch_320_sheaf_test/manifests/af3_jobs_320_sheaf.tsv \
  --references-dir af3_batch_320_sheaf_test/references \
  --af3-output-dir af3_batch_320_sheaf_test/af3_output \
  --out af3_batch_320_sheaf_test/scores
```

Native:

```bash
python score_pullback_batch.py \
  --manifest af3_batch_320_native_test/manifests/af3_jobs_320_native.tsv \
  --references-dir af3_batch_320_native_test/references \
  --af3-output-dir af3_batch_320_native_test/af3_output \
  --out af3_batch_320_native_test/scores
```

Output:

```text
scores/
├── all_af3_samples.csv
├── sequence_state_summary.csv
├── sequence_summary.csv
└── domain_summary.csv
```

---

## 9. Native-normalized comparison

After both Sheaf and native are scored, compare matched domain/state rows:

```text
delta_rmsd = rmsd_sheaf - rmsd_native
delta_tm   = tm_sheaf - tm_native
delta_lddt = lddt_sheaf - lddt_native
```

Interpretation:
- lower `delta_rmsd` is better
- higher `delta_tm` is better
- higher `delta_lddt` is better

Do not compare raw Sheaf values across states without the native baseline.

---

## 10. Repeat independently for other categories

Keep categories independent:

```text
af3_batch_320_native/
af3_batch_320_sheaf/
af3_batch_320_mlp/
af3_batch_450_native/
af3_batch_450_sheaf/
af3_batch_450_mlp/
```

A new MLP or 450 K experiment should not require rerunning already completed categories.
