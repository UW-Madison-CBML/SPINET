#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, shutil, tarfile, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from dynamicmpnn.eval.af3 import write_af3_jsons
from dynamicmpnn.eval.inputs import PreparedState, parse_structure
from dynamicmpnn.eval.templates import _artifact_from_state
from dynamicmpnn.eval.types import AF3JobSpec, AF3ProteinSpec

RESNAME_MAP = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS"}


def normalize_pdb_resnames(src: Path, dst: Path) -> None:
    lines = []
    for raw in src.read_text(encoding="utf-8").splitlines():
        if raw.startswith(("ATOM  ", "HETATM")):
            line = raw.ljust(80)
            resname = line[17:20].strip().upper()
            canonical = RESNAME_MAP.get(resname, resname)
            if canonical != resname:
                line = line[:17] + f"{canonical:>3}" + line[20:]
            raw = line.rstrip()
        lines.append(raw)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def detect_single_protein_chain(pdb_path: Path) -> str:
    model = next(PDBParser(QUIET=True).get_structure("state", pdb_path).get_models())
    chains = [str(chain.id) for chain in model.get_chains() if any(is_aa(r, standard=False) for r in chain.get_residues())]
    if len(chains) != 1:
        raise ValueError(f"{pdb_path}: expected one protein chain, found {chains}")
    return chains[0]


def prepare_template(state_pdb: Path, domain: str, state_id: str, out_dir: Path):
    normalized = out_dir / f"{domain}__{state_id}__source.pdb"
    normalize_pdb_resnames(state_pdb, normalized)
    chain_id = detect_single_protein_chain(normalized)
    parsed = parse_structure(name=f"{domain}__{state_id}", pdb_path=normalized, chain_id=chain_id)
    target = parsed.target_chain
    mapping = tuple(range(len(target.sequence)))
    prepared = PreparedState(
        name=f"{domain}__{state_id}", pdb_path=normalized, chain_id=chain_id,
        sequence=target.sequence, aligned_sequence=target.sequence,
        residues_3=target.residues_3, residue_numbers=target.residue_numbers,
        residue_insertions=target.residue_insertions, coords=target.coords,
        query_indices=mapping, template_indices=mapping,
        context_chains=parsed.context_chains, pyg_data=None,
    )
    artifact = _artifact_from_state(prepared, structure_kind="target", output_dir=out_dir)
    return artifact, target.sequence, mapping


def load_selected_sequences(path: Path, method: str, temperature: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"domain", "method", "design_id", "sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df = df[df["method"].astype(str) == method].copy()
    if "temperature" in df.columns:
        df = df[df["temperature"].astype(str) == temperature].copy()
    if df.empty:
        raise ValueError(f"No rows for method={method}, temperature={temperature}")
    return df


def requested_domains(df: pd.DataFrame, domains_file: Path | None, max_domains: int | None) -> list[str]:
    domains = sorted(df["domain"].astype(str).unique())
    if domains_file:
        requested = {x.strip() for x in domains_file.read_text().splitlines() if x.strip()}
        domains = [d for d in domains if d in requested]
    if max_domains is not None:
        domains = domains[:max_domains]
    return domains


def index_pca_tarballs(pca_tar_dir: Path) -> dict[str, Path]:
    mapping = {}
    for tar_path in sorted(pca_tar_dir.glob("*.tar.gz")):
        with tarfile.open(tar_path, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.isdir():
                    continue
                parts = Path(member.name.rstrip("/")).parts
                if len(parts) == 2 and parts[0].startswith("pca_states"):
                    domain = parts[1]
                    if domain in mapping and mapping[domain] != tar_path:
                        raise ValueError(f"Domain {domain} appears in multiple PCA tarballs")
                    mapping[domain] = tar_path
    return mapping


def extract_domain(tar_path: Path, domain: str, work_dir: Path) -> Path:
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if domain in Path(m.name).parts]
        tf.extractall(work_dir, members=members)
    matches = list(work_dir.glob(f"pca_states*/{domain}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not uniquely extract {domain} from {tar_path}")
    return matches[0]


def build_domain_package(domain: str, temperature: str, method: str, seq_rows: pd.DataFrame, pca_tar: Path,
                         package_dir: Path, reference_dir: Path, detailed_rows: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"{domain}_") as tmp_name:
        tmp = Path(tmp_name)
        domain_dir = extract_domain(pca_tar, domain, tmp)
        state_pdbs = sorted(domain_dir.glob("state_*.pdb"))
        state_manifest = domain_dir / "state_manifest.csv"
        if not state_pdbs or not state_manifest.exists():
            raise FileNotFoundError(f"{domain}: missing state PDBs or state_manifest.csv")

        prep_dir, inputs_dir = tmp / "prepared", tmp / "inputs"
        prep_dir.mkdir(); inputs_dir.mkdir()
        templates = {}
        native_sequence = None

        for state_pdb in state_pdbs:
            state_id = state_pdb.stem
            state_out = prep_dir / state_id
            state_out.mkdir()
            artifact, state_sequence, mapping = prepare_template(state_pdb, domain, state_id, state_out)
            if native_sequence is None:
                native_sequence = state_sequence
            elif native_sequence != state_sequence:
                raise ValueError(f"{domain}: native sequence differs across states")
            cif_name = f"{domain}__{state_id}__target.cif"
            pdb_name = f"{domain}__{state_id}__target.pdb"
            shutil.copy2(artifact.mmcif_path, inputs_dir / cif_name)
            ref_dir = reference_dir / domain
            ref_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact.pdb_path, ref_dir / pdb_name)
            templates[state_id] = {"cif": inputs_dir / cif_name, "reference_pdb": ref_dir / pdb_name, "mapping": mapping}

        jobs = []
        for _, row in seq_rows.sort_values("design_id").iterrows():
            sequence = str(row["sequence"]).strip().upper()
            design_id = str(row["design_id"])
            if len(sequence) != len(native_sequence):
                raise ValueError(f"{domain}/{design_id}: sequence length {len(sequence)} != native length {len(native_sequence)}")
            sequence_id = f"{domain}__{design_id}"
            for state_id in sorted(templates):
                template = templates[state_id]
                job_name = f"{domain}__{design_id}__{state_id}__target"
                protein = AF3ProteinSpec(
                    entity_id="A", sequence=sequence, template_mmcif=Path(template["cif"].name),
                    query_indices=template["mapping"], template_indices=template["mapping"], use_msa=False,
                )
                job = AF3JobSpec(
                    job_name=job_name, sequence_id=sequence_id, state_name=state_id, structure_kind="target",
                    input_dir=inputs_dir, json_path=inputs_dir / f"{job_name}.json",
                    output_dir=Path("af3_output") / job_name, proteins=(protein,), model_seeds=(1,), predicted_target_chain_id="A",
                )
                jobs.append(job)
                detailed_rows.append({
                    "job_name": job_name, "domain": domain, "temperature": temperature, "method": method,
                    "design_id": design_id, "sequence_id": sequence_id, "state_id": state_id, "sequence": sequence,
                    "sequence_length": len(sequence), "reference_pdb": str(template["reference_pdb"].relative_to(reference_dir.parent)),
                    "template_cif_name": template["cif"].name,
                    "query_indices": json.dumps(list(template["mapping"])),
                    "template_indices": json.dumps(list(template["mapping"])),
                })

        write_af3_jsons(tuple(jobs))
        package_id = f"{temperature}__{method}__{domain}"
        package_dir.mkdir(parents=True, exist_ok=True)
        package_tar = package_dir / f"{package_id}.tar.gz"
        with tarfile.open(package_tar, "w:gz") as tf:
            for path in sorted(inputs_dir.iterdir()):
                tf.add(path, arcname=f"inputs/{path.name}", recursive=False)

        return {
            "package_id": package_id, "domain": domain, "temperature": temperature, "method": method,
            "n_sequences": int(seq_rows["design_id"].nunique()), "n_states": len(state_pdbs), "n_af3_jobs": len(jobs),
            "sequence_length": len(native_sequence), "total_residue_work": len(jobs) * len(native_sequence),
            "package_tar": str(package_tar.relative_to(package_dir.parent)), "source_pca_tar": str(pca_tar),
        }


def main() -> None:
    p = argparse.ArgumentParser(description="Build one AF3 domain package from PCA states and selected sequences.")
    p.add_argument("--pca-tar-dir", type=Path, required=True)
    p.add_argument("--sequences", type=Path, required=True)
    p.add_argument("--temperature", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--domains-file", type=Path)
    p.add_argument("--max-domains", type=int)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    seq_df = load_selected_sequences(args.sequences, args.method, str(args.temperature))
    domains = requested_domains(seq_df, args.domains_file, args.max_domains)
    pca_index = index_pca_tarballs(args.pca_tar_dir)
    missing = [d for d in domains if d not in pca_index]
    if missing:
        raise FileNotFoundError(f"{len(missing)} requested domains have no PCA result; first: {missing[:10]}")

    package_dir = args.out / "packages"
    reference_dir = args.out / "references"
    manifest_dir = args.out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    package_rows, detailed_rows, failures = [], [], []

    for i, domain in enumerate(domains, 1):
        package_tar = package_dir / f"{args.temperature}__{args.method}__{domain}.tar.gz"
        if args.skip_existing and package_tar.exists():
            print(f"[{i}/{len(domains)}] skip {domain}")
            continue
        print(f"[{i}/{len(domains)}] build {domain}")
        try:
            rows = seq_df[seq_df["domain"].astype(str) == domain].copy()
            package_rows.append(build_domain_package(domain, str(args.temperature), args.method, rows, pca_index[domain], package_dir, reference_dir, detailed_rows))
        except Exception as exc:
            failures.append({"domain": domain, "error": repr(exc)})
            print(f"  FAILED: {exc}")

    packages = pd.DataFrame(package_rows)
    details = pd.DataFrame(detailed_rows)
    failures_df = pd.DataFrame(failures)
    packages_path = manifest_dir / f"packages_{args.temperature}_{args.method}.tsv"
    details_path = manifest_dir / f"af3_jobs_{args.temperature}_{args.method}.tsv"
    failures_path = manifest_dir / f"build_failures_{args.temperature}_{args.method}.csv"
    packages.to_csv(packages_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    details.to_csv(details_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    failures_df.to_csv(failures_path, index=False)
    print(f"\nBuilt {len(packages)} packages, {len(details)} AF3 folds, {len(failures_df)} failures")
    print(packages_path)
    print(details_path)
    print(failures_path)


if __name__ == "__main__":
    main()
