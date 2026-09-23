#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

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
            old = line[17:20].strip().upper()
            new = RESNAME_MAP.get(old, old)
            if new != old:
                line = line[:17] + f"{new:>3}" + line[20:]
            raw = line.rstrip()
        lines.append(raw)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")

def detect_single_protein_chain(pdb_path: Path) -> str:
    model = next(PDBParser(QUIET=True).get_structure("state", pdb_path).get_models())
    chains = [
        str(chain.id) for chain in model.get_chains()
        if any(is_aa(r, standard=False) for r in chain.get_residues())
    ]
    if len(chains) != 1:
        raise ValueError(f"{pdb_path}: expected exactly one protein chain, found {chains}")
    return chains[0]

def prepare_state(state_pdb: Path, domain: str, state_id: str, out_dir: Path):
    normalized = out_dir / f"{domain}__{state_id}__source.pdb"
    normalize_pdb_resnames(state_pdb, normalized)
    chain_id = detect_single_protein_chain(normalized)
    parsed = parse_structure(name=f"{domain}__{state_id}", pdb_path=normalized, chain_id=chain_id)
    target = parsed.target_chain
    mapping = tuple(range(len(target.sequence)))
    prepared = PreparedState(
        name=f"{domain}__{state_id}",
        pdb_path=normalized,
        chain_id=chain_id,
        sequence=target.sequence,
        aligned_sequence=target.sequence,
        residues_3=target.residues_3,
        residue_numbers=target.residue_numbers,
        residue_insertions=target.residue_insertions,
        coords=target.coords,
        query_indices=mapping,
        template_indices=mapping,
        context_chains=parsed.context_chains,
        pyg_data=None,
    )
    artifact = _artifact_from_state(prepared, structure_kind="target", output_dir=out_dir)
    return artifact, target.sequence, mapping

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

def build_domain_package(domain: str, temperature: str, pca_tar: Path, package_dir: Path, reference_dir: Path, job_rows: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"{domain}_") as tmpdir:
        tmp = Path(tmpdir)
        domain_dir = extract_domain(pca_tar, domain, tmp)
        state_pdbs = sorted(domain_dir.glob("state_*.pdb"))
        if not state_pdbs:
            raise FileNotFoundError(f"{domain}: no state_*.pdb found")

        prepared_dir = tmp / "prepared"
        inputs_dir = tmp / "inputs"
        prepared_dir.mkdir()
        inputs_dir.mkdir()

        native_sequence = None
        templates = {}
        for state_pdb in state_pdbs:
            state_id = state_pdb.stem
            state_out = prepared_dir / state_id
            state_out.mkdir()
            artifact, state_sequence, mapping = prepare_state(state_pdb, domain, state_id, state_out)
            if native_sequence is None:
                native_sequence = state_sequence
            elif native_sequence != state_sequence:
                raise ValueError(f"{domain}: native sequence differs across PCA states")

            cif_name = f"{domain}__{state_id}__target.cif"
            pdb_name = f"{domain}__{state_id}__target.pdb"
            shutil.copy2(artifact.mmcif_path, inputs_dir / cif_name)
            ref_domain = reference_dir / domain
            ref_domain.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact.pdb_path, ref_domain / pdb_name)
            templates[state_id] = (inputs_dir / cif_name, mapping, ref_domain / pdb_name)

        jobs = []
        sequence_id = f"{domain}__native"
        for state_id in sorted(templates):
            cif_path, mapping, ref_pdb = templates[state_id]
            job_name = f"{domain}__native__{state_id}__target"
            protein = AF3ProteinSpec(
                entity_id="A",
                sequence=native_sequence,
                template_mmcif=Path(cif_path.name),
                query_indices=mapping,
                template_indices=mapping,
                use_msa=False,
            )
            job = AF3JobSpec(
                job_name=job_name,
                sequence_id=sequence_id,
                state_name=state_id,
                structure_kind="target",
                input_dir=inputs_dir,
                json_path=inputs_dir / f"{job_name}.json",
                output_dir=Path("af3_output") / job_name,
                proteins=(protein,),
                model_seeds=(1,),
                predicted_target_chain_id="A",
            )
            jobs.append(job)
            job_rows.append({
                "job_name": job_name,
                "domain": domain,
                "temperature": temperature,
                "method": "native",
                "design_id": "native",
                "sequence_id": sequence_id,
                "state_id": state_id,
                "sequence": native_sequence,
                "sequence_length": len(native_sequence),
                "reference_pdb": str(ref_pdb),
                "template_cif_name": cif_path.name,
                "query_indices": json.dumps(list(mapping)),
                "template_indices": json.dumps(list(mapping)),
            })

        write_af3_jsons(tuple(jobs))
        package_id = f"{temperature}__native__{domain}"
        package_tar = package_dir / f"{package_id}.tar.gz"
        package_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(package_tar, "w:gz") as tf:
            for path in sorted(inputs_dir.iterdir()):
                tf.add(path, arcname=f"inputs/{path.name}", recursive=False)

        return {
            "package_id": package_id,
            "domain": domain,
            "temperature": temperature,
            "method": "native",
            "n_sequences": 1,
            "n_states": len(state_pdbs),
            "n_af3_jobs": len(jobs),
            "sequence_length": len(native_sequence),
            "total_residue_work": len(native_sequence) * len(jobs),
            "package_tar": str(package_tar),
            "source_pca_tar": str(pca_tar),
        }

def main() -> None:
    parser = argparse.ArgumentParser(description="Build native AF3 pullback packages from PCA state tarballs.")
    parser.add_argument("--pca-tar-dir", type=Path, required=True)
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--domains-file", type=Path)
    parser.add_argument("--max-domains", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    pca_index = index_pca_tarballs(args.pca_tar_dir)
    domains = sorted(pca_index)
    if args.domains_file:
        keep = {x.strip() for x in args.domains_file.read_text().splitlines() if x.strip()}
        domains = [d for d in domains if d in keep]
    if args.max_domains is not None:
        domains = domains[:args.max_domains]

    package_dir = args.out / "packages"
    reference_dir = args.out / "references"
    manifest_dir = args.out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    package_rows, job_rows, failures = [], [], []
    for i, domain in enumerate(domains, 1):
        package_tar = package_dir / f"{args.temperature}__native__{domain}.tar.gz"
        if args.skip_existing and package_tar.exists():
            print(f"[{i}/{len(domains)}] skip {domain}")
            continue
        print(f"[{i}/{len(domains)}] build {domain}")
        try:
            package_rows.append(build_domain_package(
                domain, str(args.temperature), pca_index[domain],
                package_dir, reference_dir, job_rows
            ))
        except Exception as exc:
            failures.append({"domain": domain, "error": repr(exc)})
            print(f"  FAILED: {exc}")

    packages = pd.DataFrame(package_rows)
    jobs = pd.DataFrame(job_rows)
    failures_df = pd.DataFrame(failures)
    packages_path = manifest_dir / f"packages_{args.temperature}_native.tsv"
    jobs_path = manifest_dir / f"af3_jobs_{args.temperature}_native.tsv"
    failures_path = manifest_dir / f"build_failures_{args.temperature}_native.csv"
    packages.to_csv(packages_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    jobs.to_csv(jobs_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    failures_df.to_csv(failures_path, index=False)

    print("\\nBuild complete")
    print(f"Domains requested: {len(domains)}")
    print(f"Packages built:    {len(packages)}")
    print(f"AF3 jobs:          {len(jobs)}")
    print(f"Failures:          {len(failures_df)}")
    print(f"Packages manifest: {packages_path}")
    print(f"AF3 jobs manifest: {jobs_path}")

if __name__ == "__main__":
    main()
