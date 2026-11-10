"""Multi-state (Algorithm 1) sampling for ADFLIP -- the paper's ensemble path, which the
released code does not ship.

Upstream's two samplers (`DiscreteFlow_AA.sample`, `DiscreteFlow_AA.adaptive_sample`) each take
a single `pdb_path` and build one `data` dict from it. The ICML paper's multi-state experiment
(Algorithm 1, Table 3) instead runs the denoiser on *all* N conformations at every flow step and
averages the per-state predicted distributions before stepping:

    p(s_hat_1) = (1/N) sum_n p^n(s_hat_1 | s_t, x_n, chi_t)

then samples one shared sequence from that average and repacks side chains *per conformation*
(chi_1^n = g(s_1, x_n)). States are not cycled one per step -- all of them contribute at every
step. `test/benchmark.py`'s `ensemble_logits` is a different thing: the mean over repeated
stochastic samples of a *single* structure.

This module adds that path without touching the checkout. `adaptive_sample_multi_state` is a
free function over an existing `DiscreteFlow_AA`, reusing its `_prepare_sampling_io`,
`corrupt_data_by_sample` and `sc_packing` unchanged, so the N == 1 case reproduces upstream's
`adaptive_sample` step for step (same tempered step probabilities, same conserve mask, same
`argmax_final` tail, same side-chain repacking order).

Two deliberate differences from upstream, both noted where they occur:

* `sample()` (fixed-dt) has no multi-state counterpart here on purpose -- it never calls
  `sc_packing`, so a residue that has been decided keeps whatever side-chain atoms the input
  file happened to carry. With an all-atom input those are the *native* side chains, which
  leaks the answer into later steps even when the predicted residue type differs; with a
  backbone-only input there are none at all, leaving the model without the side-chain context
  it was trained to see. Either way the adaptive sampler, which rebuilds side chains from its
  own prediction, is the one to score.
* The adaptive step size is `dt = (fraction of positions now confident) - t`, which can be <= 0
  when confidence does not grow; upstream can then spin without advancing. `min_dt` forces
  progress.

Why this is only used through the *returned* averaged log-probabilities: the loop below is a
sampler, so recovery comes from the samples it draws, while perplexity/top-k come from the
averaged distribution at the final step. Both are returned.
"""
import os
import os.path as osp
import sys

import torch
import torch.nn.functional as F

# The checkout lives beside this file; `from data...` / `from model...` need it on sys.path.
ADFLIP_ROOT = osp.join(osp.dirname(osp.abspath(__file__)), 'ADFLIP')
if ADFLIP_ROOT not in sys.path:
    sys.path.insert(0, ADFLIP_ROOT)


def _import_vocabulary():
    """`data.all_atom_parse`'s token table, importable from anywhere.

    That module reads `data/misc/af3_*.txt` with *relative* paths at import time, so it only
    imports when the process cwd is the checkout root (upstream always runs from there --
    `export PYTHONPATH=$PWD`). Rather than force that on importers, chdir for the duration of
    the import and restore. Note this covers the import only: `DiscreteFlow_AA.load_sc_model`
    reads `PIPPack/model_weights/...` relatively at *call* time too, so anything that actually
    samples must run with cwd == ADFLIP_ROOT (eval_pretrained.py chdirs there).
    """
    origin = os.getcwd()
    try:
        os.chdir(ADFLIP_ROOT)
        from data.all_atom_parse import residue_tokens
    finally:
        os.chdir(origin)
    return residue_tokens


residue_tokens = _import_vocabulary()
MASK_TOKEN = residue_tokens['<MASK>']


def _designable(data):
    """Upstream's designable-position mask: centre atoms of protein residues with a complete
    backbone (`parse_mmcif_to_structure_data` sets `backbone_mask` per residue from N/CA/C/O)."""
    mask = data['is_center'].bool() & data['is_protein'].bool()
    if 'backbone_mask' in data:
        mask = mask & data['backbone_mask'].bool()
    return mask


def prepare_states(flow, pdb_paths):
    """One parsed, io-prepared state per conformation.

    Each entry is `{'data', 'pdb_path', 'save_folder'}`. `_prepare_sampling_io` copies the input
    beside a per-stem save folder, so the conformations must have *distinct file stems* or their
    `sc_packing` outputs (`<stem>_0.pdb` under `side_chain_t=<t>/`) collide; that is asserted
    here rather than left to produce a silently wrong ensemble.
    """
    pdb_paths = list(pdb_paths)
    if not pdb_paths:
        raise ValueError('no conformations given')

    stems = [osp.splitext(osp.basename(p))[0] for p in pdb_paths]
    if len(set(stems)) != len(stems):
        raise ValueError(
            'conformations must have distinct file stems, got {!r} -- sc_packing writes '
            '<stem>_0.pdb per state and would overwrite'.format(stems))

    states, device = [], None
    for path in pdb_paths:
        data, resolved_path, save_folder, device = flow._prepare_sampling_io(path)
        states.append({'data': data, 'pdb_path': resolved_path, 'save_folder': save_folder})

    reference = _designable(states[0]['data'])
    reference_tokens = states[0]['data']['residue_token'][reference]
    if reference_tokens.numel() == 0:
        raise ValueError('no designable protein residues (incomplete backbone, or no protein)')

    # Every conformation must present the same residues in the same order, or the averaged
    # distribution would mix positions. True for MD frames of one trajectory; checked anyway.
    for i, state in enumerate(states[1:], start=1):
        mask = _designable(state['data'])
        tokens = state['data']['residue_token'][mask]
        if tokens.shape != reference_tokens.shape:
            raise ValueError(
                'conformation {} has {} designable residues, conformation 0 has {}'.format(
                    i, tokens.shape[0], reference_tokens.shape[0]))
        if not torch.equal(tokens, reference_tokens):
            raise ValueError(
                'conformation {} disagrees with conformation 0 on the native sequence at '
                '{} designable positions'.format(i, int((tokens != reference_tokens).sum())))

    return states, reference_tokens, device


@torch.no_grad()
def adaptive_sample_multi_state(flow, pdb_paths, num_step=8, argmax_final=True, temp=0.1,
                                noise=1.0, threshold=0.9, min_dt=1e-3, verbose=False):
    """ADFLIP adaptive sampling conditioned on N conformations at once (Algorithm 1).

    :param flow: a `DiscreteFlow_AA` (upstream wraps it in EMA -- pass `ema_flow.ema_model`).
        Side-chain repacking happens only if it was built with `sidechain_packing=True`.
    :param pdb_paths: the conformations. One path reproduces upstream `adaptive_sample`.
    :param num_step, argmax_final, temp, noise, threshold: upstream's meanings and `test/design.py`
        defaults (`num_step=8`, `threshold=0.9`, `argmax_final=True`).
    :param min_dt: floor on the adaptive step, so a step that gains no confidence still advances.
    :returns: `(samples, mean_log_probs, true_tokens)` -- `(1, T)` sampled tokens, the `(T, V)`
        log of the state-averaged final distribution, and the `(T,)` native tokens at those same
        designable positions. `log` of a normalised distribution is already a log-softmax, so
        `F.cross_entropy(mean_log_probs, targets)` is exactly the mean NLL.
    """
    flow.model.eval()
    states, true_tokens, device = prepare_states(flow, pdb_paths)
    num_states = len(states)

    samples = torch.full_like(true_tokens, MASK_TOKEN, device=device).unsqueeze(0)
    B, T = samples.size()

    t = 0.0
    sample_times = 0

    while t <= 1.0:
        # --- Per-state prediction, then average (Algorithm 1 steps 1-2) ---
        # `probs` drives the final argmax and the returned log-probs (temperature 1, as
        # upstream's `argmax_final` branch reads raw logits); `probs_temp` drives the flow step
        # and the confidence threshold (upstream tempers with `temp` for exactly those).
        probs = None
        probs_temp = None
        for state in states:
            state['data'], noisy = flow.corrupt_data_by_sample(state['data'], t, samples)
            logits, _ = flow.model(noisy, torch.tensor([[t]], device=device))
            logits = logits.view(B, T, -1)
            step = F.softmax(logits, dim=-1)
            step_temp = F.softmax(logits / temp, dim=-1)
            probs = step if probs is None else probs + step
            probs_temp = step_temp if probs_temp is None else probs_temp + step_temp
        probs = probs / num_states
        probs_temp = probs_temp / num_states
        V = probs.size(-1)

        if verbose:
            recovery = (probs.view(T, V).argmax(dim=-1) == true_tokens).float().mean().item()
            print('time: {:.3f}  states: {}  argmax recovery: {:.4f}'.format(
                round(t, 3), num_states, recovery))

        # --- Final step: fill whatever is still masked, matching upstream's tail ---
        if round(t, 3) >= 1.0 or sample_times >= num_step:
            if argmax_final:
                final = probs.argmax(dim=-1).view(B, T)
            else:
                final = torch.multinomial(probs.view(B * T, V), num_samples=1).view(B, T)
            still_masked = samples == MASK_TOKEN
            samples[still_masked] = final[still_masked]
            mean_log_probs = torch.log(probs.view(T, V).clamp_min(torch.finfo(probs.dtype).tiny))
            if verbose:
                print('final recovery: {:.4f}'.format(
                    (samples.view(T) == true_tokens).float().mean().item()))
            return samples, mean_log_probs, true_tokens

        # --- Adaptive step size from how much of the sequence is now confident ---
        conserve_mask = (probs_temp.max(dim=-1)[0] > threshold).view(B, T)
        next_t = (conserve_mask.sum() / conserve_mask.shape[1]).item()
        dt = max(next_t - t, min_dt)
        conserve_sample = probs_temp.argmax(dim=-1).view(B, T)

        # --- Forward transition: masked positions take probability of each token ---
        sample_is_mask = (samples == MASK_TOKEN).view(B, T, 1).float()
        step_probs = dt * probs_temp * ((1 + noise * t) / (1 - t))
        step_probs = step_probs * sample_is_mask

        # --- Backward noise: unmasked positions take probability of re-masking ---
        mask_onehot = F.one_hot(torch.tensor(MASK_TOKEN), num_classes=V).view(1, 1, -1).to(device)
        step_probs = step_probs + dt * (1 - sample_is_mask) * mask_onehot * noise

        # --- Normalise: the remaining mass stays on the current token ---
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        rows = torch.arange(B, device=device).repeat_interleave(T)
        cols = torch.arange(T, device=device).repeat(B)
        step_probs[rows, cols, samples.flatten()] = 0.0
        step_probs[rows, cols, samples.flatten()] = 1.0 - step_probs.sum(dim=-1).flatten()
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

        # Never draw <MASK> here: re-masking is applied explicitly below, after repacking.
        no_mask_prob = step_probs.clone()
        no_mask_prob[:, :, MASK_TOKEN] = 0
        degenerate = no_mask_prob.sum(dim=-1) <= 0
        if degenerate.any():
            # Upstream falls back to the raw tempered distribution here, which still carries
            # <MASK> mass -- a position that then draws <MASK> reaches `sc_packing`, whose
            # three-to-one letter lookup has no entry for it and raises KeyError. Drop <MASK>
            # from the fallback and renormalise so the draw stays a real residue.
            fallback = probs_temp[degenerate].to(no_mask_prob.dtype).clone()
            fallback[:, MASK_TOKEN] = 0
            fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(fallback.dtype).tiny)
            no_mask_prob[degenerate] = fallback

        samples = torch.multinomial(no_mask_prob.view(-1, V), num_samples=1).view(B, T)
        samples = samples * ~conserve_mask + conserve_sample * conserve_mask

        # --- Repack side chains per conformation (chi_1^n = g(s_1, x_n)) ---
        # Order matters and matches upstream: `samples` holds a complete sequence here (no
        # <MASK>, which `sc_packing`'s 3->1 letter lookup could not decode), and re-masking of
        # the non-conserved positions happens only afterwards. The next iteration's
        # `corrupt_data_by_sample` then strips the side chains of whatever is masked again, so
        # only the conserved residues' repacked side chains survive as context.
        if hasattr(flow, 'sc'):
            for state in states:
                state['data'] = flow.sc_packing(
                    samples, state['pdb_path'], t, state['save_folder'],
                    models=flow.sc, infer_cfg=flow.infer_cfg, device=device)

        samples[~conserve_mask] = MASK_TOKEN
        # Clamped, so a step that overshoots lands exactly on t=1.0 and the next iteration takes
        # the final branch. Without this the `while` guard would fail first and the call would
        # return nothing, losing the protein.
        t = min(t + dt, 1.0)
        sample_times += 1

    raise RuntimeError('sampling loop exited without returning (t={})'.format(t))
