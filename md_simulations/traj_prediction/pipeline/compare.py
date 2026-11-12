"""Compare the dynamics of two sets of backbone trajectories.

All trajectories must share the backbone topology from common.to_backbone
(N, CA, C, O per residue, same number of residues). A "group" is a list of
Features objects, one per trajectory (for example the replicas of one sequence).

Four families of metrics, all computed by compare_groups():

  distributions  phi/psi Jensen-Shannon divergence and circular Wasserstein
                 distance, CA-CA distance Wasserstein distance, contact
                 frequency differences. No frame matching.
  states         Shared conformational states (see fit_states); occupancies and
                 transition probabilities at one lag time.
  fluctuations   Per-residue RMSF and dynamic cross-correlation maps (DCCM).
  structure      RMSD between mean structures, mean distance-map difference.

Lengths are in Angstrom and angles in degrees. Jensen-Shannon divergences use
log base 2, so they lie between 0 (same distribution) and 1 (no overlap).
"""
from dataclasses import dataclass, field

import mdtraj as md
import numpy as np
from scipy.stats import pearsonr, spearmanr

CONTACT_A = 8.0          # CA-CA distance that counts as a contact
MIN_SEPARATION = 3       # ignore residue pairs closer than this in sequence
RAMA_BINS = 12           # 30 degree bins for the Ramachandran histograms
ANGLE_BINS = 72          # 5 degree bins for the circular Wasserstein distance


# --------------------------------------------------------------------------- features

@dataclass
class Features:
    """Everything the metrics need from one trajectory."""
    name: str
    n_frames: int
    n_residues: int
    phi: np.ndarray            # (T, R-2) degrees, residues 1 .. R-2
    psi: np.ndarray            # (T, R-2) degrees, residues 1 .. R-2
    pairs: np.ndarray          # (P, 2) residue pairs with |i - j| >= MIN_SEPARATION
    dist: np.ndarray           # (T, P) CA-CA distances
    ca: np.ndarray             # (T, R, 3) CA coordinates fitted on the mean structure
    rmsf: np.ndarray           # (R,)
    dccm: np.ndarray           # (R, R)
    labels: np.ndarray = field(default=None)   # (T,) state index, set by fit_states


def _fit_to_mean(traj, ca_idx, rounds=3):
    """Superpose all frames on their own mean CA structure. Returns CA xyz in Angstrom."""
    t = md.Trajectory(traj.xyz[:, ca_idx].copy(), traj.topology.subset(ca_idx))
    ref = t[0]
    for _ in range(rounds):
        t.superpose(ref)
        ref = md.Trajectory(t.xyz.mean(axis=0, keepdims=True), t.topology)
    return t.xyz * 10.0


def featurize(traj, name=""):
    top = traj.topology
    n_res = top.n_residues
    ca_idx = top.select("name CA")
    _, phi = md.compute_phi(traj)          # residues 1 .. R-1
    _, psi = md.compute_psi(traj)          # residues 0 .. R-2
    i, j = np.triu_indices(n_res, k=MIN_SEPARATION)
    pairs = np.stack([i, j], axis=1)
    dist = md.compute_distances(traj, ca_idx[pairs]) * 10.0
    ca = _fit_to_mean(traj, ca_idx)
    disp = ca - ca.mean(axis=0)
    cov = np.einsum("tix,tjx->ij", disp, disp) / len(disp)
    var = np.diag(cov)
    norm = np.sqrt(np.outer(var, var))
    dccm = np.divide(cov, norm, out=np.zeros_like(cov), where=norm > 0)
    return Features(name=name, n_frames=traj.n_frames, n_residues=n_res,
                    phi=np.degrees(phi[:, :-1]), psi=np.degrees(psi[:, 1:]),
                    pairs=pairs, dist=dist, ca=ca, rmsf=np.sqrt(var), dccm=dccm)


def pooled(group, attr):
    return np.concatenate([getattr(f, attr) for f in group], axis=0)


# --------------------------------------------------------------------------- distributions

def js_divergence(p, q):
    """Jensen-Shannon divergence (base 2) between the rows of two count arrays."""
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)

    def kl(a):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a > 0, a * np.log2(a / m), 0.0).sum(axis=-1)
    return np.clip(0.5 * kl(p) + 0.5 * kl(q), 0.0, 1.0)


def _angle_bin(x, bins):
    return np.clip(((x + 180.0) / 360.0 * bins).astype(int), 0, bins - 1)


def rama_counts(phi, psi, bins=RAMA_BINS):
    """(R-2, bins*bins) Ramachandran histogram counts per residue."""
    flat = _angle_bin(phi, bins) * bins + _angle_bin(psi, bins)
    return np.stack([np.bincount(flat[:, r], minlength=bins * bins)
                     for r in range(flat.shape[1])]).astype(float)


def circular_wasserstein(a, b, bins=ANGLE_BINS):
    """Wasserstein-1 distance on the circle, per column, in degrees.

    On a circle the optimal transport cost is the integral of |F - G - c| where
    F, G are the cumulative histograms and c is the median of F - G.
    """
    out = np.empty(a.shape[1])
    for r in range(a.shape[1]):
        p = np.bincount(_angle_bin(a[:, r], bins), minlength=bins) / a.shape[0]
        q = np.bincount(_angle_bin(b[:, r], bins), minlength=bins) / b.shape[0]
        d = np.cumsum(p - q)
        out[r] = np.abs(d - np.median(d)).sum() * (360.0 / bins)
    return out


def wasserstein_columns(a, b, n_quantiles=101):
    """Wasserstein-1 distance per column, from the two quantile functions."""
    q = np.linspace(0.0, 1.0, n_quantiles)
    return np.abs(np.quantile(a, q, axis=0) - np.quantile(b, q, axis=0)).mean(axis=0)


def per_residue_from_pairs(values, pairs, n_res):
    """Mean of a per-pair quantity over all pairs that contain each residue."""
    total, count = np.zeros(n_res), np.zeros(n_res)
    for col in (0, 1):
        np.add.at(total, pairs[:, col], values)
        np.add.at(count, pairs[:, col], 1)
    return np.divide(total, count, out=np.zeros(n_res), where=count > 0)


def distribution_metrics(a, b):
    n_res, pairs = a[0].n_residues, a[0].pairs
    phi_a, psi_a, phi_b, psi_b = (pooled(g, k) for g in (a, b) for k in ("phi", "psi"))
    rama = js_divergence(rama_counts(phi_a, psi_a), rama_counts(phi_b, psi_b))
    w_phi, w_psi = circular_wasserstein(phi_a, phi_b), circular_wasserstein(psi_a, psi_b)
    da, db = pooled(a, "dist"), pooled(b, "dist")
    w_dist = wasserstein_columns(da, db)
    contact = (da < CONTACT_A).mean(axis=0) - (db < CONTACT_A).mean(axis=0)
    pad = lambda x: np.concatenate([[np.nan], x, [np.nan]])     # residues 0 and R-1 have no phi/psi pair
    scalars = dict(rama_jsd_mean=float(rama.mean()), rama_jsd_max=float(rama.max()),
                   phi_wasserstein_deg_mean=float(w_phi.mean()),
                   psi_wasserstein_deg_mean=float(w_psi.mean()),
                   ca_dist_wasserstein_A_mean=float(w_dist.mean()),
                   ca_dist_wasserstein_A_max=float(w_dist.max()),
                   contact_freq_absdiff_mean=float(np.abs(contact).mean()),
                   contact_freq_absdiff_max=float(np.abs(contact).max()))
    arrays = dict(rama_jsd=pad(rama), phi_wasserstein_deg=pad(w_phi),
                  psi_wasserstein_deg=pad(w_psi),
                  ca_dist_wasserstein_A=per_residue_from_pairs(w_dist, pairs, n_res),
                  contact_freq_diff=_square(contact, pairs, n_res))
    return scalars, arrays


def _square(values, pairs, n_res):
    m = np.zeros((n_res, n_res))
    m[pairs[:, 0], pairs[:, 1]] = values
    return m + m.T


# --------------------------------------------------------------------------- states

def fit_states(all_features, lag=1, n_pca=10, n_tica=3, k=5, max_pairs=2000, seed=0):
    """Define one set of conformational states for all trajectories and label every frame.

    Features: sin/cos of phi and psi, and CA-CA distances. They are z-scored and
    reduced with PCA, then with TICA (the slowest linear combinations at the
    given lag) when every trajectory is longer than the lag. k-means on the
    result gives the states; the number of states (2 .. k) is the one with the
    best silhouette score. The fit pools all trajectories, so the states mean
    the same thing in each of them.

    Returns (projection per trajectory, k).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    step = max(1, all_features[0].dist.shape[1] // max_pairs)
    blocks = []
    for f in all_features:
        ang = np.radians(np.concatenate([f.phi, f.psi], axis=1))
        blocks.append(np.concatenate([np.sin(ang), np.cos(ang), f.dist[:, ::step]], axis=1))
    x = np.concatenate(blocks)
    sd = x.std(axis=0)
    x = (x[:, sd > 1e-8] - x[:, sd > 1e-8].mean(axis=0)) / sd[sd > 1e-8]

    n_comp = max(1, min(n_pca, x.shape[0] - 1, x.shape[1]))
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    y = u[:, :n_comp] * np.sqrt(x.shape[0])            # whitened PCA coordinates
    bounds = np.cumsum([0] + [f.n_frames for f in all_features])
    parts = [y[bounds[i]:bounds[i + 1]] for i in range(len(all_features))]

    if all(len(part) > lag for part in parts):
        c = sum(part[:-lag].T @ part[lag:] for part in parts)
        c = 0.5 * (c + c.T) / sum(len(part) - lag for part in parts)
        w, v = np.linalg.eigh(c)
        v = v[:, np.argsort(w)[::-1][:min(n_tica, n_comp)]]
        parts = [part @ v for part in parts]
    else:
        parts = [part[:, :min(n_tica, n_comp)] for part in parts]

    proj = np.concatenate(parts)
    # Try 2 .. k states and keep the clustering with the best silhouette score.
    k_max = int(max(2, min(k, len(proj) // 20)))
    best = None
    for n in range(2, k_max + 1):
        trial = KMeans(n_clusters=n, n_init=10, random_state=seed).fit_predict(proj)
        score = silhouette_score(proj, trial) if len(set(trial)) > 1 else -1.0
        if best is None or score > best[0]:
            best = (score, n, trial)
    _, k, labels = best
    # Number the states by total population, largest first.
    order = np.argsort(-np.bincount(labels, minlength=k))
    labels = np.argsort(order)[labels]
    for i, f in enumerate(all_features):
        f.labels = labels[bounds[i]:bounds[i + 1]]
    return parts, k


def occupancy(group, k):
    return np.bincount(pooled(group, "labels"), minlength=k) / sum(f.n_frames for f in group)


def transition_counts(group, k, lag=1):
    c = np.zeros((k, k))
    for f in group:
        np.add.at(c, (f.labels[:-lag], f.labels[lag:]), 1)
    return c


def implied_timescales(group, k, lags, frame_ns):
    """Two slowest implied timescales (ns) at each lag, from the row-normalised count matrix."""
    out = {}
    for lag in lags:
        c = transition_counts(group, k, lag)
        seen = c.sum(axis=1) > 0
        c = c[np.ix_(seen, seen)]
        rows = c.sum(axis=1, keepdims=True)
        if len(c) < 2 or (rows == 0).any():
            out[str(lag)] = [None, None]
            continue
        ev = np.sort(np.abs(np.linalg.eigvals(c / rows)))[::-1][1:3]
        ts = [float(-lag * frame_ns / np.log(e)) if 0.0 < e < 1.0 else None for e in ev]
        out[str(lag)] = (ts + [None, None])[:2]
    return out


def state_summary(group, k, frame_ns, lag=1):
    """Occupancy and kinetics of one group."""
    n_frames = min(f.n_frames for f in group)
    switches = sum(int((f.labels[1:] != f.labels[:-1]).sum()) for f in group)
    steps = sum(f.n_frames - 1 for f in group)
    lags = [l for l in (1, 2, 5, 10) if l < n_frames / 3]
    return dict(occupancy=occupancy(group, k).tolist(),
                transition_counts=transition_counts(group, k, lag).tolist(),
                state_switches_per_ns=switches / (steps * frame_ns) if steps else None,
                implied_timescales_ns_by_lag=implied_timescales(group, k, lags, frame_ns))


def state_metrics(a, b, k, lag=1):
    occ_a, occ_b = occupancy(a, k), occupancy(b, k)
    ca, cb = transition_counts(a, k, lag), transition_counts(b, k, lag)
    both = (ca.sum(axis=1) > 0) & (cb.sum(axis=1) > 0)
    scalars = dict(state_occupancy_jsd=float(js_divergence(occ_a, occ_b)),
                   states_visited_by_both=int(both.sum()))
    if both.any():
        ta = ca[both] / ca[both].sum(axis=1, keepdims=True)
        tb = cb[both] / cb[both].sum(axis=1, keepdims=True)
        weight = (occ_a + occ_b)[both]
        scalars["transition_row_jsd"] = float(
            (js_divergence(ta, tb) * weight).sum() / weight.sum())
        scalars["transition_stay_prob_absdiff"] = float(
            (np.abs(np.diag(ca)[both] / ca[both].sum(axis=1)
                    - np.diag(cb)[both] / cb[both].sum(axis=1)) * weight).sum() / weight.sum())
    else:
        scalars["transition_row_jsd"] = None
        scalars["transition_stay_prob_absdiff"] = None
    return scalars


# --------------------------------------------------------------------------- fluctuations, structure

def _corr(x, y, fn=pearsonr):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(fn(x, y)[0])


def group_mean(group, attr):
    return np.mean([getattr(f, attr) for f in group], axis=0)


def fluctuation_metrics(a, b):
    ra, rb = group_mean(a, "rmsf"), group_mean(b, "rmsf")
    da, db = group_mean(a, "dccm"), group_mean(b, "dccm")
    iu = np.triu_indices_from(da, k=1)
    scalars = dict(rmsf_pearson=_corr(ra, rb), rmsf_spearman=_corr(ra, rb, spearmanr),
                   rmsf_absdiff_A_mean=float(np.abs(ra - rb).mean()),
                   rmsf_ratio_mean=float(ra.mean() / rb.mean()) if rb.mean() > 0 else None,
                   dccm_pearson=_corr(da[iu], db[iu]),
                   dccm_rms_diff=float(np.sqrt(((da - db)[iu] ** 2).mean())))
    return scalars, dict(rmsf_diff_A=ra - rb, dccm_diff=da - db)


def mean_structure(group):
    """Mean CA structure (R, 3) of a group, after fitting the per-trajectory means together."""
    means = np.stack([f.ca.mean(axis=0) for f in group])
    top = md.Topology()
    chain = top.add_chain()
    for _ in range(means.shape[1]):
        top.add_atom("CA", md.element.carbon, top.add_residue("GLY", chain))
    t = md.Trajectory(means / 10.0, top)
    t.superpose(t, 0)
    return t.xyz.mean(axis=0) * 10.0


def kabsch_rmsd(x, y):
    """RMSD in the units of x, y after optimal superposition. x, y have shape (N, 3)."""
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    h = x.T @ y
    s = np.linalg.svd(h, compute_uv=False)
    s[2] *= np.sign(np.linalg.det(h))
    return float(np.sqrt(max(((x ** 2).sum() + (y ** 2).sum() - 2.0 * s.sum()) / len(x), 0.0)))


def structure_metrics(a, b):
    pairs, n_res = a[0].pairs, a[0].n_residues
    dmap = pooled(a, "dist").mean(axis=0) - pooled(b, "dist").mean(axis=0)
    scalars = dict(mean_structure_rmsd_A=kabsch_rmsd(mean_structure(a), mean_structure(b)),
                   mean_dist_map_absdiff_A_mean=float(np.abs(dmap).mean()),
                   mean_dist_map_absdiff_A_max=float(np.abs(dmap).max()))
    return scalars, dict(mean_dist_map_diff_A=_square(dmap, pairs, n_res))


def rmsd_series(traj, reference):
    """CA RMSD (A) of every frame of traj to the single-frame reference."""
    ca = traj.topology.select("name CA")
    return md.rmsd(traj, reference, atom_indices=ca) * 10.0


# --------------------------------------------------------------------------- top level

def compare_groups(a, b, k, lag=1):
    """All metrics between group a and group b. Returns (scalars, arrays)."""
    scalars, arrays = {}, {}
    for fn in (distribution_metrics, fluctuation_metrics, structure_metrics):
        s, arr = fn(a, b)
        scalars.update(s)
        arrays.update(arr)
    scalars.update(state_metrics(a, b, k, lag))
    return scalars, arrays


def summarise(list_of_scalars):
    """Mean, standard deviation and values of each metric over several comparisons."""
    out = {}
    for key in list_of_scalars[0]:
        vals = [s[key] for s in list_of_scalars if s[key] is not None]
        out[key] = dict(mean=float(np.mean(vals)) if vals else None,
                        std=float(np.std(vals)) if vals else None, n=len(vals), values=vals)
    return out
