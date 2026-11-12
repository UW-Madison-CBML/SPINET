"""Tests for compare.py on synthetic backbone trajectories. Run: pytest tests/"""
import sys
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare as cmp                                  # noqa: E402
from common import backbone_topology, to_backbone      # noqa: E402


def helix(n_res):
    """Rough alpha-helix backbone, (n_res * 4, 3) in nm, atom order N, CA, C, O."""
    radius = {"N": 0.157, "CA": 0.228, "C": 0.166, "O": 0.194}
    phase = {"N": -27.0, "CA": 0.0, "C": 28.0, "O": 33.0}
    rise = {"N": -0.087, "CA": 0.0, "C": 0.105, "O": 0.225}
    xyz = []
    for i in range(n_res):
        for a in ("N", "CA", "C", "O"):
            ang = np.radians(100.0 * i + phase[a])
            xyz.append([radius[a] * np.cos(ang), radius[a] * np.sin(ang), 0.15 * i + rise[a]])
    return np.array(xyz)


def fake_traj(n_frames=120, n_res=24, seed=0, dwell=30, noise=0.01):
    """Helix that switches between a straight and a bent state every `dwell` frames."""
    rng = np.random.default_rng(seed)
    base = helix(n_res)
    hinge = n_res // 2 * 4
    pivot = base[hinge]
    bend = np.radians(50.0)
    rot = np.array([[1, 0, 0], [0, np.cos(bend), -np.sin(bend)], [0, np.sin(bend), np.cos(bend)]])
    bent = base.copy()
    bent[hinge:] = (base[hinge:] - pivot) @ rot.T + pivot
    state = (np.arange(n_frames) // dwell) % 2
    xyz = np.where(state[:, None, None] == 1, bent, base) + rng.normal(0, noise, (n_frames, len(base), 3))
    return md.Trajectory(xyz.astype(np.float32), backbone_topology(["ALA"] * n_res)), state


def features(traj, name="t"):
    return cmp.featurize(traj, name)


def test_same_trajectory_gives_zero_difference():
    traj, _ = fake_traj()
    a, b = features(traj, "a"), features(traj, "b")
    _, k = cmp.fit_states([a, b])
    s, _ = cmp.compare_groups([a], [b], k)
    for key in ("rama_jsd_mean", "phi_wasserstein_deg_mean", "ca_dist_wasserstein_A_mean",
                "contact_freq_absdiff_mean", "state_occupancy_jsd", "transition_row_jsd",
                "rmsf_absdiff_A_mean", "dccm_rms_diff", "mean_dist_map_absdiff_A_mean"):
        assert s[key] == pytest.approx(0.0, abs=1e-6), key
    assert s["mean_structure_rmsd_A"] == pytest.approx(0.0, abs=1e-3)
    assert s["rmsf_pearson"] == pytest.approx(1.0)
    assert s["dccm_pearson"] == pytest.approx(1.0)


def test_shuffled_frames_keep_distributions_but_change_transitions():
    traj, state = fake_traj()
    order = np.random.default_rng(1).permutation(traj.n_frames)
    a, b = features(traj, "a"), features(traj[order], "b")
    _, k = cmp.fit_states([a, b], k=2)
    s, _ = cmp.compare_groups([a], [b], k)
    assert s["rama_jsd_mean"] == pytest.approx(0.0, abs=1e-6)
    assert s["ca_dist_wasserstein_A_mean"] == pytest.approx(0.0, abs=1e-6)
    assert s["state_occupancy_jsd"] == pytest.approx(0.0, abs=1e-6)
    assert s["transition_row_jsd"] > 0.2
    # The ordered trajectory changes state 3 times in 119 steps; the shuffled one far more.
    ka = cmp.state_summary([a], k, frame_ns=1.0)["state_switches_per_ns"]
    kb = cmp.state_summary([b], k, frame_ns=1.0)["state_switches_per_ns"]
    assert ka == pytest.approx(3 / 119)
    assert kb > 5 * ka


def test_states_recover_the_two_conformations():
    traj, state = fake_traj()
    f = features(traj)
    cmp.fit_states([f], k=2)
    agree = (f.labels == state).mean()
    assert max(agree, 1 - agree) == 1.0


def test_different_dynamics_are_detected():
    still, _ = fake_traj(dwell=10 ** 6)            # never leaves the straight state
    moving, _ = fake_traj(seed=2)
    a, b = features(still, "a"), features(moving, "b")
    _, k = cmp.fit_states([a, b], k=2)
    s, _ = cmp.compare_groups([a], [b], k)
    assert s["state_occupancy_jsd"] > 0.2
    assert s["rmsf_ratio_mean"] < 0.5
    assert s["ca_dist_wasserstein_A_max"] > 1.0


def test_circular_wasserstein_wraps_around():
    a = np.full((50, 1), 170.0)
    b = np.full((50, 1), -170.0)
    assert cmp.circular_wasserstein(a, b)[0] == pytest.approx(20.0, abs=5.0)
    assert cmp.circular_wasserstein(a, a)[0] == 0.0


def test_js_divergence_limits():
    p = np.array([[1.0, 0.0], [1.0, 1.0]])
    q = np.array([[0.0, 1.0], [2.0, 2.0]])
    assert cmp.js_divergence(p, q) == pytest.approx([1.0, 0.0])


def test_wasserstein_columns_shift():
    x = np.random.default_rng(0).normal(size=(500, 3))
    assert cmp.wasserstein_columns(x, x + 2.0) == pytest.approx([2.0] * 3)


def test_kabsch_rmsd_ignores_rotation():
    x = helix(10)
    c, s = np.cos(1.0), np.sin(1.0)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    assert cmp.kabsch_rmsd(x, x @ rot.T + 3.0) == pytest.approx(0.0, abs=1e-6)


def test_to_backbone_drops_other_atoms_and_renames_histidine():
    top = md.Topology()
    chain = top.add_chain()
    for name in ("HSD", "ALA"):
        res = top.add_residue(name, chain)
        for atom in ("N", "HN", "CA", "CB", "C", "O"):
            top.add_atom(atom, md.element.carbon, res)
    water = top.add_residue("TIP3", top.add_chain())
    top.add_atom("OH2", md.element.oxygen, water)
    traj = md.Trajectory(np.zeros((2, top.n_atoms, 3), dtype=np.float32), top)
    bb = to_backbone(traj)
    assert [a.name for a in bb.topology.atoms] == ["N", "CA", "C", "O"] * 2
    assert [r.name for r in bb.topology.residues] == ["HIS", "ALA"]
