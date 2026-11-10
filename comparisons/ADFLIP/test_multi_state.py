"""Offline smoke test for multi_state.adaptive_sample_multi_state -- `python3 test_multi_state.py`.

ADFLIP's real deps (prody, scipy, torch_scatter, ema_pytorch) are not installed outside its
Docker image, so prody/scipy are stubbed here just far enough to import data.all_atom_parse for
its token vocabulary. The model, io and corruption steps are faked; what is under test is the
sampling loop itself -- shapes, termination, and the claim that N==1 reproduces the single-state
trajectory step for step.
"""
import os
import sys
import types

import numpy as np
import torch

STDAA = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
         'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

prody_stub = types.ModuleType('prody')
prody_stub.flagDefinition = lambda flag: {
    'stdaa': STDAA,
    'nonstdaa': ['MSE', 'SEP'],
    'nucleotide': ['DA', 'DC', 'DG', 'DT', 'A', 'C', 'G', 'U'],
}[flag]
prody_stub.parsePDB = lambda *a, **k: None
prody_stub.proteins = types.SimpleNamespace(
    writePDB=lambda *a, **k: None,
    pdbfile=types.SimpleNamespace(writePDB=lambda *a, **k: None))
sys.modules['prody'] = prody_stub

scipy_stub = types.ModuleType('scipy')
spatial_stub = types.ModuleType('scipy.spatial')
spatial_stub.cKDTree = object
scipy_stub.spatial = spatial_stub
sys.modules['scipy'] = scipy_stub
sys.modules['scipy.spatial'] = spatial_stub

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multi_state  # noqa: E402

residue_tokens = multi_state.residue_tokens
restype_1to3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}

MASK = residue_tokens['<MASK>']
V = len(residue_tokens)
T = 30

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
STANDARD_TOKENS = [residue_tokens[restype_1to3[c]] for c in ALPHABET]


class _Denoiser:
    """Callable with the nn.Module surface the sampler touches."""

    def __init__(self, fn):
        self.fn = fn

    def eval(self):
        return self

    def __call__(self, noisy, t):
        return self.fn(noisy, t)


class FakeFlow:
    """Enough of DiscreteFlow_AA for the loop: io, corruption, and a denoiser.

    The denoiser returns logits peaked on the native sequence with a fixed confidence, so the
    adaptive threshold is crossed and the loop terminates. `per_state_offset` lets a state return
    something different from its neighbours, to check averaging actually mixes them.
    """

    def __init__(self, native, confidence=4.0, per_state_offset=None):
        self.native = native
        self.confidence = confidence
        self.per_state_offset = per_state_offset or {}
        self.calls = []
        # Upstream's flow.model is an nn.Module, so the sampler calls .eval() on it.
        self.model = _Denoiser(self._denoise)

    def _denoise(self, noisy, t):
        path = noisy['__path__']
        self.calls.append((float(t.item()), path))
        logits = torch.zeros(T, V)
        logits[torch.arange(T), self.native] = self.confidence + self.per_state_offset.get(path, 0.0)
        return logits, None

    def _prepare_sampling_io(self, path):
        data = {
            'is_center': torch.ones(T, dtype=torch.bool),
            'is_protein': torch.ones(T, dtype=torch.bool),
            'backbone_mask': torch.ones(T, dtype=torch.bool),
            'residue_token': self.native.clone(),
            '__path__': path,
        }
        return data, path, '/tmp/fake_save', torch.device('cpu')

    def corrupt_data_by_sample(self, data, time, sample):
        return data, {'__path__': data['__path__']}


def native_sequence(seed=0):
    generator = torch.Generator().manual_seed(seed)
    classes = torch.randint(0, 20, (T,), generator=generator)
    return torch.tensor([STANDARD_TOKENS[c] for c in classes])


def run(flow, paths, seed=0, **kwargs):
    torch.manual_seed(seed)
    return multi_state.adaptive_sample_multi_state(flow, paths, verbose=False, **kwargs)


def main():
    native = native_sequence()
    failures = []

    def check(name, condition, detail=''):
        print('{:<58} {}'.format(name, 'PASS' if condition else 'FAIL ' + detail))
        if not condition:
            failures.append(name)

    # 1. Single state: returns, shapes are right, high-confidence logits are recovered.
    flow = FakeFlow(native)
    samples, log_probs, true_tokens = run(flow, ['a.pdb'])
    check('single state: samples shape (1, T)', tuple(samples.shape) == (1, T),
          str(tuple(samples.shape)))
    check('single state: log_probs shape (T, V)', tuple(log_probs.shape) == (T, V),
          str(tuple(log_probs.shape)))
    check('single state: true_tokens returned', torch.equal(true_tokens, native))
    check('single state: log_probs are normalised log-probs',
          torch.allclose(log_probs.exp().sum(-1), torch.ones(T), atol=1e-5))
    recovery = (samples.view(-1) == native).float().mean().item()
    check('single state: recovers the peaked native sequence', recovery == 1.0,
          'recovery={:.3f}'.format(recovery))
    check('single state: no <MASK> left in the design', not bool((samples == MASK).any()))

    # 2. N identical states must reproduce the single-state trajectory exactly -- averaging
    #    identical distributions is a no-op, so the drawn samples and the forward-call times
    #    must match call for call.
    flow_single = FakeFlow(native)
    s1, lp1, _ = run(flow_single, ['a.pdb'])
    times_single = [t for t, _ in flow_single.calls]

    flow_multi = FakeFlow(native)
    s3, lp3, _ = run(flow_multi, ['a.pdb', 'b.pdb', 'c.pdb'])
    times_multi = sorted(set(t for t, _ in flow_multi.calls))

    check('N identical states: same samples as N=1', torch.equal(s1, s3))
    check('N identical states: same final distribution as N=1',
          torch.allclose(lp1, lp3, atol=1e-6))
    check('N identical states: same step times as N=1',
          [round(t, 6) for t in sorted(set(times_single))] == [round(t, 6) for t in times_multi],
          '{} vs {}'.format(sorted(set(times_single)), times_multi))
    check('N states: denoiser called once per state per step',
          len(flow_multi.calls) == 3 * len(set(times_multi)),
          '{} calls, {} steps'.format(len(flow_multi.calls), len(set(times_multi))))

    # 3. Averaging actually mixes states: one state voting for a different sequence must move
    #    the final distribution away from the unanimous case.
    other = native_sequence(seed=1)
    disagreeing = FakeFlow(native)
    disagreeing.native = native

    class Disagreeing(FakeFlow):
        def _denoise(self, noisy, t):
            path = noisy['__path__']
            self.calls.append((float(t.item()), path))
            target = other if path == 'c.pdb' else self.native
            logits = torch.zeros(T, V)
            logits[torch.arange(T), target] = self.confidence
            return logits, None

    mixed = Disagreeing(native)
    _, lp_mixed, _ = run(mixed, ['a.pdb', 'b.pdb', 'c.pdb'])
    differing = (native != other)
    check('averaging mixes states (dissenting state shifts the mean)',
          not torch.allclose(lp_mixed[differing], lp3[differing], atol=1e-3))
    check('averaging: majority still wins at differing positions',
          bool((lp_mixed[differing].argmax(-1) == native[differing]).all()))

    # 4. A model that never becomes confident must still terminate, via min_dt.
    unconfident = FakeFlow(native, confidence=0.0)
    samples_u, _, _ = run(unconfident, ['a.pdb'], num_step=8)
    check('zero-confidence model still terminates', tuple(samples_u.shape) == (1, T))
    check('zero-confidence model leaves no <MASK>', not bool((samples_u == MASK).any()))

    # 5. Guards.
    try:
        run(FakeFlow(native), ['a.pdb', 'a.pdb'])
        check('duplicate stems rejected', False, 'no error raised')
    except ValueError as exc:
        check('duplicate stems rejected', 'distinct file stems' in str(exc))

    class Mismatched(FakeFlow):
        def _prepare_sampling_io(self, path):
            data, p, folder, device = super()._prepare_sampling_io(path)
            if path == 'b.pdb':
                data['residue_token'] = native_sequence(seed=7)
            return data, p, folder, device

    try:
        run(Mismatched(native), ['a.pdb', 'b.pdb'])
        check('disagreeing native sequences rejected', False, 'no error raised')
    except ValueError as exc:
        check('disagreeing native sequences rejected', 'disagrees' in str(exc))

    # 6. The 20-class restriction the eval script applies.
    probs20 = log_probs.exp()[:, STANDARD_TOKENS]
    probs20 = probs20 / probs20.sum(-1, keepdim=True)
    check('20-class restriction keeps the argmax',
          bool((probs20.argmax(-1) == torch.tensor(
              [STANDARD_TOKENS.index(int(t)) for t in native])).all()))

    print()
    if failures:
        print('{} FAILED: {}'.format(len(failures), failures))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
