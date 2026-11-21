"""Proves the checker catches the bugs it is meant to catch (and the original Colab error)."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import devcheck as dc
dc.enable(); D = dc.FAKE_DEV
def expect_fail(f, what):
    try: f(); print('NOT CAUGHT:', what); return False
    except dc.DeviceMismatch as e: print('caught:', what, '->', str(e)[:90]); return True
ok = True
a = torch.ones(3).to(D); b = torch.ones(3)
ok &= expect_fail(lambda: a + b, 'cuda + cpu')
ok &= expect_fail(lambda: torch.searchsorted(b, a), 'searchsorted cpu table, cuda input (the Colab bug)')
ok &= expect_fail(lambda: a.numpy(), 'numpy on cuda')
ok &= expect_fail(lambda: b[torch.tensor([0], device=D)], 'cpu[cuda index]')
g = torch.Generator(device=D)
ok &= expect_fail(lambda: torch.rand(3, generator=g), 'cuda generator, cpu output')
m = torch.nn.Linear(3, 1).to(D)
ok &= expect_fail(lambda: m(torch.ones(2, 3)), 'cuda module, cpu input')
assert (a + 1).device == D and a.cpu().device == torch.device('cpu') and (a * torch.tensor(2.0)).device == D
assert m(torch.ones(2, 3, device=a.device)).device == D and torch.zeros(2, device=a.device).device == D
x = a.cpu(); _ = x + b   # legal
_ = a[torch.tensor([0])]  # legal on CUDA
# the real Transmittance class: old behaviour reproduced, new behaviour fixed
import synplume as sp
tm = sp.Transmittance(sys.argv[1])
r = tm.rel(torch.tensor([0, 1], device=D), torch.tensor([2.3, 2.5], device=D), torch.full((2, 5), 1000.0, device=D))
assert r[0].device == D
r = tm.rel(torch.tensor([0]), torch.tensor([2.3]), torch.tensor([[1000.0]])); assert r[0].device == torch.device('cpu')
print('self-test', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
