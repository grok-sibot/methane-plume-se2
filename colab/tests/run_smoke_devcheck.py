"""Run the synthetic-plume notebook's smoke test on a CPU box with the fake-GPU device checker (devcheck.py) active
from the setup cell on, so every section runs with DEV = a pretend CUDA device and any CPU/GPU tensor mix raises.
usage: python run_smoke_devcheck.py OUT_DIR [--notebook NB.ipynb] [--module path/to/synplume.py] [--amp-bf16] [extra papermill -p k v ...]
--amp-bf16 also forces mf.amp_dtype_for to bfloat16 so the autocast branches (CUDA-only otherwise) run too."""
import sys, os, json, subprocess, nbformat as nbf
HERE = os.path.dirname(os.path.abspath(__file__)); NB = os.path.join(os.path.dirname(HERE), 'synthetic_plumes_colab.ipynb')
out = os.path.abspath(sys.argv[1]); args = sys.argv[2:]; os.makedirs(out, exist_ok=True)
module = None
if '--notebook' in args: i = args.index('--notebook'); NB = os.path.abspath(args[i + 1]); del args[i:i + 2]
if '--module' in args: i = args.index('--module'); module = args[i + 1]; del args[i:i + 2]
nb = nbf.read(NB, as_version=4)
setup = next(i for i, c in enumerate(nb.cells) if c.cell_type == 'code' and "DEV = torch.device('cuda'" in c.source)
nb.cells.insert(setup + 1, nbf.v4.new_code_cell(
    f"sys.path.insert(0, {HERE!r}); import devcheck\nDEVCHECK = devcheck.enable(); DEV = devcheck.FAKE_DEV\n"
    "print('devcheck: fake CUDA device active; DEV =', DEV)"))
if module:
    w = next(c for c in nb.cells if c.cell_type == 'code' and c.source.startswith('%%writefile synplume.py'))
    w.source = '%%writefile synplume.py\n' + open(module).read()
if '--amp-bf16' in args:
    args.remove('--amp-bf16')
    g = next(i for i, c in enumerate(nb.cells) if c.cell_type == 'code' and 'sp.sanity_gate(' in c.source)
    nb.cells.insert(g, nbf.v4.new_code_cell("mf.amp_dtype_for = lambda dev: torch.bfloat16; print('devcheck: autocast forced to bfloat16')"))
nb.cells.append(nbf.v4.new_code_cell("print(f'devcheck: {DEVCHECK.ops} torch calls checked, {DEVCHECK.tagged_ops} on the fake GPU, 0 mismatches')"))
src = f'{out}/devcheck_input.ipynb'; nbf.write(nb, src)
cmd = ['papermill', src, f'{out}/devcheck_executed.ipynb', '-p', 'SMOKE_TEST', 'True', '-p', 'WORK', f'{out}/work',
       '-p', 'FT_WORK', f'{out}/ft', '-p', 'WORKERS', '2', '--cwd', out, '--log-output'] + args
r = subprocess.run(cmd, capture_output=True, text=True)
open(f'{out}/devcheck.log', 'w').write(r.stdout + r.stderr)
tail = [l for l in (r.stdout + r.stderr).splitlines() if 'devcheck' in l or 'DeviceMismatch' in l or 'Exception encountered' in l]
print('\n'.join(tail[-8:])); print('EXIT', r.returncode); sys.exit(r.returncode)
