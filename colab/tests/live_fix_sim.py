"""Replays Simon's live Colab session under the fake-GPU checker: the OLD notebook (ef15216) runs until the gate
crashes, then the documented fix cell (fresh synplume.py, reload, rebuild TM), the gate cell again, and every cell
after it unchanged (the old section-11 cell still calls TM.to(DEV)).
usage: python live_fix_sim.py OLD_NOTEBOOK NEW_SYNPLUME OUT_DIR"""
import sys, os, papermill as pm, nbformat as nbf
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
HERE = os.path.dirname(os.path.abspath(__file__)); old_nb, new_mod, out = sys.argv[1], os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
os.makedirs(out, exist_ok=True)
nb = nbf.read(old_nb, as_version=4)
for c in nb.cells: c.metadata.setdefault('tags', [])
nb.metadata.setdefault('papermill', {})
pm.parameterize.parameterize_notebook(nb, dict(SMOKE_TEST=True, WORK=f'{out}/work', FT_WORK=f'{out}/ft', WORKERS=2))
setup = next(i for i, c in enumerate(nb.cells) if c.cell_type == 'code' and "DEV = torch.device('cuda'" in c.source)
nb.cells.insert(setup + 1, nbf.v4.new_code_cell(f"sys.path.insert(0, {HERE!r}); import devcheck\nDEVCHECK = devcheck.enable(); DEV = devcheck.FAKE_DEV"))
gate = next(i for i, c in enumerate(nb.cells) if c.cell_type == 'code' and 'sp.sanity_gate(' in c.source)
code = [i for i, c in enumerate(nb.cells) if c.cell_type == 'code']
kc = NotebookClient(nb, timeout=3600, kernel_name='python3', resources={'metadata': {'path': out}})
with kc.setup_kernel():
    for i in [j for j in code if j < gate]: kc.execute_cell(nb.cells[i], i)
    try: kc.execute_cell(nb.cells[gate], gate); print('gate unexpectedly passed with the old module'); sys.exit(1)
    except CellExecutionError as e: print('old module, gate cell fails as on Colab:', str(e).strip().splitlines()[-1][:200])
    fix = nbf.v4.new_code_cell(f"import importlib, shutil; shutil.copy({new_mod!r}, 'synplume.py')\n"
                               "importlib.reload(sp); shutil.copy('synplume.py', f'{OUT}/synplume.py'); TM = sp.Transmittance(TJ)")
    nb.cells.append(fix); kc.execute_cell(fix, len(nb.cells) - 1); print('fix cell ran')
    for i in [j for j in code if j >= gate]: kc.execute_cell(nb.cells[i], i)
    fin = nbf.v4.new_code_cell("print(f'devcheck: {DEVCHECK.ops} torch calls checked, {DEVCHECK.tagged_ops} on the fake GPU, 0 mismatches')")
    nb.cells.append(fin); kc.execute_cell(fin, len(nb.cells) - 1); print(''.join(o.get('text', '') for o in fin.outputs).strip())
nbf.write(nb, f'{out}/live_fix_executed.ipynb'); print('live-fix replay PASS')
