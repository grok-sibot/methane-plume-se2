"""Smoke-runs the round-2 notebook with two forced edge cases, under the fake-GPU device checker:
  * disk: before section 6, DISK_RESERVE_GB is raised so the pair caches do not fit -> the training pool must shrink
    (and be recorded), then the reserve is restored;
  * time: just before training, TIME_BUDGET_MIN is set so only the first variant fits -> the others must be skipped
    and reported, and evaluation must run on the trained one.
usage: python edge_cases_r2.py OUT_DIR"""
import sys, os, subprocess, json, nbformat as nbf
HERE = os.path.dirname(os.path.abspath(__file__)); NB = os.path.join(os.path.dirname(HERE), 'synthetic_plumes_r2_colab.ipynb')
out = os.path.abspath(sys.argv[1]); os.makedirs(out, exist_ok=True)
nb = nbf.read(NB, as_version=4)
find = lambda key: next(i for i, c in enumerate(nb.cells) if c.cell_type == 'code' and key in c.source)
nb.cells.insert(find("DEV = torch.device('cuda'") + 1, nbf.v4.new_code_cell(
    f"sys.path.insert(0, {HERE!r}); import devcheck\nDEVCHECK = devcheck.enable(); DEV = devcheck.FAKE_DEV"))
i6 = find("PSUB = 'methaneset-s2-pretraining'"); c6 = nb.cells[i6]
key = "SHRUNK = None\n"; assert c6.source.count(key) == 1
c6.source = c6.source.replace(key, key + "DISK_RESERVE_GB = free_gb() - (cache_left('pairs_valfa', len(VAL_ROWS)) + cache_left('pairs_testfa', len(FA_ROWS)) "
                              "+ 0.5 * cache_left('pairs_train', len(TRAIN_ROWS))); print('edge test: reserve', DISK_RESERVE_GB)\n")
nb.cells.insert(i6 + 1, nbf.v4.new_code_cell("assert SHRUNK and len(TRAIN_ROWS) < SHRUNK[0], SHRUNK; print('edge test: pool shrunk', SHRUNK); DISK_RESERVE_GB = 0.0"))
i11 = find("sp.train_syn(")
nb.cells.insert(i11, nbf.v4.new_code_cell("TIME_BUDGET_MIN = EVAL_RESERVE_MIN + (time.time() - T0) / 60 + 0.05; print('edge test: budget', TIME_BUDGET_MIN)"))
nb.cells.insert(i11 + 2, nbf.v4.new_code_cell("assert TRAINED == RUNS_TO_TRAIN[:1] and set(SKIPPED) == set(RUNS_TO_TRAIN[1:]), (TRAINED, SKIPPED); print('edge test: skipped', list(SKIPPED))"))
nb.cells.append(nbf.v4.new_code_cell("S = json.load(open(f'{OUT}/syn_results.json')); assert S['train_pool_shrunk_for_disk'] and S['skipped'] and S['trained'] == TRAINED\n"
                                     "print('edge test PASS | devcheck', DEVCHECK.ops, 'torch calls,', DEVCHECK.tagged_ops, 'on the fake GPU, 0 mismatches')"))
src = f'{out}/edge_input.ipynb'; nbf.write(nb, src)
r = subprocess.run(['papermill', src, f'{out}/edge_executed.ipynb', '-p', 'SMOKE_TEST', 'True', '-p', 'WORK', f'{out}/work', '-p', 'FT_WORK', f'{out}/ft',
                    '-p', 'WORKERS', '2', '--cwd', out, '--log-output'], capture_output=True, text=True)
log = r.stdout + r.stderr; open(f'{out}/edge.log', 'w').write(log)
print('\n'.join(l for l in log.splitlines() if 'edge test' in l or 'DeviceMismatch' in l or 'Exception encountered' in l or 'SKIPPED' in l or 'DISK' in l))
print('EXIT', r.returncode); sys.exit(r.returncode)
