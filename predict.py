#!/usr/bin/env python3
"""|M|^2 of the FA surrogate of one KIT process, float64, CPU or GPU (README.txt).
    python predict.py <ee_uug|ee_uugg|ee_tt> <in.h5> --out <out.h5> [--splits <split> ...]
<in.h5>: the benchmark's layout, <split>/<stratum>/momenta (N, n, 4), E px py pz in GeV, CM frame, order e+ e- finals,
and the root attribute `metadata` of the benchmark's file for that process. Out (float64): <split>/<stratum>/log10_me2
(ee_tt: log10_abs_virt_finite) and the value itself under the target's name (me2; ee_tt: virt_finite, sign included)."""
import argparse, json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument("process"); ap.add_argument("inp"); ap.add_argument("--out", required=True)
ap.add_argument("--splits", nargs="*", default=None); a = ap.parse_args()
os.environ["KIT_HARDPS_DIR"] = os.path.join(HERE, "kit_data")      # the models' own training events, shipped here
for k in ("DATA_DIR", "SCRATCH"):
    if k not in os.environ:
        os.environ[k] = tempfile.mkdtemp(prefix="fa_kit_")
os.environ.setdefault("KIT_THREADS", "1")   # CPU results reproduce ours exactly at 1 thread (README)
meta = json.load(open(os.path.join(HERE, "models.json")))
(m,) = [x for x in meta["models"] if x["file"] == a.process] or sys.exit(f"process: one of {[x['file'] for x in meta['models']]}")
run = os.path.join(HERE, "runs", "kit", m["run"])
if HERE not in open(os.path.join(run, "config.yaml")).read():
    subprocess.run([sys.executable, os.path.join(HERE, "tools", "kit_package.py"), "relocate"], cwd=HERE, check=True)
import h5py, numpy as np
from tools.kit_predict import predict
with h5py.File(a.inp, "r") as f:
    splits = a.splits or sorted(f)
tmp = tempfile.mkdtemp(prefix="fa_kit_out_", dir=os.environ["SCRATCH"])
_, path = predict(run, splits, tmp, h5=os.path.abspath(a.inp))
with h5py.File(path, "r") as p, h5py.File(a.out, "w") as o:
    for sp in p:
        for st in p[sp]:
            (k,) = list(p[f"{sp}/{st}"])
            y = p[f"{sp}/{st}/{k}"][:].astype(np.float64)
            o[f"{sp}/{st}/{k}"] = y    # the benchmark's own name: log10_me2, or log10_abs_virt_finite for ee_tt
            o[f"{sp}/{st}/{m['target']}"] = np.power(10.0, y)   # and the value, signed (every model here predicts > 0)
    o.attrs["model"] = m["run"]
import shutil
shutil.rmtree(tmp, ignore_errors=True)
print(f"wrote {a.out}")
