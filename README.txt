
FA surrogates of e+e- -> u u~ g, e+e- -> u u~ g g (|M|^2) and e+e- -> t t~ one-loop virtual (finite part)
for the KIT hard phase-space benchmark: checkpoints, everything needed to load them, and predict.py. float64 throughout.

Install (Python 3.11; a fresh venv):
  CPU:  pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
        pip install -r scripts/cpu_env_requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
  GPU:  pip install -r requirements.txt     (torch 2.1.2 for your CUDA, xformers)
Run (one command per process; <in.h5> in the benchmark's layout: <split>/<stratum>/momenta, E px py pz in GeV, CM frame,
order e+ e- finals, and the benchmark file's root attribute `metadata`; any target datasets are ignored):
  python predict.py ee_uug  <in.h5> --out uug_me2.h5
  python predict.py ee_uugg <in.h5> --out uugg_me2.h5
  python predict.py ee_tt   <in.h5> --out tt_me2.h5
Output, float64, under the benchmark's own dataset names: <split>/<stratum>/log10_me2 = log10|M|^2 (ee_tt:
log10_abs_virt_finite = log10|V|, V the finite one-loop virtual) and the value itself (me2; ee_tt: virt_finite, with its
sign: the ee_tt model predicts V > 0, as V is on every event of the benchmark's ee_tt file). CPU threads are pinned to 1 (KIT_THREADS), where the results are reproducible to float64 rounding; our own
predictions (made on GPU) differ from them by up to 2.6e-5 (ee_uug), 6.6e-5 (ee_uugg) and, on ee_tt, 2e-4 (bulk),
4.9e-4 (threshold) and 5.5e-3 (Sudakov bins) in log10, a rounding sensitivity of the models well below their errors there.
Check: python predict.py ee_uug examples/ee_uug_bulk5.h5 --out /tmp/x.h5 must give examples/ee_uug_bulk5.expected.txt
(log10 of the first 5 bulk_test events of your ee_uug.h5; same for ee_uugg, ee_tt) to float64 rounding.
Limits: trained on the bulk only (100000 bulk_train events each; no hard_train point): for ee_uug/ee_uugg the s_ij/s > 0.01
cut on the singular pairs at sqrt s = 1 TeV, for ee_tt sqrt s in [353.1, 1738.7] GeV. Outside that, in the soft,
collinear, threshold and high-energy limits, the asymptotic factor in the model carries the known singular scaling;
the benchmark's hard strata measure how well. kit_data/ holds the models' own training events (their statistics are
rebuilt from them and checked); models.json names the runs and the code's commit.
