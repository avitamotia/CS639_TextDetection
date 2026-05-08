# CS639 Text Detection (Group 1)

LLM-generated text detection — **Pranshu Dewagan, Ernie Dippold, Akshat Jain, Aviaditya Tamotia** (University of Wisconsin–Madison).

## Environment

Use a Conda (or venv) environment with Python ≥ 3.10. **Do not use Apple’s Xcode `python`** (it lacks dependencies).

```bash
conda create -n cs639 python=3.10
conda activate cs639
pip install -r requirements.txt
```

## EDA (MAGE / DeepFake text)

```bash
jupyter notebook eda_deepfake.ipynb
```

Loads [`yaful/MAGE`](https://huggingface.co/datasets/yaful/MAGE) from Hugging Face (shapes, labels, domains, length stats, samples).

## RAID dataset (EDA)

```bash
jupyter notebook raid_exploration.ipynb
```

Uses [`liamdugan/raid`](https://huggingface.co/datasets/liamdugan/raid) on Hugging Face.

## Experiment C — frozen SBERT + hypersphere (one-class vs soft-boundary)

Implementation: `experiments/objective_ablation/`. **Metrics:** ROC–AUC, PR–AUC, FPR at fixed TPR (default 0.95), score means/std per class, accuracy / precision / recall at the F1-optimal threshold; **checkpoint:** best validation ROC–AUC (not last epoch). **Figures** (if `matplotlib` is installed): `figures_c/` — ROC, PR, score histograms, PCA of projected `z`.

### One-command reproduction

```bash
bash scripts/run_experiment_c_fast.sh
```

Equivalent:

```bash
python -m experiments.objective_ablation.run_experiment_c --fast --out-dir runs/exp_c_fast
```

**Fast preset:** 3k random train rows, 4 epochs, batch 48, soft-boundary warmup 1, `ν=0.1`, stratified 10% val — matches the short-run setting used for internal comparisons.

### Held-out MAGE evaluation (optional)

Stratified subset of official `test` split (8k rows by default):

```bash
bash scripts/run_experiment_c_with_heldout.sh
```

Manual:

```bash
python -m experiments.objective_ablation.run_experiment_c --fast --out-dir runs/exp_c_heldout \
  --report-split test --report-max-samples 8000
```

`--report-max-samples 0` uses the full split. Metrics also appear under `report_split_metrics` in each `metrics_*.json`.

### Longer / full runs

```bash
# Full MAGE train (no subsample), more epochs (slow)
python -m experiments.objective_ablation.run_experiment_c --out-dir runs/exp_c_full \
  --max-samples 0 --epochs 25
```

Intermediate settings (e.g. `--max-samples 15000 --epochs 10 --batch-size 64`) are fine; **document whatever you use in the report**.

### Outputs

Under `--out-dir` (e.g. `runs/exp_c_fast/`):

- `metrics_one_class_nu0.1.json`, `metrics_soft_boundary_nu0.1.json` — full history + `best_metrics`
- `head_*.pt` — projection head weights (**best val ROC–AUC** epoch)
- `experiment_c_summary.json`
- `figures_c/*.png` — curves and diagnostics

## HW5 — Final report (`report.pdf`) and submission

Course expectation: **`report.pdf` in the top level of this repository** (ACL format, ≤ 8 pages excluding references; appendices optional).

### Figures inside `report/`

After you re-run Experiment C, refresh the images copied into the report tree so `make` does not embed stale plots:

```bash
cp runs/exp_c_fast/figures_c/*.png report/figures_exp_c/
```

### Installing LaTeX (macOS)

If `make` prints `pdflatex: command not found`, you do not have TeX on your **PATH** (or it is not installed).

**Option A: MacTeX (full, ~4 GB, easiest)**  
Download from [tug.org/mactex](https://www.tug.org/mactex/). After install, open a **new** terminal and check:

```bash
which pdflatex
# should show something like /Library/TeX/texbin/pdflatex
```

**Option B: BasicTeX (small, then add packages)**  
`brew install --cask basictex` then **log out and back in** (or reboot). Add TeX to `PATH` if needed:

```bash
export PATH="/Library/TeX/texbin:$PATH"
which pdflatex
```

If the build complains about a missing style file, run:

```bash
sudo tlmgr update --self
sudo tlmgr install collection-fontsrecommended
```

**Option C: Overleaf (no local install)**  
Zip the contents of the `report/` folder (`main.tex`, `references.bib`, `acl.sty`, `acl_natbib.bst`, `figures_exp_c/`). Create a blank project on [Overleaf](https://www.overleaf.com/), upload the zip, set the main document to `main.tex`, click Recompile, then download the PDF and save it as **`report.pdf`** in the repo root for Canvas.

### Build the PDF locally

1. Ensure `pdflatex` works (`which pdflatex`).
2. From the repo:

```bash
cd report
make
```

The `Makefile` uses **`latexmk` if it exists**, otherwise runs **`pdflatex` → `bibtex` → `pdflatex` ×2**. If `pdflatex` is missing, `make` prints a short hint.

This writes **`../report.pdf`** (repo root).

3. If you re-ran experiments, update numbers in `report/main.tex` if needed, refresh `report/figures_exp_c/`, then `make` again.
4. Check page count (≤ 8 pages body per course rules) and author contributions.

### Code / reproducibility checklist (grading)

- [ ] `bash scripts/run_experiment_c_fast.sh` works after `pip install -r requirements.txt`.
- [ ] README matches the commands you actually used for the report.
- [ ] All teammates have meaningful commits (see report “Author contributions”).

## References (code inspiration)

OOD LLM detection framing and SVDD-style training (public code): [cong-zeng/ood-llm-detect](https://github.com/cong-zeng/ood-llm-detect) (NeurIPS 2025; [paper](https://arxiv.org/abs/2510.08602)).
