#!/usr/bin/env python3
"""
Experiment C — **One-class vs soft-boundary** hypersphere (MAGE + frozen SBERT).

What this script does
---------------------
1. Load MAGE; stratified train/val split.
2. Encode text with a **frozen** SentenceTransformer; train only a linear head into ``out_dim``.
3. Initialize hypersphere center ``c`` from **LLM-only** training projections.
4. Train with either:
   - **one_class:** separation loss only;
   - **soft_boundary:** same separation + soft-boundary hinge on LLM rows; after
     ``warmup_epochs``, update radius ``R`` each epoch from the ``(1-nu)`` quantile
     of ``sqrt(d_sq)`` on training LLM points (reference-style schedule).
5. Log **validation score scale** (mean/std per class) and **FPR @ fixed TPR** (default 0.95).
6. **Best-checkpoint** by validation ROC-AUC: weights and final tables use that epoch
   (not the last epoch); optional **held-out** MAGE split via ``--report-split``.
7. **AUROC / PR-AUC** plus **Acc / Prec / Rec** at the F1-optimal score threshold; PR + PCA figures.
8. Optional **nu sweep** for soft-boundary only.
9. Save JSON metrics + PNG figures (if matplotlib installed).

Run (from repo root, Conda env with deps)::

    python -m experiments.objective_ablation.run_experiment_c \\
        --out-dir runs/exp_c --max-samples 8000 --epochs 15

If ``python`` is Xcode on your Mac, activate conda first or use::

    conda run -n your_env python -m experiments.objective_ablation.run_experiment_c ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch

from experiments.objective_ablation.data import (
    batches,
    load_mage_rows,
    stratified_downsample,
    stratified_train_val_indices,
)
from experiments.objective_ablation.diagrams import save_figure_bundle
from experiments.objective_ablation.hypersphere import HypersphereProjection
from experiments.objective_ablation.metrics import check_sklearn_available, evaluate
from experiments.objective_ablation.objectives import (
    ObjectiveName,
    eval_scores,
    training_loss,
    update_radius_from_train_machine,
)


def _abort_if_xcode_python() -> None:
    """Xcode’s python has no course packages; exit with a short hint."""
    exe = sys.executable
    if "Xcode" in exe or "Python3.framework" in exe:
        print(
            "This is Apple’s Xcode Python, not your Conda env.\n"
            f"  {exe}\n"
            "Run: conda activate <env>   then check: which python\n",
            file=sys.stderr,
        )
        raise SystemExit(2)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_batch(st_model, texts: List[str], device: torch.device) -> torch.Tensor:
    """
    Sentence-BERT encode; no gradients through the trunk.

    Newer ``sentence_transformers`` + PyTorch can return **inference-mode** tensors.
    ``.detach().clone()`` is not always enough; copying through NumPy guarantees a
    normal tensor that ``nn.Linear`` can use in ``backward`` (grad only on the head).
    """
    with torch.no_grad():
        t = st_model.encode(
            texts,
            convert_to_tensor=True,
            device=str(device),
            show_progress_bar=False,
        )
        arr = t.float().detach().cpu().numpy()
    return torch.from_numpy(np.ascontiguousarray(arr)).to(
        device=device, dtype=torch.float32
    )


@torch.no_grad()
def init_center_machine(
    st_model,
    head: HypersphereProjection,
    train_texts: List[str],
    train_labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> None:
    """Accumulate all training LLM rows, set ``c`` to normalized mean projection."""
    chunks: List[torch.Tensor] = []
    for i in range(0, len(train_texts), batch_size):
        tx = train_texts[i : i + batch_size]
        lb = train_labels[i : i + batch_size]
        emb = encode_batch(st_model, tx, device)
        z = head(emb)
        mask = torch.tensor(lb == 0, device=device)
        if mask.any():
            chunks.append(z[mask])
    if not chunks:
        raise RuntimeError("No LLM (label 0) rows in training split.")
    head.init_center_from_machine(torch.cat(chunks, dim=0))


@torch.no_grad()
def collect_llm_squared_distances(
    st_model,
    head: HypersphereProjection,
    texts: List[str],
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Concatenate ``d_sq`` for every row with label 0 (LLM)."""
    parts: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        tx = texts[i : i + batch_size]
        lb = labels[i : i + batch_size]
        emb = encode_batch(st_model, tx, device)
        z = head(emb)
        d = head.squared_distances(z).cpu().numpy()
        parts.append(d[lb == 0])
    if not parts:
        return np.array([], dtype=np.float64)
    return np.concatenate(parts, axis=0)


@torch.no_grad()
def collect_z_projections(
    st_model,
    head: HypersphereProjection,
    texts: List[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Projected vectors ``z`` for PCA / geometry diagnostics (after best checkpoint)."""
    was_training = head.training
    head.eval()
    parts: List[np.ndarray] = []
    try:
        for i in range(0, len(texts), batch_size):
            tx = texts[i : i + batch_size]
            emb = encode_batch(st_model, tx, device)
            z = head(emb)
            parts.append(z.float().cpu().numpy())
    finally:
        if was_training:
            head.train()
    if not parts:
        return np.zeros((0, head.out_dim), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def run_validation(
    st_model,
    head: HypersphereProjection,
    val_texts: List[str],
    val_labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    objective: ObjectiveName,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns ``y_true`` (1=LLM), ``scores`` (higher → more LLM).

    Used for ROC, FPR@TPR, and histograms.  Runs under ``no_grad`` because we are
    not training here; avoids ``.numpy()`` on tensors that still have ``requires_grad``
    when ``head`` was left in ``train()`` mode after the inner training loop.
    """
    was_training = head.training
    head.eval()
    ys: List[np.ndarray] = []
    ss: List[np.ndarray] = []
    try:
        with torch.no_grad():
            for i in range(0, len(val_texts), batch_size):
                tx = val_texts[i : i + batch_size]
                lb = val_labels[i : i + batch_size]
                emb = encode_batch(st_model, tx, device)
                z = head(emb)
                d_sq = head.squared_distances(z).cpu().numpy()
                y_b = (lb == 0).astype(np.int64)  # LLM positive
                s_b = eval_scores(d_sq, objective, radius)
                ys.append(y_b)
                ss.append(s_b)
    finally:
        if was_training:
            head.train()
    return np.concatenate(ys), np.concatenate(ss)


def train_one_setting(
    objective: ObjectiveName,
    nu: float,
    warmup_epochs: int,
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    report_texts: List[str] | None,
    report_labels: np.ndarray | None,
    out_dir: Path,
    sbert_name: str,
    out_dim: int,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    target_tpr: float,
    device: torch.device,
) -> Dict:
    """
    Full train + val for one (objective, nu) config.

    ``radius`` is in **linear** distance units; updated after warm-up for soft_boundary.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(sbert_name, device=str(device))
    st_model.eval()
    for p in st_model.parameters():
        p.requires_grad = False

    sdim = st_model.get_sentence_embedding_dimension()
    head = HypersphereProjection(sdim, out_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)

    # --- Hypersphere center from LLM training mass ---
    init_center_machine(
        st_model, head, train_texts, train_labels, device, batch_size
    )

    # Initial radius from current geometry (soft-boundary; harmless for one-class eval)
    d_llm0 = collect_llm_squared_distances(
        st_model, head, train_texts, train_labels, device, batch_size
    )
    radius = update_radius_from_train_machine(d_llm0, nu)

    history: List[Dict] = []
    best_roc = float("-inf")
    best_state = None
    best_radius_snapshot: float | None = None
    best_epoch_idx = 0

    for epoch in range(epochs):
        head.train()
        n_b, loss_acc = 0, 0.0
        for batch in batches(
            train_texts,
            train_labels,
            batch_size,
            shuffle=True,
            seed=seed + 10_000 + epoch,
        ):
            emb = encode_batch(st_model, batch.texts, device)
            z = head(emb)
            d_sq = head.squared_distances(z)
            lab = torch.tensor(batch.labels, device=device, dtype=torch.long)
            loss = training_loss(d_sq, lab, objective, radius, nu)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_acc += float(loss.item())
            n_b += 1

        # After warm-up: refresh R from **all** training LLM distances (epoch-end update)
        if objective == "soft_boundary" and epoch + 1 >= warmup_epochs:
            d_llm = collect_llm_squared_distances(
                st_model, head, train_texts, train_labels, device, batch_size
            )
            radius = update_radius_from_train_machine(d_llm, nu)

        y_val, s_val = run_validation(
            st_model,
            head,
            val_texts,
            val_labels,
            device,
            batch_size,
            objective,
            radius,
        )
        rep = evaluate(y_val, s_val, target_tpr=target_tpr)
        history.append(
            {
                "epoch": epoch + 1,
                "loss_mean": loss_acc / max(n_b, 1),
                "radius_R": float(radius),
                **rep.to_dict(),
            }
        )
        if rep.roc_auc >= best_roc:
            best_roc = rep.roc_auc
            best_epoch_idx = epoch + 1
            best_state = deepcopy(head.state_dict())
            best_radius_snapshot = float(radius)

    if best_state is None or best_radius_snapshot is None:
        raise RuntimeError("Failed to track a best validation checkpoint.")

    head.load_state_dict(best_state)
    radius = best_radius_snapshot

    y_final, s_final = run_validation(
        st_model,
        head,
        val_texts,
        val_labels,
        device,
        batch_size,
        objective,
        radius,
    )
    rep_best = evaluate(y_final, s_final, target_tpr=target_tpr)
    hist_best = history[best_epoch_idx - 1]
    tag = f"{objective}_nu{nu}"
    report_block: Dict | None = None
    if report_texts is not None and report_labels is not None:
        y_rep, s_rep = run_validation(
            st_model,
            head,
            report_texts,
            report_labels,
            device,
            batch_size,
            objective,
            radius,
        )
        report_block = evaluate(y_rep, s_rep, target_tpr=target_tpr).to_dict()
    meta = {
        "tag": tag,
        "objective": objective,
        "nu": nu,
        "warmup_epochs": warmup_epochs,
        "sbert_name": sbert_name,
        "out_dim": out_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "target_tpr": target_tpr,
        "best_epoch": best_epoch_idx,
        "best_metrics": {
            "loss_mean": hist_best["loss_mean"],
            "radius_R": float(radius),
            **rep_best.to_dict(),
        },
        "report_split_metrics": report_block,
        "history": history,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"metrics_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    torch.save(head.state_dict(), out_dir / f"head_{tag}.pt")

    val_z = collect_z_projections(
        st_model, head, val_texts, device, batch_size
    )
    return {
        "meta": meta,
        "val_y": y_final,
        "val_scores": s_final,
        "val_z": val_z,
        "radius": radius,
    }


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment C: one-class vs soft-boundary")
    p.add_argument("--out-dir", type=str, default="runs/experiment_c")
    p.add_argument(
        "--max-samples",
        type=int,
        default=12_000,
        help="Cap training rows (random subsample). Use 0 for the full MAGE train split.",
    )
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out-dim", type=int, default=128)
    p.add_argument(
        "--sbert-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    p.add_argument("--nu", type=float, default=0.1, help="Soft-boundary ν (default like ref)")
    p.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Epochs before radius R starts updating (soft-boundary only)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-tpr", type=float, default=0.95)
    p.add_argument("--skip-one-class", action="store_true")
    p.add_argument("--skip-soft-boundary", action="store_true")
    p.add_argument(
        "--nu-sweep",
        type=str,
        default="",
        help="Extra ν values for soft-boundary only, comma-separated, e.g. 0.05,0.15,0.2",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Smallest useful run: both objectives, 3k rows, 4 epochs, batch 48, warmup 1 (minutes on CPU).",
    )
    p.add_argument(
        "--report-split",
        type=str,
        default="",
        help="After training, evaluate the best checkpoint on this MAGE split (e.g. test, validation). Empty = skip.",
    )
    p.add_argument(
        "--report-max-samples",
        type=int,
        default=8_000,
        help="Max rows from --report-split (stratified subsample; 0 = full split).",
    )
    p.add_argument(
        "--no-pca-figure",
        action="store_true",
        help="Skip 2D PCA scatter of projected z on the validation set.",
    )
    ns = p.parse_args(argv if argv is not None else sys.argv[1:])
    if ns.fast:
        ns.max_samples = 3000
        ns.epochs = 4
        ns.batch_size = 48
        ns.warmup_epochs = 1  # so R updates within 4 epochs (default 5 would skip all updates)
    return ns


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    _abort_if_xcode_python()
    check_sklearn_available()

    device = pick_device()
    out_dir = Path(args.out_dir)

    # --- Data (same split for all runs in one invocation = fair comparison) ---
    max_samp = None if args.max_samples == 0 else args.max_samples
    texts, labels = load_mage_rows("train", max_samples=max_samp, seed=args.seed)
    tr_i, va_i = stratified_train_val_indices(
        labels, val_fraction=args.val_fraction, seed=args.seed
    )
    train_texts = [texts[i] for i in tr_i]
    train_labels = labels[tr_i]
    val_texts = [texts[i] for i in va_i]
    val_labels = labels[va_i]

    report_texts: List[str] | None = None
    report_labels: np.ndarray | None = None
    if args.report_split.strip():
        rpt = args.report_split.strip()
        rmax = None if args.report_max_samples == 0 else args.report_max_samples
        rt, rl = load_mage_rows(rpt, max_samples=None, seed=args.seed + 1)
        if rmax is not None:
            rt, rl = stratified_downsample(rt, rl, rmax, seed=args.seed + 2)
        n0 = int((rl == 0).sum())
        n1 = int((rl == 1).sum())
        if n0 < 2 or n1 < 2:
            raise SystemExit(
                f"Report split {rpt!r} has too few rows per class after subsample "
                f"(LLM={n0}, human={n1}). Increase --report-max-samples or use full split (0)."
            )
        report_texts, report_labels = rt, rl

    results: Dict[str, Dict] = {}
    val_payloads: Dict[str, Dict[str, np.ndarray]] = {}
    roc_series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    if not args.skip_one_class:
        results["one_class"] = train_one_setting(
            "one_class",
            nu=args.nu,
            warmup_epochs=args.warmup_epochs,
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            report_texts=report_texts,
            report_labels=report_labels,
            out_dir=out_dir,
            sbert_name=args.sbert_name,
            out_dim=args.out_dim,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            target_tpr=args.target_tpr,
            device=device,
        )
        val_payloads["one_class"] = {
            "scores": results["one_class"]["val_scores"],
            "y_true": results["one_class"]["val_y"],
        }
        roc_series["one_class"] = (
            results["one_class"]["val_y"],
            results["one_class"]["val_scores"],
        )

    if not args.skip_soft_boundary:
        tag_sb = f"soft_boundary_nu{args.nu}"
        results[tag_sb] = train_one_setting(
            "soft_boundary",
            nu=args.nu,
            warmup_epochs=args.warmup_epochs,
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            report_texts=report_texts,
            report_labels=report_labels,
            out_dir=out_dir,
            sbert_name=args.sbert_name,
            out_dim=args.out_dim,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            target_tpr=args.target_tpr,
            device=device,
        )
        val_payloads[tag_sb] = {
            "scores": results[tag_sb]["val_scores"],
            "y_true": results[tag_sb]["val_y"],
        }
        roc_series[tag_sb] = (
            results[tag_sb]["val_y"],
            results[tag_sb]["val_scores"],
        )

    nu_list: List[float] = []
    fpr_list: List[float] = []
    if args.nu_sweep.strip():
        for part in args.nu_sweep.split(","):
            nu = float(part.strip())
            tag = f"soft_boundary_nu{nu}"
            if tag in results:
                nu_list.append(nu)
                fpr_list.append(results[tag]["meta"]["best_metrics"]["fpr_at_target_tpr"])
                continue
            results[tag] = train_one_setting(
                "soft_boundary",
                nu=nu,
                warmup_epochs=args.warmup_epochs,
                train_texts=train_texts,
                train_labels=train_labels,
                val_texts=val_texts,
                val_labels=val_labels,
                report_texts=report_texts,
                report_labels=report_labels,
                out_dir=out_dir,
                sbert_name=args.sbert_name,
                out_dim=args.out_dim,
                lr=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                target_tpr=args.target_tpr,
                device=device,
            )
            nu_list.append(nu)
            fpr_list.append(results[tag]["meta"]["best_metrics"]["fpr_at_target_tpr"])

    summary = {
        "args": vars(args),
        "runs": {k: v["meta"] for k, v in results.items()},
    }
    with open(out_dir / "experiment_c_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    nu_bundle = None
    if nu_list:
        nu_bundle = (nu_list, fpr_list, args.target_tpr)

    pca_payloads = None
    if val_payloads and not args.no_pca_figure:
        pca_payloads = {k: (v["val_z"], v["val_y"]) for k, v in results.items()}

    if val_payloads:
        save_figure_bundle(
            out_dir,
            val_payloads,
            roc_series,
            nu_bundle,
            pr_series=roc_series,
            pca_payloads=pca_payloads,
        )

    # --- Console: emphasize operational metrics ---
    print("\n=== Experiment C (best epoch by ROC-AUC) ===\n")
    for name, pack in results.items():
        b = pack["meta"]["best_metrics"]
        print(f"{name}:")
        print(
            f"  score LLM:   mean={b['score_mean_llm']:.4f}  std={b['score_std_llm']:.4f}"
        )
        print(
            f"  score human: mean={b['score_mean_human']:.4f}  std={b['score_std_human']:.4f}"
        )
        print(
            f"  FPR @ TPR={args.target_tpr}: {b['fpr_at_target_tpr']:.4f}  "
            f"ROC-AUC={b['roc_auc']:.4f}  PR-AUC={b['pr_auc']:.4f}"
        )
        print(
            f"  F1 (quantile grid)={b['best_f1']:.4f}  thr={b['best_f1_threshold']:.4f}"
        )
        print(
            f"  at that thr:  Acc={b['accuracy_at_best_f1']:.4f}  "
            f"Prec={b['precision_at_best_f1']:.4f}  Rec={b['recall_at_best_f1']:.4f}"
        )
        rpt = pack["meta"].get("report_split_metrics")
        if rpt:
            print(
                f"  held-out {args.report_split!r}:  ROC-AUC={rpt['roc_auc']:.4f}  "
                f"PR-AUC={rpt['pr_auc']:.4f}  FPR@TPR={rpt['fpr_at_target_tpr']:.4f}"
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
