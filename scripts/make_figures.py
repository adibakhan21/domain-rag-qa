#!/usr/bin/env python
"""Generate every figure in the README from the saved result JSON files.

    python scripts/make_figures.py

Reads only from results/ -- if an experiment has not been run, its figure is
skipped with a warning rather than drawn from placeholder data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")                                 # headless: no display needed
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402

from rag_system.utils.config import RESULTS_DIR       # noqa: E402
from rag_system.utils.runtime import get_logger, read_json  # noqa: E402

LOGGER = get_logger()

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#9D755D"]
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _load(path: Path) -> Optional[dict]:
    if not path.exists():
        LOGGER.warning("skipping figure: %s not found", path)
        return None
    return read_json(path)


def fig_retrieval_comparison(out_dir: Path) -> None:
    data = _load(RESULTS_DIR / "retrieval" / "retrieval_results.json")
    if not data:
        return
    systems = list(data["systems"])
    labels = [s.replace("_", "\n") for s in systems]
    metrics = ["hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.16
    x = np.arange(len(systems))
    for i, metric in enumerate(metrics):
        values = [data["systems"][s]["metrics"][metric] for s in systems]
        errs = None
        if "ci95" in data["systems"][systems[0]]["metrics"]:
            lo = [data["systems"][s]["metrics"]["ci95"][metric][0] for s in systems]
            hi = [data["systems"][s]["metrics"]["ci95"][metric][1] for s in systems]
            errs = [np.array(values) - np.array(lo), np.array(hi) - np.array(values)]
        ax1.bar(x + (i - 2) * width, values, width, label=metric,
                color=PALETTE[i % len(PALETTE)], yerr=errs, capsize=2, error_kw={"lw": 0.7})
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_ylabel("score"); ax1.set_ylim(0, 1.0)
    ax1.set_title("Retrieval quality on the test split (95% CI)")
    ax1.legend(fontsize=7, ncol=5, loc="upper left")

    # Accuracy vs latency: the trade-off the reranker actually forces.
    lat = [data["systems"][s]["timings"]["ms_per_query"] for s in systems]
    hit5 = [data["systems"][s]["metrics"]["hit@5"] for s in systems]
    for i, s in enumerate(systems):
        ax2.scatter(lat[i], hit5[i], s=90, color=PALETTE[i % len(PALETTE)], zorder=3)
        ax2.annotate(s, (lat[i], hit5[i]), textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("latency (ms/query, batched, log scale)")
    ax2.set_ylabel("hit@5")
    ax2.set_title("Accuracy vs latency")
    fig.tight_layout()
    fig.savefig(out_dir / "retrieval_comparison.png")
    plt.close(fig)
    LOGGER.info("wrote retrieval_comparison.png")


def fig_ablations(out_dir: Path) -> None:
    chunking = _load(RESULTS_DIR / "retrieval" / "ablation_chunking.json")
    embedding = _load(RESULTS_DIR / "retrieval" / "ablation_embedding.json")
    if not chunking and not embedding:
        return

    n_panels = int(bool(chunking)) * 2 + int(bool(embedding))
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.8))
    axes = np.atleast_1d(axes)
    panel = 0

    if chunking:
        cells = chunking["cells"]
        sizes = sorted(int(k.split("_")[1]) for k in cells)
        ax = axes[panel]; panel += 1
        for i, system in enumerate(["bm25", "dense", "hybrid_rrf"]):
            ys = [cells[f"chunk_{s}"]["systems"][system]["metrics"]["hit@5"] for s in sizes]
            ax.plot(sizes, ys, "o-", label=system, color=PALETTE[i], lw=1.8)
        ax.set_xlabel("chunk size (characters)"); ax.set_ylabel("hit@5")
        ax.set_title("Chunk size vs retrieval")
        ax.legend(fontsize=7.5)

        ax = axes[panel]; panel += 1
        counts = [cells[f"chunk_{s}"]["chunk_stats"]["n_chunks"] for s in sizes]
        ax.bar([str(s) for s in sizes], counts, color=PALETTE[4])
        ax.set_xlabel("chunk size (characters)"); ax.set_ylabel("chunks in corpus")
        ax.set_title("Corpus granularity")
        for i, c in enumerate(counts):
            ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=7.5)

    if embedding:
        cells = embedding["cells"]
        names = list(cells)
        ax = axes[panel]
        x = np.arange(len(names)); width = 0.35
        for i, metric in enumerate(["hit@1", "hit@5"]):
            ax.bar(x + (i - 0.5) * width,
                   [cells[n]["systems"]["dense"]["metrics"][metric] for n in names],
                   width, label=metric, color=PALETTE[i])
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("-", "\n") for n in names], fontsize=7)
        ax.set_ylabel("score"); ax.set_title("Dense retrieval: embedding model")
        ax.legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out_dir / "ablations.png")
    plt.close(fig)
    LOGGER.info("wrote ablations.png")


def fig_finetuning(out_dir: Path) -> None:
    data = _load(RESULTS_DIR / "finetuning" / "finetuning_results.json")
    if not data:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    m = data["metrics"]
    groups = ["val EM", "val F1", "test EM", "test F1"]
    base = [m["base"]["validation"]["exact_match"], m["base"]["validation"]["f1"],
            m["base"]["test"]["exact_match"], m["base"]["test"]["f1"]]
    ft = [m["finetuned"]["validation"]["exact_match"], m["finetuned"]["validation"]["f1"],
          m["finetuned"]["test"]["exact_match"], m["finetuned"]["test"]["f1"]]
    x = np.arange(len(groups)); width = 0.36
    ax1.bar(x - width / 2, base, width, label="base (SQuAD2, zero-shot)", color=PALETTE[0])
    ax1.bar(x + width / 2, ft, width, label="LoRA fine-tuned", color=PALETTE[2])
    for xi, (b, f) in enumerate(zip(base, ft)):
        ax1.text(xi - width / 2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=7)
        ax1.text(xi + width / 2, f, f"{f:.1f}", ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(x); ax1.set_xticklabels(groups)
    ax1.set_ylabel("score"); ax1.set_ylim(0, 100)
    ax1.set_title("Extractive reader: base vs LoRA")
    ax1.legend(fontsize=7.5, loc="upper left")

    history = data["training"]["log_history"]
    train_pts = [(h["epoch"], h["loss"]) for h in history if "loss" in h and "eval_loss" not in h]
    eval_pts = [(h["epoch"], h["eval_loss"]) for h in history if "eval_loss" in h]
    if train_pts:
        ax2.plot(*zip(*train_pts), "o-", label="train loss", color=PALETTE[0], ms=3, lw=1.5)
    if eval_pts:
        ax2.plot(*zip(*eval_pts), "s-", label="validation loss", color=PALETTE[3], ms=5, lw=1.5)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("loss")
    ax2.set_title("LoRA training curve")
    ax2.legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out_dir / "finetuning.png")
    plt.close(fig)
    LOGGER.info("wrote finetuning.png")


def fig_rag_comparison(out_dir: Path) -> None:
    data = _load(RESULTS_DIR / "rag" / "rag_results.json")
    if not data:
        return
    order = ["closed_book", "rag_bm25", "rag_hybrid_rerank", "rag_hybrid_lora",
             "oracle_context", "oracle_context_lora"]
    # Any system present but not in `order` is appended, so a new system is
    # never silently dropped from the figure.
    systems = [s for s in order if s in data["systems"]]
    systems += [s for s in data["systems"] if s not in order]
    labels = [s.replace("_", "\n") for s in systems]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(systems)); width = 0.38
    em = [data["systems"][s]["exact_match"] for s in systems]
    f1 = [data["systems"][s]["f1"] for s in systems]
    ax1.bar(x - width / 2, em, width, label="Exact Match", color=PALETTE[0])
    ax1.bar(x + width / 2, f1, width, label="token F1", color=PALETTE[2])
    for xi, (a, b) in enumerate(zip(em, f1)):
        ax1.text(xi - width / 2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=7)
        ax1.text(xi + width / 2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_ylabel("score"); ax1.set_ylim(0, 100)
    ax1.set_title("End-to-end answer quality")
    ax1.legend(fontsize=7.5)

    ctx = [data["systems"][s]["answer_in_context"] for s in systems]
    gold = [data["systems"][s]["gold_chunk_retrieved"] for s in systems]
    ax2.bar(x - width / 2, gold, width, label="gold chunk retrieved", color=PALETTE[1])
    ax2.bar(x + width / 2, ctx, width, label="answer present in context", color=PALETTE[4])
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_ylabel("% of questions"); ax2.set_ylim(0, 105)
    ax2.set_title("Retrieval grounding")
    ax2.legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out_dir / "rag_comparison.png")
    plt.close(fig)
    LOGGER.info("wrote rag_comparison.png")


def fig_errors(out_dir: Path) -> None:
    data = _load(RESULTS_DIR / "error_analysis" / "error_analysis.json")
    if not data:
        return
    report = data["report"]
    cats = [(c, n) for c, n in report["counts"].items() if c != "correct" and n > 0]
    if not cats:
        return
    cats.sort(key=lambda kv: -kv[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax1.barh([c for c, _ in cats][::-1], [n for _, n in cats][::-1],
             color=[PALETTE[i % len(PALETTE)] for i in range(len(cats))][::-1])
    ax1.set_xlabel("questions"); ax1.set_title(f"Failure modes — {data['system']}")
    for i, (_, n) in enumerate(cats[::-1]):
        ax1.text(n, i, f" {n}", va="center", fontsize=7.5)

    lengths = data["by_answer_length"]
    ax2.bar(list(lengths), [v["mean_f1"] for v in lengths.values()], color=PALETTE[0])
    for i, (label, v) in enumerate(lengths.items()):
        ax2.text(i, v["mean_f1"], f"{v['mean_f1']:.0f}\n(n={v['n']})",
                 ha="center", va="bottom", fontsize=7)
    ax2.set_ylabel("mean F1"); ax2.set_ylim(0, 105)
    ax2.set_title("F1 by gold answer length")
    ax2.tick_params(axis="x", labelsize=7.5)

    fig.tight_layout()
    fig.savefig(out_dir / "error_analysis.png")
    plt.close(fig)
    LOGGER.info("wrote error_analysis.png")


def fig_latency(out_dir: Path) -> None:
    data = _load(RESULTS_DIR / "benchmarks" / "benchmark_results.json")
    if not data:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))

    comps = data["components"]
    names = list(comps)
    x = np.arange(len(names)); width = 0.38
    ax1.bar(x - width / 2, [comps[n]["p50_ms"] for n in names], width, label="p50", color=PALETTE[0])
    ax1.bar(x + width / 2, [comps[n]["p95_ms"] for n in names], width, label="p95", color=PALETTE[3])
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax1.set_ylabel("ms"); ax1.set_yscale("log")
    ax1.set_title(f"Per-stage latency, batch=1 ({data['environment']['device']})")
    ax1.legend(fontsize=7.5)

    tp = data["throughput"]
    sizes = [tp[k]["batch_size"] for k in tp]
    qps = [tp[k]["queries_per_second"] for k in tp]
    ax2.plot(sizes, qps, "o-", color=PALETTE[2], lw=1.8)
    for s, q in zip(sizes, qps):
        ax2.annotate(f"{q:.1f}", (s, q), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=7.5)
    ax2.set_xlabel("batch size"); ax2.set_ylabel("queries / second")
    ax2.set_title("Throughput (hybrid + rerank)")

    fig.tight_layout()
    fig.savefig(out_dir / "latency.png")
    plt.close(fig)
    LOGGER.info("wrote latency.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "figures")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for fn in (fig_retrieval_comparison, fig_ablations, fig_finetuning,
               fig_rag_comparison, fig_errors, fig_latency):
        try:
            fn(args.out_dir)
        except Exception as exc:  # a broken figure must not kill the rest
            LOGGER.warning("%s failed: %s: %s", fn.__name__, type(exc).__name__, exc)
    LOGGER.info("figures in %s", args.out_dir)


if __name__ == "__main__":
    main()
